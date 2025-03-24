"""Machine learning integration for alert correlation and prediction."""

import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pickle

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from .test_costbenefit_alerts import Alert, AlertThreshold, AlertManager
from .test_costbenefit_correlation import AlertPattern, AlertCorrelator

@dataclass
class MLConfig:
    """Configuration for machine learning models."""
    enable_anomaly_detection: bool = True
    enable_pattern_prediction: bool = True
    enable_root_cause_analysis: bool = True
    min_training_samples: int = 1000
    retraining_interval: int = 3600  # 1 hour
    model_save_path: str = "ml_models"
    feature_importance_threshold: float = 0.05
    anomaly_contamination: float = 0.1
    prediction_horizon: int = 300  # 5 minutes
    confidence_threshold: float = 0.8

class MLPredictor:
    """Machine learning predictor for alerts and patterns."""

    def __init__(
        self,
        config: MLConfig,
        correlator: AlertCorrelator
    ):
        self.config = config
        self.correlator = correlator
        self.scaler = StandardScaler()
        
        # Models
        self.pattern_classifier: Optional[RandomForestClassifier] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.sequence_model: Optional[tf.keras.Model] = None
        
        # State
        self.last_training_time = datetime.min
        self.feature_cache: List[np.ndarray] = []
        self.label_cache: List[str] = []
        
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize ML models."""
        if self.config.enable_pattern_prediction:
            self.pattern_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        if self.config.enable_anomaly_detection:
            self.anomaly_detector = IsolationForest(
                contamination=self.config.anomaly_contamination,
                random_state=42
            )
        
        if self.config.enable_root_cause_analysis:
            self.sequence_model = self._build_sequence_model()

    def _build_sequence_model(self) -> tf.keras.Model:
        """Build sequence model for root cause analysis."""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(len(self.correlator.rules), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    async def process_alert(self, alert: Alert) -> Dict[str, Any]:
        """Process new alert through ML pipeline."""
        # Extract features
        features = self._extract_alert_features(alert)
        
        results = {}
        
        # Run anomaly detection
        if self.config.enable_anomaly_detection and self.anomaly_detector:
            is_anomaly = self.anomaly_detector.predict([features])[0] == -1
            results["anomaly_score"] = float(is_anomaly)
        
        # Run pattern prediction
        if self.config.enable_pattern_prediction and self.pattern_classifier:
            pattern_probs = self.pattern_classifier.predict_proba([features])[0]
            max_prob_idx = np.argmax(pattern_probs)
            if pattern_probs[max_prob_idx] >= self.config.confidence_threshold:
                results["predicted_pattern"] = self.pattern_classifier.classes_[max_prob_idx]
                results["pattern_confidence"] = float(pattern_probs[max_prob_idx])
        
        # Cache features for training
        self.feature_cache.append(features)
        self.label_cache.append(alert.threshold.metric_name)
        
        # Check if retraining is needed
        await self._check_retraining()
        
        return results

    def _extract_alert_features(self, alert: Alert) -> np.ndarray:
        """Extract ML features from alert."""
        features = [
            alert.current_value,
            alert.threshold_value,
            float(alert.severity == "critical"),
            alert.duration,
            (datetime.now() - alert.start_time).total_seconds()
        ]
        
        # Add rule-based features
        for rule in self.correlator.rules.values():
            features.append(float(alert.threshold.metric_name in rule.metrics))
        
        return np.array(features)

    async def _check_retraining(self) -> None:
        """Check and perform model retraining if needed."""
        current_time = datetime.now()
        
        if (
            len(self.feature_cache) >= self.config.min_training_samples and
            (current_time - self.last_training_time).total_seconds() >= self.config.retraining_interval
        ):
            await self.train_models()

    async def train_models(self) -> Dict[str, float]:
        """Train all enabled ML models."""
        if len(self.feature_cache) < self.config.min_training_samples:
            return {}
        
        X = np.array(self.feature_cache)
        y = np.array(self.label_cache)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = {}
        
        # Train pattern classifier
        if self.config.enable_pattern_prediction and self.pattern_classifier:
            self.pattern_classifier.fit(X_train, y_train)
            accuracy = self.pattern_classifier.score(X_test, y_test)
            results["pattern_accuracy"] = accuracy
        
        # Train anomaly detector
        if self.config.enable_anomaly_detection and self.anomaly_detector:
            self.anomaly_detector.fit(X_train)
            results["anomaly_samples"] = len(X_train)
        
        # Train sequence model
        if self.config.enable_root_cause_analysis and self.sequence_model:
            # Prepare sequence data
            sequence_length = 10
            X_seq, y_seq = self._prepare_sequences(X, y, sequence_length)
            
            if len(X_seq) > 0:
                history = self.sequence_model.fit(
                    X_seq, y_seq,
                    epochs=10,
                    validation_split=0.2,
                    verbose=0
                )
                results["sequence_accuracy"] = float(history.history['accuracy'][-1])
        
        self.last_training_time = datetime.now()
        return results

    def _prepare_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for training."""
        X_seq = []
        y_seq = []
        
        for i in range(len(features) - sequence_length):
            X_seq.append(features[i:i + sequence_length])
            y_seq.append(self._encode_label(labels[i + sequence_length]))
        
        return np.array(X_seq), np.array(y_seq)

    def _encode_label(self, label: str) -> np.ndarray:
        """One-hot encode label."""
        encoding = np.zeros(len(self.correlator.rules))
        for i, rule_id in enumerate(self.correlator.rules.keys()):
            if label in self.correlator.rules[rule_id].metrics:
                encoding[i] = 1
        return encoding

    def save_models(self) -> None:
        """Save ML models to disk."""
        if not self.config.model_save_path:
            return
        
        models = {
            "pattern_classifier": self.pattern_classifier,
            "anomaly_detector": self.anomaly_detector,
            "scaler": self.scaler
        }
        
        for name, model in models.items():
            if model:
                with open(f"{self.config.model_save_path}/{name}.pkl", "wb") as f:
                    pickle.dump(model, f)
        
        if self.sequence_model:
            self.sequence_model.save(f"{self.config.model_save_path}/sequence_model")

    def load_models(self) -> None:
        """Load ML models from disk."""
        try:
            with open(f"{self.config.model_save_path}/pattern_classifier.pkl", "rb") as f:
                self.pattern_classifier = pickle.load(f)
            
            with open(f"{self.config.model_save_path}/anomaly_detector.pkl", "rb") as f:
                self.anomaly_detector = pickle.load(f)
            
            with open(f"{self.config.model_save_path}/scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            
            if self.config.enable_root_cause_analysis:
                self.sequence_model = tf.keras.models.load_model(
                    f"{self.config.model_save_path}/sequence_model"
                )
        
        except Exception as e:
            print(f"Error loading models: {e}")
            self._initialize_models()

@pytest.fixture
def ml_predictor(alert_correlator):
    """Create ML predictor for testing."""
    config = MLConfig(
        enable_anomaly_detection=True,
        enable_pattern_prediction=True,
        enable_root_cause_analysis=True,
        min_training_samples=10  # Small for testing
    )
    return MLPredictor(config, alert_correlator)

@pytest.mark.asyncio
async def test_alert_prediction(ml_predictor):
    """Test alert prediction."""
    # Generate training data
    alerts = []
    base_time = datetime.now()
    
    for i in range(20):
        alerts.append(Alert(
            alert_id=f"test_alert_{i}",
            threshold=AlertThreshold(
                metric_name="cpu_percent",
                warning_threshold=70,
                critical_threshold=90
            ),
            current_value=75 + i,
            threshold_value=70,
            severity="warning",
            start_time=base_time + timedelta(seconds=i * 30),
            duration=float(i * 30),
            message=f"Test alert {i}"
        ))
    
    # Train models
    for alert in alerts[:15]:  # Use first 15 for training
        await ml_predictor.process_alert(alert)
    
    await ml_predictor.train_models()
    
    # Test prediction
    for alert in alerts[15:]:  # Use last 5 for testing
        results = await ml_predictor.process_alert(alert)
        assert "anomaly_score" in results
        if "predicted_pattern" in results:
            assert results["pattern_confidence"] >= ml_predictor.config.confidence_threshold

@pytest.mark.asyncio
async def test_model_persistence(ml_predictor, tmp_path):
    """Test model saving and loading."""
    # Set temporary save path
    ml_predictor.config.model_save_path = str(tmp_path)
    
    # Train models with some data
    alert = Alert(
        alert_id="test_alert",
        threshold=AlertThreshold(
            metric_name="cpu_percent",
            warning_threshold=70,
            critical_threshold=90
        ),
        current_value=85,
        threshold_value=70,
        severity="warning",
        start_time=datetime.now(),
        duration=0.0,
        message="Test alert"
    )
    
    for _ in range(ml_predictor.config.min_training_samples):
        await ml_predictor.process_alert(alert)
    
    await ml_predictor.train_models()
    
    # Save models
    ml_predictor.save_models()
    
    # Create new predictor and load models
    new_predictor = MLPredictor(ml_predictor.config, ml_predictor.correlator)
    new_predictor.load_models()
    
    # Verify loaded models
    assert new_predictor.pattern_classifier is not None
    assert new_predictor.anomaly_detector is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
