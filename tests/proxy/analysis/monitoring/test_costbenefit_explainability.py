"""Model explainability for cost-benefit ML predictions."""

import asyncio
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

import pytest
import numpy as np
import shap
import lime
import lime.lime_tabular
import eli5
from sklearn.pipeline import Pipeline
from scipy import stats

from .test_costbenefit_ml import MLPredictor, MLConfig
from .test_costbenefit_alerts import Alert, AlertThreshold
from .test_costbenefit_correlation import AlertCorrelator

@dataclass
class ExplainabilityConfig:
    """Configuration for model explainability."""
    enable_shap: bool = True
    enable_lime: bool = True
    enable_feature_importance: bool = True
    num_background_samples: int = 100
    lime_num_features: int = 10
    min_feature_importance: float = 0.05
    explanation_cache_size: int = 1000
    store_explanations: bool = True
    explanation_output_path: Optional[str] = "explanations"

class ModelExplainer:
    """Explain ML model predictions."""

    def __init__(
        self,
        predictor: MLPredictor,
        config: ExplainabilityConfig
    ):
        self.predictor = predictor
        self.config = config
        
        # Explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Caches
        self.explanation_cache: Dict[str, Dict[str, Any]] = {}
        self.feature_importance_cache: Dict[str, np.ndarray] = {}
        
        self._initialize_explainers()

    def _initialize_explainers(self) -> None:
        """Initialize explanation models."""
        if self.config.enable_shap and self.predictor.pattern_classifier:
            # Create background dataset for SHAP
            background_data = np.array(
                self.predictor.feature_cache[:self.config.num_background_samples]
            )
            self.shap_explainer = shap.TreeExplainer(
                self.predictor.pattern_classifier,
                background_data
            )
        
        if self.config.enable_lime and self.predictor.pattern_classifier:
            # Initialize LIME with feature names
            feature_names = self._get_feature_names()
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(self.predictor.feature_cache),
                feature_names=feature_names,
                class_names=self.predictor.pattern_classifier.classes_,
                mode='classification'
            )

    def _get_feature_names(self) -> List[str]:
        """Get feature names for explanations."""
        base_features = [
            "current_value",
            "threshold_value",
            "is_critical",
            "duration",
            "time_since_start"
        ]
        
        rule_features = [
            f"rule_{rule_id}"
            for rule_id in self.predictor.correlator.rules.keys()
        ]
        
        return base_features + rule_features

    async def explain_prediction(
        self,
        alert: Alert,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate explanation for a prediction."""
        alert_id = alert.alert_id
        
        # Check cache first
        if alert_id in self.explanation_cache:
            return self.explanation_cache[alert_id]
        
        explanations = {}
        features = self.predictor._extract_alert_features(alert)
        
        # Generate SHAP explanation
        if self.config.enable_shap and self.shap_explainer:
            shap_values = self.shap_explainer.shap_values(features)
            explanations["shap"] = {
                "values": shap_values,
                "base_value": self.shap_explainer.expected_value
            }
            
            # Get feature importance from SHAP
            if self.config.enable_feature_importance:
                importance = np.abs(shap_values).mean(0)
                explanations["feature_importance"] = {
                    name: value
                    for name, value in zip(self._get_feature_names(), importance)
                    if value >= self.config.min_feature_importance
                }
        
        # Generate LIME explanation
        if self.config.enable_lime and self.lime_explainer:
            lime_exp = self.lime_explainer.explain_instance(
                features,
                self.predictor.pattern_classifier.predict_proba,
                num_features=self.config.lime_num_features
            )
            explanations["lime"] = {
                "local_exp": lime_exp.local_exp,
                "intercept": lime_exp.intercept,
                "score": lime_exp.score
            }
        
        # Add prediction context
        explanations["prediction"] = prediction
        explanations["feature_values"] = dict(
            zip(self._get_feature_names(), features)
        )
        
        # Cache explanation
        if self.config.store_explanations:
            self.explanation_cache[alert_id] = explanations
            if len(self.explanation_cache) > self.config.explanation_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.explanation_cache))
                del self.explanation_cache[oldest_key]
        
        return explanations

    def analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze global feature importance."""
        if not self.predictor.pattern_classifier:
            return {}
        
        results = {}
        
        # Get feature names
        feature_names = self._get_feature_names()
        
        # Random Forest feature importance
        if hasattr(self.predictor.pattern_classifier, "feature_importances_"):
            importance = self.predictor.pattern_classifier.feature_importances_
            results["random_forest"] = {
                name: value
                for name, value in zip(feature_names, importance)
                if value >= self.config.min_feature_importance
            }
        
        # Permutation importance
        if len(self.predictor.feature_cache) > 0:
            X = np.array(self.predictor.feature_cache)
            y = np.array(self.predictor.label_cache)
            
            from sklearn.inspection import permutation_importance
            perm_importance = permutation_importance(
                self.predictor.pattern_classifier,
                X, y,
                n_repeats=10,
                random_state=42
            )
            
            results["permutation"] = {
                name: value
                for name, value in zip(feature_names, perm_importance.importances_mean)
                if value >= self.config.min_feature_importance
            }
        
        # Add statistical correlations
        results["correlations"] = self._analyze_correlations()
        
        return results

    def _analyze_correlations(self) -> Dict[str, float]:
        """Analyze statistical correlations between features and predictions."""
        if len(self.predictor.feature_cache) == 0:
            return {}
        
        correlations = {}
        feature_names = self._get_feature_names()
        X = np.array(self.predictor.feature_cache)
        y = np.array(self.predictor.label_cache)
        
        for i, name in enumerate(feature_names):
            feature_values = X[:, i]
            correlation, p_value = stats.spearmanr(feature_values, y)
            if abs(correlation) >= self.config.min_feature_importance:
                correlations[name] = {
                    "correlation": correlation,
                    "p_value": p_value
                }
        
        return correlations

    async def save_explanations(self, alert_id: str) -> None:
        """Save explanations to disk."""
        if not self.config.explanation_output_path:
            return
        
        if alert_id not in self.explanation_cache:
            return
        
        import json
        from pathlib import Path
        
        explanation = self.explanation_cache[alert_id]
        output_path = Path(self.config.explanation_output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_exp = self._make_serializable(explanation)
        
        with open(output_path / f"{alert_id}_explanation.json", "w") as f:
            json.dump(serializable_exp, f, indent=2)

    def _make_serializable(self, obj: Any) -> Any:
        """Convert explanation objects to JSON serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            return str(obj)

@pytest.fixture
def model_explainer(ml_predictor):
    """Create model explainer for testing."""
    config = ExplainabilityConfig()
    return ModelExplainer(ml_predictor, config)

@pytest.mark.asyncio
async def test_shap_explanation(model_explainer, ml_predictor):
    """Test SHAP explanations."""
    # Train the model first
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
    
    # Train with multiple samples
    for i in range(20):
        modified_alert = Alert(
            alert_id=f"test_alert_{i}",
            threshold=alert.threshold,
            current_value=alert.current_value + i,
            threshold_value=alert.threshold_value,
            severity=alert.severity,
            start_time=alert.start_time,
            duration=float(i),
            message=f"Test alert {i}"
        )
        await ml_predictor.process_alert(modified_alert)
    
    await ml_predictor.train_models()
    
    # Get prediction and explanation
    prediction = await ml_predictor.process_alert(alert)
    explanation = await model_explainer.explain_prediction(alert, prediction)
    
    # Verify SHAP explanation
    assert "shap" in explanation
    assert "values" in explanation["shap"]
    assert "base_value" in explanation["shap"]

@pytest.mark.asyncio
async def test_feature_importance(model_explainer, ml_predictor):
    """Test feature importance analysis."""
    # Train the model first (similar to previous test)
    alerts = []
    for i in range(20):
        alert = Alert(
            alert_id=f"test_alert_{i}",
            threshold=AlertThreshold(
                metric_name="cpu_percent",
                warning_threshold=70,
                critical_threshold=90
            ),
            current_value=75 + i,
            threshold_value=70,
            severity="warning",
            start_time=datetime.now(),
            duration=float(i),
            message=f"Test alert {i}"
        )
        alerts.append(alert)
        await ml_predictor.process_alert(alert)
    
    await ml_predictor.train_models()
    
    # Get feature importance
    importance = model_explainer.analyze_feature_importance()
    
    # Verify results
    assert "random_forest" in importance
    assert len(importance["random_forest"]) > 0
    assert "correlations" in importance
    assert len(importance["correlations"]) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
