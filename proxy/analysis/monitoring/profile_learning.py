"""Machine learning analysis for profiling data."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
import xgboost as xgb

from .profile_analysis import ProfileAnalyzer, AnalysisConfig

logger = logging.getLogger(__name__)

@dataclass
class LearningConfig:
    """Configuration for machine learning analysis."""
    test_size: float = 0.2
    n_estimators: int = 100
    max_depth: Optional[int] = None
    cv_folds: int = 5
    lstm_units: int = 64
    dropout_rate: float = 0.2
    epochs: int = 100
    batch_size: int = 32
    patience: int = 10
    output_path: Optional[Path] = None

class ProfileLearning:
    """Machine learning analysis for profiles."""
    
    def __init__(
        self,
        analyzer: ProfileAnalyzer,
        config: LearningConfig
    ):
        self.analyzer = analyzer
        self.config = config
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.training_history: Dict[str, Any] = {}
    
    def train_performance_model(
        self,
        metric: str
    ) -> Dict[str, Any]:
        """Train model to predict performance metrics."""
        # Prepare data
        data = self._prepare_training_data(metric)
        if data is None:
            return {}
        
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[metric] = scaler
        
        # Train models
        models = {
            "rf": self._train_random_forest(X_train_scaled, y_train),
            "xgb": self._train_xgboost(X_train_scaled, y_train),
            "nn": self._train_neural_network(X_train_scaled, y_train),
            "lstm": self._train_lstm(X_train_scaled, y_train)
        }
        
        # Evaluate models
        results = {}
        for name, model in models.items():
            predictions = self._predict_with_model(
                model,
                X_test_scaled,
                name == "lstm"
            )
            results[name] = self._evaluate_model(predictions, y_test)
        
        # Store best model
        best_model = max(results.items(), key=lambda x: x[1]["r2"])
        self.models[metric] = models[best_model[0]]
        
        return {
            "metric": metric,
            "results": results,
            "best_model": best_model[0],
            "feature_importance": self._get_feature_importance(
                models,
                best_model[0]
            ),
            "cross_validation": self._cross_validate_model(
                models[best_model[0]],
                X,
                y,
                best_model[0]
            )
        }
    
    def analyze_patterns(
        self,
        metric: str
    ) -> Dict[str, Any]:
        """Analyze complex patterns using ML."""
        # Prepare data
        data = self._prepare_training_data(metric)
        if data is None:
            return {}
        
        X, y = data
        
        patterns = {
            "anomalies": self._detect_complex_anomalies(X),
            "clusters": self._analyze_complex_clusters(X),
            "relationships": self._analyze_nonlinear_relationships(X, y),
            "importance": self._analyze_feature_importance(X, y),
            "interactions": self._analyze_feature_interactions(X, y)
        }
        
        return patterns
    
    def predict_performance(
        self,
        metric: str,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Predict performance metrics."""
        if metric not in self.models or metric not in self.scalers:
            return {}
        
        # Prepare input
        X = pd.DataFrame([features])
        X_scaled = self.scalers[metric].transform(X)
        
        # Make prediction
        model = self.models[metric]
        prediction = self._predict_with_model(
            model,
            X_scaled,
            isinstance(model, tf.keras.Model)
        )
        
        return {
            "prediction": float(prediction[0]),
            "confidence": self._estimate_prediction_confidence(
                model,
                X_scaled,
                metric
            )
        }
    
    def _prepare_training_data(
        self,
        metric: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Prepare data for training."""
        stats = self.analyzer.profiler.function_stats
        if not stats:
            return None
        
        # Extract features and target
        data = []
        targets = []
        
        for func_stats in stats.values():
            features = [
                func_stats.get("call_count", 0),
                func_stats.get("cumulative_time", 0),
                func_stats.get("memory_delta", 0),
                func_stats.get("callers", 0)
            ]
            target = func_stats.get(metric, 0)
            
            data.append(features)
            targets.append(target)
        
        return np.array(data), np.array(targets)
    
    def _train_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> RandomForestRegressor:
        """Train Random Forest model."""
        model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=42
        )
        model.fit(X, y)
        return model
    
    def _train_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> xgb.XGBRegressor:
        """Train XGBoost model."""
        model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth or 6,
            random_state=42
        )
        model.fit(X, y)
        return model
    
    def _train_neural_network(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> MLPRegressor:
        """Train Neural Network model."""
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=self.config.epochs,
            random_state=42
        )
        model.fit(X, y)
        return model
    
    def _train_lstm(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> tf.keras.Model:
        """Train LSTM model."""
        # Reshape input for LSTM
        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        
        model = Sequential([
            LSTM(self.config.lstm_units, input_shape=(1, X.shape[1])),
            Dropout(self.config.dropout_rate),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        
        model.compile(
            optimizer="adam",
            loss="mse"
        )
        
        # Add early stopping
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.config.patience
        )
        
        history = model.fit(
            X_reshaped,
            y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=0.2,
            callbacks=[callback],
            verbose=0
        )
        
        self.training_history["lstm"] = history.history
        return model
    
    def _predict_with_model(
        self,
        model: Any,
        X: np.ndarray,
        is_lstm: bool
    ) -> np.ndarray:
        """Make predictions with model."""
        if is_lstm:
            X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
            return model.predict(X_reshaped, verbose=0)
        return model.predict(X)
    
    def _evaluate_model(
        self,
        predictions: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        return {
            "mse": float(mean_squared_error(y_true, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
            "r2": float(r2_score(y_true, predictions))
        }
    
    def _get_feature_importance(
        self,
        models: Dict[str, Any],
        best_model: str
    ) -> Dict[str, List[float]]:
        """Get feature importance scores."""
        importance = {}
        model = models[best_model]
        
        if hasattr(model, "feature_importances_"):
            importance["importance"] = model.feature_importances_.tolist()
        
        return importance
    
    def _cross_validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str
    ) -> Dict[str, List[float]]:
        """Perform cross-validation."""
        if model_type == "lstm":
            return {}  # Skip cross-validation for LSTM
        
        scores = cross_validate(
            model,
            X,
            y,
            cv=self.config.cv_folds,
            scoring=["r2", "neg_mean_squared_error"]
        )
        
        return {
            "r2_scores": scores["test_r2"].tolist(),
            "mse_scores": (-scores["test_neg_mean_squared_error"]).tolist()
        }
    
    def _detect_complex_anomalies(
        self,
        X: np.ndarray
    ) -> Dict[str, Any]:
        """Detect complex anomalies using Isolation Forest."""
        model = IsolationForest(
            n_estimators=self.config.n_estimators,
            random_state=42
        )
        scores = model.fit_predict(X)
        
        return {
            "anomaly_indices": np.where(scores == -1)[0].tolist(),
            "anomaly_scores": model.score_samples(X).tolist()
        }
    
    def _analyze_complex_clusters(
        self,
        X: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze complex clustering patterns."""
        from sklearn.mixture import GaussianMixture
        
        model = GaussianMixture(n_components=3, random_state=42)
        labels = model.fit_predict(X)
        
        return {
            "cluster_labels": labels.tolist(),
            "cluster_probabilities": model.predict_proba(X).tolist(),
            "n_components": int(model.n_components_),
            "converged": bool(model.converged_)
        }
    
    def _analyze_nonlinear_relationships(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze nonlinear relationships."""
        from scipy.stats import spearmanr
        
        relationships = []
        for i in range(X.shape[1]):
            correlation, p_value = spearmanr(X[:, i], y)
            relationships.append({
                "feature_index": i,
                "correlation": float(correlation),
                "p_value": float(p_value)
            })
        
        return {
            "relationships": relationships,
            "significant": [
                r for r in relationships
                if r["p_value"] < 0.05
            ]
        }
    
    def _analyze_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze feature importance using multiple methods."""
        # Random Forest importance
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        # Permutation importance
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(
            rf, X, y, n_repeats=10, random_state=42
        )
        
        return {
            "random_forest": rf_importance.tolist(),
            "permutation": {
                "importances_mean": perm_importance.importances_mean.tolist(),
                "importances_std": perm_importance.importances_std.tolist()
            }
        }
    
    def _analyze_feature_interactions(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze feature interactions."""
        from sklearn.ensemble import RandomForestRegressor
        
        rf = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=3,
            random_state=42
        )
        rf.fit(X, y)
        
        interactions = []
        for tree in rf.estimators_:
            interactions.extend(
                self._extract_tree_interactions(tree, range(X.shape[1]))
            )
        
        return {
            "interactions": interactions,
            "strength": self._calculate_interaction_strength(interactions)
        }
    
    def _extract_tree_interactions(
        self,
        tree: Any,
        features: range
    ) -> List[Tuple[int, int]]:
        """Extract feature interactions from decision tree."""
        interactions = set()
        
        def traverse(node_id: int, path: set):
            if tree.tree_.children_left[node_id] == -1:  # Leaf node
                return
            
            feature = tree.tree_.feature[node_id]
            path.add(feature)
            
            # Add interactions between current feature and path features
            for prev_feature in path:
                if prev_feature != feature:
                    interactions.add(tuple(sorted([prev_feature, prev_feature])))
            
            traverse(tree.tree_.children_left[node_id], path.copy())
            traverse(tree.tree_.children_right[node_id], path.copy())
        
        traverse(0, set())
        return list(interactions)
    
    def _calculate_interaction_strength(
        self,
        interactions: List[Tuple[int, int]]
    ) -> Dict[str, float]:
        """Calculate interaction strength scores."""
        from collections import Counter
        
        counts = Counter(interactions)
        total = sum(counts.values())
        
        return {
            str(interaction): count / total
            for interaction, count in counts.items()
        }
    
    def save_models(self):
        """Save trained models."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save models and scalers
            for metric, model in self.models.items():
                if isinstance(model, tf.keras.Model):
                    model.save(str(output_path / f"{metric}_lstm"))
                else:
                    with open(output_path / f"{metric}_model.pkl", "wb") as f:
                        import pickle
                        pickle.dump(model, f)
                
                with open(output_path / f"{metric}_scaler.pkl", "wb") as f:
                    import pickle
                    pickle.dump(self.scalers[metric], f)
            
            logger.info(f"Saved models to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def load_models(self):
        """Load saved models."""
        if not self.config.output_path:
            return
        
        try:
            for model_path in self.config.output_path.glob("*_model.pkl"):
                metric = model_path.stem.replace("_model", "")
                
                with open(model_path, "rb") as f:
                    import pickle
                    self.models[metric] = pickle.load(f)
                
                scaler_path = model_path.parent / f"{metric}_scaler.pkl"
                if scaler_path.exists():
                    with open(scaler_path, "rb") as f:
                        self.scalers[metric] = pickle.load(f)
            
            # Load LSTM models
            for model_path in self.config.output_path.glob("*_lstm"):
                if model_path.is_dir():
                    metric = model_path.name.replace("_lstm", "")
                    self.models[metric] = tf.keras.models.load_model(
                        str(model_path)
                    )
            
            logger.info(f"Loaded models from {self.config.output_path}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")

def create_profile_learning(
    analyzer: ProfileAnalyzer,
    output_path: Optional[Path] = None
) -> ProfileLearning:
    """Create profile learning analyzer."""
    config = LearningConfig(output_path=output_path)
    return ProfileLearning(analyzer, config)

if __name__ == "__main__":
    # Example usage
    from .profile_analysis import create_profile_analyzer
    from .prediction_profiling import create_prediction_profiler
    from .prediction_performance import create_prediction_performance
    from .realtime_prediction import create_realtime_prediction
    from .prediction_controls import create_interactive_controls
    from .prediction_visualization import create_prediction_visualizer
    from .easing_prediction import create_easing_predictor
    from .easing_statistics import create_easing_statistics
    from .easing_metrics import create_easing_metrics
    from .easing_functions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    metrics = create_easing_metrics(easing)
    stats = create_easing_statistics(metrics)
    predictor = create_easing_predictor(stats)
    visualizer = create_prediction_visualizer(predictor)
    controls = create_interactive_controls(visualizer)
    realtime = create_realtime_prediction(controls)
    performance = create_prediction_performance(realtime)
    profiler = create_prediction_profiler(performance)
    analyzer = create_profile_analyzer(profiler)
    learning = create_profile_learning(
        analyzer,
        output_path=Path("profile_models")
    )
    
    # Train and evaluate models
    results = learning.train_performance_model("execution_time")
    print("Training results:", json.dumps(results, indent=2))
    
    # Analyze patterns
    patterns = learning.analyze_patterns("execution_time")
    print("\nPatterns:", json.dumps(patterns, indent=2))
    
    # Make predictions
    prediction = learning.predict_performance(
        "execution_time",
        {
            "call_count": 100,
            "cumulative_time": 0.5,
            "memory_delta": 1024,
            "callers": 3
        }
    )
    print("\nPrediction:", json.dumps(prediction, indent=2))
    
    # Save models
    learning.save_models()
