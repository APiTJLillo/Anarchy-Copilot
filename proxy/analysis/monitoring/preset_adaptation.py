"""Online adaptation for ensemble predictions."""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from river import ensemble, metrics, preprocessing

from .preset_ensemble import PresetEnsemble, EnsembleConfig
from .preset_predictions import PresetPredictor

@dataclass
class AdaptationConfig:
    """Configuration for online adaptation."""
    window_size: int = 100
    learning_rate: float = 0.1
    forgetting_factor: float = 0.95
    concept_drift_threshold: float = 0.2
    min_samples_drift: int = 30
    retraining_interval: int = 50
    max_online_models: int = 5
    enable_drift_detection: bool = True

@dataclass
class AdaptationMetrics:
    """Metrics for adaptation performance."""
    drift_detected: int = 0
    model_updates: int = 0
    online_error: float = 0.0
    batch_error: float = 0.0
    adaptation_time: float = 0.0
    drift_points: List[datetime] = field(default_factory=list)

class OnlineAdapter:
    """Online adaptation for ensemble predictions."""
    
    def __init__(
        self,
        ensemble: PresetEnsemble,
        config: AdaptationConfig = None
    ):
        self.ensemble = ensemble
        self.config = config or AdaptationConfig()
        
        # Online components
        self.online_models: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, AdaptationMetrics] = {}
        self.recent_errors: Dict[str, Deque[float]] = {}
        
        # Setup online learning
        self._initialize_online_models()
    
    def _initialize_online_models(self):
        """Initialize online learning models."""
        for preset_name in self.ensemble.models:
            # Create online models
            self.online_models[preset_name] = {
                # Adaptive Random Forest
                "arf": ensemble.AdaptiveRandomForestRegressor(
                    n_models=10,
                    drift_detector=True,
                    warning_detector=True,
                    drift_detection_criteria="error"
                ),
                # Streaming SGD
                "sgd": preprocessing.StandardScaler() | \
                       ensemble.SoftmaxRegressor(
                           optimizer="adam",
                           loss="squared"
                       ),
                # Hoeffding Tree
                "ht": ensemble.HoeffdingAdaptiveTreeRegressor(
                    grace_period=50,
                    split_confidence=1e-5,
                    leaf_prediction="adaptive"
                )
            }
            
            # Initialize metrics
            self.metrics[preset_name] = AdaptationMetrics()
            self.recent_errors[preset_name] = deque(maxlen=self.config.window_size)
    
    async def adapt_ensemble(
        self,
        preset_name: str,
        new_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, Any]:
        """Adapt ensemble to new data."""
        if preset_name not in self.online_models:
            return {"status": "no_online_models"}
        
        X, y = new_data
        metrics = self.metrics[preset_name]
        start_time = datetime.now()
        
        # Update online models
        for name, model in self.online_models[preset_name].items():
            # Learn from new data
            if isinstance(X, np.ndarray):
                X_dict = {f"f{i}": x for i, x in enumerate(X)}
            else:
                X_dict = X
            
            model.learn_one(X_dict, float(y))
        
        # Calculate online prediction error
        online_pred = self._get_online_prediction(preset_name, X)
        error = mean_squared_error([y], [online_pred])
        self.recent_errors[preset_name].append(error)
        
        # Check for concept drift
        if self.config.enable_drift_detection:
            drift_detected = self._detect_concept_drift(preset_name)
            if drift_detected:
                metrics.drift_detected += 1
                metrics.drift_points.append(datetime.now())
                
                # Trigger ensemble retraining
                await self._retrain_ensemble(preset_name)
        
        # Update adaptation metrics
        metrics.online_error = np.mean(list(self.recent_errors[preset_name]))
        metrics.adaptation_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "status": "success",
            "metrics": {
                "online_error": metrics.online_error,
                "drift_detected": metrics.drift_detected,
                "adaptation_time": metrics.adaptation_time
            }
        }
    
    def _get_online_prediction(
        self,
        preset_name: str,
        X: np.ndarray
    ) -> float:
        """Get prediction from online models."""
        predictions = []
        
        for model in self.online_models[preset_name].values():
            if isinstance(X, np.ndarray):
                X_dict = {f"f{i}": x for i, x in enumerate(X)}
            else:
                X_dict = X
            
            try:
                pred = model.predict_one(X_dict)
                predictions.append(pred)
            except Exception:
                continue
        
        if predictions:
            return np.mean(predictions)
        return 0.0
    
    def _detect_concept_drift(
        self,
        preset_name: str
    ) -> bool:
        """Detect concept drift in recent errors."""
        if len(self.recent_errors[preset_name]) < self.config.min_samples_drift:
            return False
        
        recent = list(self.recent_errors[preset_name])
        if len(recent) < 2:
            return False
        
        # Split recent errors into windows
        mid = len(recent) // 2
        window1 = recent[:mid]
        window2 = recent[mid:]
        
        # Compare error distributions
        mean1 = np.mean(window1)
        mean2 = np.mean(window2)
        std1 = np.std(window1) + 1e-10
        std2 = np.std(window2) + 1e-10
        
        # Calculate drift measure
        drift_measure = abs(mean1 - mean2) / min(std1, std2)
        
        return drift_measure > self.config.concept_drift_threshold
    
    async def _retrain_ensemble(
        self,
        preset_name: str
    ):
        """Retrain ensemble models."""
        # Get recent data
        preset = await self.ensemble.predictor.analytics.preset_manager.get_preset(
            preset_name
        )
        recent_results = [
            res for _, name, res in self.ensemble.predictor.analytics.history[-self.config.retraining_interval:]
            if name == preset_name
        ]
        
        if not recent_results:
            return
        
        # Prepare data
        X, y = self.ensemble.predictor._prepare_features(preset, recent_results)
        X_scaled = self.ensemble.predictor.scalers[preset_name].transform(X)
        
        # Retrain models
        for name, model in self.ensemble.models[preset_name].items():
            try:
                model.partial_fit(X_scaled, y)
            except AttributeError:
                # If partial_fit not available, do full retraining
                model.fit(X_scaled, y)
        
        # Update weights
        self.ensemble._update_weights(preset_name)
        
        # Update metrics
        self.metrics[preset_name].model_updates += 1
    
    async def create_adaptation_plots(
        self,
        preset_name: str
    ) -> Dict[str, go.Figure]:
        """Create adaptation visualization plots."""
        if preset_name not in self.metrics:
            return {}
        
        plots = {}
        metrics = self.metrics[preset_name]
        
        # Error tracking plot
        error_fig = go.Figure()
        
        errors = list(self.recent_errors[preset_name])
        error_fig.add_trace(
            go.Scatter(
                y=errors,
                mode="lines",
                name="Online Error"
            )
        )
        
        # Add drift points
        if metrics.drift_points:
            drift_indices = [
                i for i, _ in enumerate(errors)
                if any(
                    abs((datetime.now() - timedelta(seconds=i)) - dp).total_seconds() < 1
                    for dp in metrics.drift_points
                )
            ]
            
            error_fig.add_trace(
                go.Scatter(
                    x=drift_indices,
                    y=[errors[i] for i in drift_indices],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=12,
                        color="red"
                    ),
                    name="Drift Detected"
                )
            )
        
        error_fig.update_layout(
            title="Online Learning Error",
            yaxis_title="Error",
            showlegend=True
        )
        plots["error"] = error_fig
        
        # Model performance comparison
        if preset_name in self.ensemble.models:
            performance_fig = go.Figure()
            
            # Batch model performance
            batch_errors = [
                np.mean(self.ensemble.states[preset_name].errors[name])
                for name in self.ensemble.models[preset_name]
            ]
            
            performance_fig.add_trace(
                go.Bar(
                    x=list(self.ensemble.models[preset_name].keys()),
                    y=batch_errors,
                    name="Batch Error"
                )
            )
            
            # Online model performance
            online_errors = [
                np.mean([
                    e for e in self.recent_errors[preset_name]
                    if e is not None
                ])
                for _ in self.online_models[preset_name]
            ]
            
            performance_fig.add_trace(
                go.Bar(
                    x=list(self.online_models[preset_name].keys()),
                    y=online_errors,
                    name="Online Error"
                )
            )
            
            performance_fig.update_layout(
                title="Model Performance Comparison",
                barmode="group",
                xaxis_title="Model",
                yaxis_title="Error"
            )
            plots["performance"] = performance_fig
        
        return plots

def create_online_adapter(
    ensemble: PresetEnsemble,
    config: Optional[AdaptationConfig] = None
) -> OnlineAdapter:
    """Create online adapter."""
    return OnlineAdapter(ensemble, config)

if __name__ == "__main__":
    from .preset_ensemble import create_preset_ensemble
    from .preset_predictions import create_preset_predictor
    from .preset_analytics import create_preset_analytics
    from .mutation_presets import create_preset_manager
    
    async def main():
        # Setup components
        manager = create_preset_manager()
        analytics = create_preset_analytics(manager)
        predictor = create_preset_predictor(analytics)
        ensemble = create_preset_ensemble(predictor)
        adapter = create_online_adapter(ensemble)
        
        # Create test preset
        await manager.save_preset(
            "test_preset",
            "Test preset",
            {
                "operators": ["type_mutation"],
                "error_types": ["TypeError"],
                "score_range": [0.5, 1.0],
                "time_range": None
            }
        )
        
        # Generate streaming data
        for i in range(200):
            # Simulate concept drift
            if i == 100:
                drift_factor = 0.3
            else:
                drift_factor = 0.0
            
            X = np.random.rand(10)
            y = np.sum(X) + drift_factor * np.random.rand()
            
            # Adapt to new data
            result = await adapter.adapt_ensemble("test_preset", (X, y))
            print(f"Step {i}: {result}")
            
            if i % 50 == 0:
                plots = await adapter.create_adaptation_plots("test_preset")
                for name, fig in plots.items():
                    fig.write_html(f"test_adaptation_{name}_{i}.html")
    
    asyncio.run(main())
