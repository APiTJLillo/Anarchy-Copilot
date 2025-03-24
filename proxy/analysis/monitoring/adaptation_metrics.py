"""Performance metrics for online adaptation."""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

from .preset_adaptation import OnlineAdapter, AdaptationConfig, AdaptationMetrics

@dataclass
class MetricsConfig:
    """Configuration for performance metrics."""
    window_sizes: List[int] = field(default_factory=lambda: [10, 50, 100])
    prequential_alpha: float = 0.1
    tracking_interval: int = 10
    significance_level: float = 0.05
    enable_statistical_tests: bool = True
    store_predictions: bool = True
    max_stored_predictions: int = 1000

@dataclass
class WindowedMetrics:
    """Windowed performance metrics."""
    mse: List[float] = field(default_factory=list)
    mae: List[float] = field(default_factory=list)
    r2: List[float] = field(default_factory=list)
    ev: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

@dataclass
class PrequentialMetrics:
    """Prequential performance metrics."""
    error: float = 0.0
    loss: float = 0.0
    count: int = 0
    drift_score: float = 0.0

class PerformanceTracker:
    """Track and analyze adaptation performance."""
    
    def __init__(
        self,
        adapter: OnlineAdapter,
        config: MetricsConfig = None
    ):
        self.adapter = adapter
        self.config = config or MetricsConfig()
        
        # Metric storage
        self.windowed: Dict[str, Dict[int, WindowedMetrics]] = {}
        self.prequential: Dict[str, PrequentialMetrics] = {}
        self.predictions: Dict[str, Deque[Tuple[float, float]]] = {}
        
        # Initialize storage
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize metric storage."""
        for preset_name in self.adapter.ensemble.models:
            # Initialize windowed metrics
            self.windowed[preset_name] = {
                size: WindowedMetrics()
                for size in self.config.window_sizes
            }
            
            # Initialize prequential metrics
            self.prequential[preset_name] = PrequentialMetrics()
            
            # Initialize prediction storage
            if self.config.store_predictions:
                self.predictions[preset_name] = deque(
                    maxlen=self.config.max_stored_predictions
                )
    
    async def update_metrics(
        self,
        preset_name: str,
        true_value: float,
        predicted_value: float
    ):
        """Update performance metrics."""
        timestamp = datetime.now()
        
        # Store prediction
        if self.config.store_predictions:
            self.predictions[preset_name].append((true_value, predicted_value))
        
        # Update windowed metrics
        for size, metrics in self.windowed[preset_name].items():
            recent_true = []
            recent_pred = []
            
            if self.config.store_predictions:
                recent = list(self.predictions[preset_name])[-size:]
                if recent:
                    recent_true, recent_pred = zip(*recent)
            
            if recent_true and recent_pred:
                metrics.mse.append(mean_squared_error(recent_true, recent_pred))
                metrics.mae.append(mean_absolute_error(recent_true, recent_pred))
                metrics.r2.append(r2_score(recent_true, recent_pred))
                metrics.ev.append(explained_variance_score(recent_true, recent_pred))
                metrics.timestamps.append(timestamp)
        
        # Update prequential metrics
        prequential = self.prequential[preset_name]
        alpha = self.config.prequential_alpha
        
        # Prequential error
        error = abs(true_value - predicted_value)
        prequential.error = (
            (1 - alpha) * prequential.error +
            alpha * error
        )
        
        # Prequential loss
        loss = (true_value - predicted_value) ** 2
        prequential.loss = (
            (1 - alpha) * prequential.loss +
            alpha * loss
        )
        
        prequential.count += 1
        
        # Update drift score
        if prequential.count > 1:
            prequential.drift_score = (
                prequential.error /
                (prequential.count ** 0.5)
            )
    
    def get_current_metrics(
        self,
        preset_name: str
    ) -> Dict[str, Any]:
        """Get current performance metrics."""
        if preset_name not in self.windowed:
            return {}
        
        metrics = {
            "windowed": {
                str(size): {
                    "mse": metrics.mse[-1] if metrics.mse else None,
                    "mae": metrics.mae[-1] if metrics.mae else None,
                    "r2": metrics.r2[-1] if metrics.r2 else None,
                    "ev": metrics.ev[-1] if metrics.ev else None
                }
                for size, metrics in self.windowed[preset_name].items()
            },
            "prequential": {
                "error": self.prequential[preset_name].error,
                "loss": self.prequential[preset_name].loss,
                "drift_score": self.prequential[preset_name].drift_score
            }
        }
        
        if self.config.enable_statistical_tests:
            metrics["statistical"] = self._calculate_statistical_metrics(preset_name)
        
        return metrics
    
    def _calculate_statistical_metrics(
        self,
        preset_name: str
    ) -> Dict[str, Any]:
        """Calculate statistical metrics."""
        if not self.config.store_predictions:
            return {}
        
        predictions = list(self.predictions[preset_name])
        if not predictions:
            return {}
        
        true_values, pred_values = zip(*predictions)
        
        # Calculate basic statistics
        stats = {
            "mean_error": np.mean(np.array(true_values) - np.array(pred_values)),
            "std_error": np.std(np.array(true_values) - np.array(pred_values)),
            "skewness": self._calculate_skewness(
                np.array(true_values) - np.array(pred_values)
            ),
            "kurtosis": self._calculate_kurtosis(
                np.array(true_values) - np.array(pred_values)
            )
        }
        
        # Calculate confidence intervals
        if len(predictions) >= 30:  # Central limit theorem
            confidence = 1 - self.config.significance_level
            z_score = {
                0.90: 1.645,
                0.95: 1.96,
                0.99: 2.576
            }.get(confidence, 1.96)
            
            std_error = stats["std_error"] / np.sqrt(len(predictions))
            stats["confidence_interval"] = (
                stats["mean_error"] - z_score * std_error,
                stats["mean_error"] + z_score * std_error
            )
        
        return stats
    
    def _calculate_skewness(
        self,
        data: np.ndarray
    ) -> float:
        """Calculate skewness of data."""
        if len(data) < 2:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(
        self,
        data: np.ndarray
    ) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 2:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3
    
    async def create_metric_plots(
        self,
        preset_name: str
    ) -> Dict[str, go.Figure]:
        """Create performance visualization plots."""
        if preset_name not in self.windowed:
            return {}
        
        plots = {}
        
        # Windowed metrics plot
        metrics_fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=["MSE", "MAE", "R²", "Explained Variance"]
        )
        
        for size, metrics in self.windowed[preset_name].items():
            # MSE
            metrics_fig.add_trace(
                go.Scatter(
                    x=metrics.timestamps,
                    y=metrics.mse,
                    name=f"Window {size}",
                    showlegend=True
                ),
                row=1,
                col=1
            )
            
            # MAE
            metrics_fig.add_trace(
                go.Scatter(
                    x=metrics.timestamps,
                    y=metrics.mae,
                    name=f"Window {size}",
                    showlegend=False
                ),
                row=1,
                col=2
            )
            
            # R²
            metrics_fig.add_trace(
                go.Scatter(
                    x=metrics.timestamps,
                    y=metrics.r2,
                    name=f"Window {size}",
                    showlegend=False
                ),
                row=2,
                col=1
            )
            
            # Explained Variance
            metrics_fig.add_trace(
                go.Scatter(
                    x=metrics.timestamps,
                    y=metrics.ev,
                    name=f"Window {size}",
                    showlegend=False
                ),
                row=2,
                col=2
            )
        
        metrics_fig.update_layout(
            height=800,
            title="Performance Metrics Over Time"
        )
        plots["metrics"] = metrics_fig
        
        # Prequential metrics plot
        prequential_fig = go.Figure()
        
        # Add error and loss
        prequential_fig.add_trace(
            go.Scatter(
                y=[self.prequential[preset_name].error],
                name="Prequential Error",
                mode="lines+markers"
            )
        )
        
        prequential_fig.add_trace(
            go.Scatter(
                y=[self.prequential[preset_name].loss],
                name="Prequential Loss",
                mode="lines+markers"
            )
        )
        
        prequential_fig.update_layout(
            title="Prequential Metrics",
            yaxis_title="Value"
        )
        plots["prequential"] = prequential_fig
        
        # Error distribution plot
        if self.config.store_predictions and self.predictions[preset_name]:
            true_values, pred_values = zip(*self.predictions[preset_name])
            errors = np.array(true_values) - np.array(pred_values)
            
            dist_fig = go.Figure()
            dist_fig.add_trace(
                go.Histogram(
                    x=errors,
                    nbinsx=30,
                    name="Error Distribution"
                )
            )
            
            if self.config.enable_statistical_tests:
                stats = self._calculate_statistical_metrics(preset_name)
                if "confidence_interval" in stats:
                    ci_low, ci_high = stats["confidence_interval"]
                    for x in [ci_low, ci_high]:
                        dist_fig.add_vline(
                            x=x,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"CI: {x:.2f}"
                        )
            
            dist_fig.update_layout(
                title="Error Distribution",
                xaxis_title="Prediction Error",
                yaxis_title="Count"
            )
            plots["distribution"] = dist_fig
        
        return plots

def create_performance_tracker(
    adapter: OnlineAdapter,
    config: Optional[MetricsConfig] = None
) -> PerformanceTracker:
    """Create performance tracker."""
    return PerformanceTracker(adapter, config)

if __name__ == "__main__":
    from .preset_adaptation import create_online_adapter
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
        tracker = create_performance_tracker(adapter)
        
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
        
        # Generate test data
        for i in range(100):
            X = np.random.rand(10)
            true_y = np.sum(X)
            pred_y = true_y + np.random.normal(0, 0.1)
            
            await tracker.update_metrics("test_preset", true_y, pred_y)
            
            if i % 20 == 0:
                metrics = tracker.get_current_metrics("test_preset")
                print(f"Step {i} metrics:", metrics)
                
                plots = await tracker.create_metric_plots("test_preset")
                for name, fig in plots.items():
                    fig.write_html(f"test_metrics_{name}_{i}.html")
    
    asyncio.run(main())
