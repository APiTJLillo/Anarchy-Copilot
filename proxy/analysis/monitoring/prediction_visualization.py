"""Visualization tools for prediction analysis."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime

from .easing_prediction import EasingPredictor, PredictionConfig

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for prediction visualization."""
    width: int = 1000
    height: int = 600
    dark_mode: bool = False
    show_confidence: bool = True
    show_components: bool = True
    animate_transitions: bool = True
    interactive: bool = True
    output_path: Optional[Path] = None

class PredictionVisualizer:
    """Visualize prediction results."""
    
    def __init__(
        self,
        predictor: EasingPredictor,
        config: VisualizationConfig
    ):
        self.predictor = predictor
        self.config = config
    
    def visualize_predictions(
        self,
        name: str,
        t: np.ndarray
    ) -> go.Figure:
        """Create comprehensive prediction visualization."""
        predictions = self.predictor.predict_behavior(name, t)
        
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Point Predictions",
                "Uncertainty Analysis",
                "Trend Decomposition",
                "Anomaly Detection"
            ]
        )
        
        # Add point predictions
        self._add_point_predictions(fig, t, predictions, row=1, col=1)
        
        # Add uncertainty analysis
        self._add_uncertainty_analysis(fig, t, predictions, row=1, col=2)
        
        # Add trend decomposition
        self._add_trend_decomposition(fig, t, predictions, row=2, col=1)
        
        # Add anomaly detection
        self._add_anomaly_detection(fig, t, predictions, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            showlegend=True
        )
        
        return fig
    
    def visualize_forecasts(
        self,
        name: str,
        steps: Optional[int] = None
    ) -> go.Figure:
        """Visualize metric forecasts."""
        forecasts = self.predictor.forecast_metrics(name, steps)
        
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Smoothness Forecast",
                "Efficiency Forecast",
                "Performance Metrics",
                "Forecast Confidence"
            ]
        )
        
        # Add metric forecasts
        self._add_metric_forecasts(fig, forecasts, row=1, col=1)
        
        # Add efficiency forecast
        self._add_efficiency_forecast(fig, forecasts, row=1, col=2)
        
        # Add performance metrics
        self._add_performance_metrics(fig, forecasts, row=2, col=1)
        
        # Add confidence intervals
        self._add_forecast_confidence(fig, forecasts, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            showlegend=True
        )
        
        return fig
    
    def visualize_variations(
        self,
        name: str,
        n_variations: Optional[int] = None
    ) -> go.Figure:
        """Visualize easing variations."""
        variations = self.predictor.simulate_variations(name, n_variations)
        
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Variation Distribution",
                "Quality Scores",
                "Parameter Space",
                "Stability Analysis"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter3d"}, {"type": "heatmap"}]
            ]
        )
        
        # Add variation distribution
        self._add_variation_distribution(fig, variations, row=1, col=1)
        
        # Add quality scores
        self._add_quality_scores(fig, variations, row=1, col=2)
        
        # Add parameter space visualization
        self._add_parameter_space(fig, variations, row=2, col=1)
        
        # Add stability analysis
        self._add_stability_analysis(fig, variations, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            showlegend=True
        )
        
        return fig
    
    def create_prediction_dashboard(
        self,
        name: str,
        t: np.ndarray
    ) -> go.Figure:
        """Create interactive prediction dashboard."""
        predictions = self.predictor.predict_behavior(name, t)
        forecasts = self.predictor.forecast_metrics(name)
        variations = self.predictor.simulate_variations(name)
        
        fig = make_subplots(
            rows=3,
            cols=3,
            subplot_titles=[
                "Predictions", "Uncertainty", "Trends",
                "Forecasts", "Variations", "Performance",
                "Anomalies", "Stability", "Quality"
            ]
        )
        
        # Add all visualizations
        self._add_dashboard_predictions(fig, t, predictions)
        self._add_dashboard_forecasts(fig, forecasts)
        self._add_dashboard_variations(fig, variations)
        
        # Add interactivity
        if self.config.interactive:
            self._add_dashboard_controls(fig)
        
        # Update layout
        fig.update_layout(
            width=self.config.width * 1.5,
            height=self.config.height * 1.5,
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            showlegend=True
        )
        
        return fig
    
    def _add_point_predictions(
        self,
        fig: go.Figure,
        t: np.ndarray,
        predictions: Dict[str, Any],
        row: int,
        col: int
    ):
        """Add point predictions subplot."""
        estimates = predictions["point_estimates"]
        
        # Add ensemble prediction
        fig.add_trace(
            go.Scatter(
                x=t,
                y=estimates["ensemble"],
                mode="lines",
                name="Ensemble Prediction",
                line=dict(color="blue", width=2)
            ),
            row=row,
            col=col
        )
        
        if self.config.show_components:
            # Add individual model predictions
            for model, pred in estimates.items():
                if model != "ensemble":
                    fig.add_trace(
                        go.Scatter(
                            x=t,
                            y=pred,
                            mode="lines",
                            name=f"{model.upper()} Prediction",
                            line=dict(dash="dot")
                        ),
                        row=row,
                        col=col
                    )
        
        fig.update_xaxes(title_text="Time", row=row, col=col)
        fig.update_yaxes(title_text="Value", row=row, col=col)
    
    def _add_uncertainty_analysis(
        self,
        fig: go.Figure,
        t: np.ndarray,
        predictions: Dict[str, Any],
        row: int,
        col: int
    ):
        """Add uncertainty analysis subplot."""
        uncertainty = predictions["uncertainty"]
        ci = uncertainty["confidence_interval"]
        
        if self.config.show_confidence:
            # Add confidence interval
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([t, t[::-1]]),
                    y=np.concatenate([ci["upper"], ci["lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(0,100,80,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Confidence Interval",
                    showlegend=True
                ),
                row=row,
                col=col
            )
        
        # Add standard deviation
        fig.add_trace(
            go.Scatter(
                x=t,
                y=uncertainty["std"],
                mode="lines",
                name="Uncertainty",
                line=dict(color="red")
            ),
            row=row,
            col=col
        )
        
        fig.update_xaxes(title_text="Time", row=row, col=col)
        fig.update_yaxes(title_text="Uncertainty", row=row, col=col)
    
    def _add_trend_decomposition(
        self,
        fig: go.Figure,
        t: np.ndarray,
        predictions: Dict[str, Any],
        row: int,
        col: int
    ):
        """Add trend decomposition subplot."""
        trends = predictions["trends"]
        decomposition = trends["decomposition"]
        
        # Add trend component
        fig.add_trace(
            go.Scatter(
                x=t,
                y=decomposition["trend"],
                mode="lines",
                name="Trend",
                line=dict(color="blue")
            ),
            row=row,
            col=col
        )
        
        # Add seasonal component if available
        if decomposition["seasonal"]:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=decomposition["seasonal"],
                    mode="lines",
                    name="Seasonal",
                    line=dict(color="green")
                ),
                row=row,
                col=col
            )
        
        fig.update_xaxes(title_text="Time", row=row, col=col)
        fig.update_yaxes(title_text="Component", row=row, col=col)
    
    def _add_anomaly_detection(
        self,
        fig: go.Figure,
        t: np.ndarray,
        predictions: Dict[str, Any],
        row: int,
        col: int
    ):
        """Add anomaly detection subplot."""
        anomalies = predictions["anomalies"]
        
        # Add anomaly scores
        fig.add_trace(
            go.Scatter(
                x=t[anomalies["indices"]],
                y=anomalies["scores"],
                mode="markers",
                name="Anomalies",
                marker=dict(
                    size=10,
                    color=anomalies["severity"],
                    colorscale="Viridis",
                    showscale=True
                )
            ),
            row=row,
            col=col
        )
        
        fig.update_xaxes(title_text="Time", row=row, col=col)
        fig.update_yaxes(title_text="Anomaly Score", row=row, col=col)
    
    def save_visualizations(
        self,
        name: str,
        t: np.ndarray
    ):
        """Save all visualizations."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save predictions visualization
            pred_fig = self.visualize_predictions(name, t)
            pred_fig.write_html(str(output_path / f"{name}_predictions.html"))
            
            # Save forecasts visualization
            forecast_fig = self.visualize_forecasts(name)
            forecast_fig.write_html(str(output_path / f"{name}_forecasts.html"))
            
            # Save variations visualization
            var_fig = self.visualize_variations(name)
            var_fig.write_html(str(output_path / f"{name}_variations.html"))
            
            # Save dashboard
            dashboard_fig = self.create_prediction_dashboard(name, t)
            dashboard_fig.write_html(str(output_path / f"{name}_dashboard.html"))
            
            logger.info(f"Saved visualizations to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save visualizations: {e}")

def create_prediction_visualizer(
    predictor: EasingPredictor,
    output_path: Optional[Path] = None
) -> PredictionVisualizer:
    """Create prediction visualizer."""
    config = VisualizationConfig(output_path=output_path)
    return PredictionVisualizer(predictor, config)

if __name__ == "__main__":
    # Example usage
    from .easing_prediction import create_easing_predictor
    from .easing_statistics import create_easing_statistics
    from .easing_metrics import create_easing_metrics
    from .easing_functions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    metrics = create_easing_metrics(easing)
    stats = create_easing_statistics(metrics)
    predictor = create_easing_predictor(stats)
    visualizer = create_prediction_visualizer(
        predictor,
        output_path=Path("prediction_viz")
    )
    
    # Generate visualizations
    t = np.linspace(0, 1, 100)
    visualizer.save_visualizations("ease-in-out-cubic", t)
