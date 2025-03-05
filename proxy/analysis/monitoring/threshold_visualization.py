"""Visualization tools for adaptive thresholds."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import logging
from pathlib import Path

from .adaptive_thresholds import AdaptiveThresholds, AdaptiveConfig

logger = logging.getLogger(__name__)

class ThresholdVisualizer:
    """Visualize adaptive thresholds and their behavior."""
    
    def __init__(
        self,
        thresholds: AdaptiveThresholds,
        output_dir: Optional[Path] = None
    ):
        self.thresholds = thresholds
        self.output_dir = output_dir or Path("threshold_visualizations")
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_threshold_evolution(
        self,
        metric_key: str,
        metric_type: str,
        time_range: timedelta = timedelta(days=7)
    ) -> go.Figure:
        """Plot threshold evolution over time."""
        # Get historical values
        values = self.thresholds.history[metric_key][metric_type]
        if not values:
            return self._empty_figure("No data available")
        
        # Create time points
        end_time = datetime.now()
        start_time = end_time - time_range
        timestamps = [
            start_time + timedelta(
                seconds=i * time_range.total_seconds() / len(values)
            )
            for i in range(len(values))
        ]
        
        # Calculate thresholds at each point
        thresholds = []
        for ts, val in zip(timestamps, values):
            threshold = self.thresholds.get_threshold(
                metric_key,
                metric_type,
                val,
                ts
            )
            thresholds.append(threshold)
        
        # Create figure
        fig = go.Figure()
        
        # Add value line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            name="Values",
            mode="lines",
            line=dict(color="blue", width=1)
        ))
        
        # Add threshold line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=thresholds,
            name="Threshold",
            mode="lines",
            line=dict(color="red", width=2, dash="dash")
        ))
        
        # Add seasonal pattern if enabled
        if self.thresholds.config.seasonal_adjust:
            seasonal = self._get_seasonal_pattern(metric_key, metric_type)
            if seasonal is not None:
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=seasonal,
                    name="Seasonal Pattern",
                    mode="lines",
                    line=dict(color="green", width=1, dash="dot")
                ))
        
        fig.update_layout(
            title=f"Threshold Evolution: {metric_key} - {metric_type}",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode="x unified"
        )
        
        return fig
    
    def plot_threshold_distribution(
        self,
        metric_key: str,
        metric_type: str
    ) -> go.Figure:
        """Plot distribution of values and thresholds."""
        values = self.thresholds.history[metric_key][metric_type]
        if not values:
            return self._empty_figure("No data available")
        
        # Calculate current threshold components
        base_threshold = self.thresholds.thresholds.get(
            metric_key, {}
        ).get(metric_type)
        
        if base_threshold is None:
            return self._empty_figure("No threshold calculated")
        
        # Create figure
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Value Distribution",
                "Threshold Components"
            )
        )
        
        # Add value histogram
        fig.add_trace(
            go.Histogram(
                x=values,
                name="Values",
                nbinsx=50
            ),
            row=1,
            col=1
        )
        
        # Add threshold line
        fig.add_vline(
            x=base_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Base Threshold",
            row=1,
            col=1
        )
        
        # Add threshold components
        components = self._get_threshold_components(
            metric_key,
            metric_type
        )
        
        fig.add_trace(
            go.Bar(
                x=list(components.keys()),
                y=list(components.values()),
                name="Components"
            ),
            row=2,
            col=1
        )
        
        fig.update_layout(
            title=f"Threshold Analysis: {metric_key} - {metric_type}",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def plot_seasonal_heatmap(
        self,
        metric_key: str,
        metric_type: str
    ) -> go.Figure:
        """Plot seasonal pattern heatmap."""
        if not self.thresholds.config.seasonal_adjust:
            return self._empty_figure("Seasonal adjustment disabled")
        
        seasonal_factors = self.thresholds.seasonal_factors.get(
            metric_key, {}
        ).get(metric_type)
        
        if seasonal_factors is None:
            return self._empty_figure("No seasonal pattern available")
        
        # Reshape into weekly pattern
        weekly = np.tile(seasonal_factors, (7, 1))
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=weekly,
            x=[f"{h:02d}:00" for h in range(24)],
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale="RdYlBu",
            zmid=1.0
        ))
        
        fig.update_layout(
            title=f"Seasonal Pattern: {metric_key} - {metric_type}",
            xaxis_title="Hour",
            yaxis_title="Day",
            height=400
        )
        
        return fig
    
    def create_dashboard(
        self,
        metric_key: str,
        metric_type: str
    ) -> None:
        """Create comprehensive threshold dashboard."""
        # Create visualizations
        evolution = self.plot_threshold_evolution(metric_key, metric_type)
        distribution = self.plot_threshold_distribution(metric_key, metric_type)
        seasonal = self.plot_seasonal_heatmap(metric_key, metric_type)
        
        # Save individual plots
        evolution.write_html(
            str(self.output_dir / f"{metric_key}_{metric_type}_evolution.html")
        )
        distribution.write_html(
            str(self.output_dir / f"{metric_key}_{metric_type}_distribution.html")
        )
        seasonal.write_html(
            str(self.output_dir / f"{metric_key}_{metric_type}_seasonal.html")
        )
        
        # Create index page
        self._create_index_page(metric_key, metric_type)
    
    def _get_seasonal_pattern(
        self,
        metric_key: str,
        metric_type: str
    ) -> Optional[np.ndarray]:
        """Get expanded seasonal pattern."""
        seasonal_factors = self.thresholds.seasonal_factors.get(
            metric_key, {}
        ).get(metric_type)
        
        if seasonal_factors is None:
            return None
        
        # Repeat pattern for entire time range
        values = self.thresholds.history[metric_key][metric_type]
        base = np.median(values)
        
        return np.tile(
            seasonal_factors * base,
            len(values) // 24 + 1
        )[:len(values)]
    
    def _get_threshold_components(
        self,
        metric_key: str,
        metric_type: str
    ) -> Dict[str, float]:
        """Get threshold component breakdown."""
        components = {}
        
        # Base threshold
        base = self.thresholds.thresholds.get(
            metric_key, {}
        ).get(metric_type, 0)
        components["Base"] = base
        
        # Seasonal adjustment
        if self.thresholds.config.seasonal_adjust:
            seasonal = self.thresholds.seasonal_factors.get(
                metric_key, {}
            ).get(metric_type)
            if seasonal is not None:
                components["Seasonal"] = float(np.std(seasonal))
        
        # Trend adjustment
        if self.thresholds.config.trend_adjust:
            trend = self.thresholds.trend_factors.get(
                metric_key, {}
            ).get(metric_type, 0)
            components["Trend"] = abs(trend)
        
        # History-based decay
        history_size = len(
            self.thresholds.history[metric_key].get(metric_type, [])
        )
        if history_size > 0:
            decay = self.thresholds.config.decay_factor ** (
                history_size / self.thresholds.config.min_history
            )
            components["Decay"] = 1 - decay
        
        return components
    
    def _empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        return fig
    
    def _create_index_page(self, metric_key: str, metric_type: str):
        """Create HTML index page."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Threshold Analysis: {metric_key} - {metric_type}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2em; }}
                iframe {{
                    width: 100%;
                    height: 600px;
                    border: none;
                    margin: 1em 0;
                }}
            </style>
        </head>
        <body>
            <h1>Threshold Analysis: {metric_key} - {metric_type}</h1>
            
            <h2>Threshold Evolution</h2>
            <iframe src="{metric_key}_{metric_type}_evolution.html"></iframe>
            
            <h2>Distribution Analysis</h2>
            <iframe src="{metric_key}_{metric_type}_distribution.html"></iframe>
            
            <h2>Seasonal Pattern</h2>
            <iframe src="{metric_key}_{metric_type}_seasonal.html"></iframe>
        </body>
        </html>
        """
        
        with open(self.output_dir / f"{metric_key}_{metric_type}_dashboard.html", "w") as f:
            f.write(html)

def visualize_thresholds(
    thresholds: AdaptiveThresholds,
    metric_key: str,
    metric_type: str,
    output_dir: Optional[Path] = None
) -> None:
    """Create threshold visualizations."""
    visualizer = ThresholdVisualizer(thresholds, output_dir)
    visualizer.create_dashboard(metric_key, metric_type)

if __name__ == "__main__":
    # Example usage
    from .adaptive_thresholds import create_adaptive_thresholds
    
    thresholds = create_adaptive_thresholds()
    
    # Add sample data
    for hour in range(24 * 7):
        value = 50 + 10 * np.sin(hour * np.pi / 12)  # Daily pattern
        value += 5 * np.sin(hour * np.pi / (24 * 7))  # Weekly pattern
        value += np.random.normal(0, 2)  # Random noise
        
        thresholds.add_value(
            "cpu",
            "percent",
            value,
            datetime.now() - timedelta(hours=24*7-hour)
        )
    
    # Create visualizations
    visualize_thresholds(thresholds, "cpu", "percent")
