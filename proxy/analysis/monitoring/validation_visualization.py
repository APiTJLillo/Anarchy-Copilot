"""Visualization tools for cluster validation metrics."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime, timedelta

from .cluster_validation import ClusterValidator

class ValidationVisualizer:
    """Visualize cluster validation metrics."""
    
    def __init__(
        self,
        validator: ClusterValidator
    ):
        self.validator = validator
    
    def create_validation_dashboard(self) -> go.Figure:
        """Create comprehensive validation dashboard."""
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "Metric Trends",
                "Cluster Evolution",
                "Stability Analysis",
                "Cohesion Distribution",
                "Temporal Patterns",
                "Severity Distribution"
            ]
        )
        
        # Get validation summary
        summary = self.validator.get_validation_summary()
        if not summary:
            return fig
        
        # Metric trends
        times = [i for i in range(len(summary["trends"]["silhouette"]))]
        
        for metric, values in summary["trends"].items():
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=values,
                    name=metric,
                    mode="lines+markers"
                ),
                row=1,
                col=1
            )
        
        # Cluster evolution
        fig.add_trace(
            go.Scatter(
                x=times,
                y=summary["cluster_evolution"]["sizes"],
                name="Number of Clusters",
                mode="lines+markers"
            ),
            row=1,
            col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=summary["cluster_evolution"]["alerts"],
                name="Number of Alerts",
                mode="lines+markers"
            ),
            row=1,
            col=2
        )
        
        # Stability analysis
        stability = summary["cluster_stability"]
        if stability:
            variances = list(stability["metric_variance"].values())
            metric_names = list(stability["metric_variance"].keys())
            
            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=variances,
                    name="Metric Variance"
                ),
                row=2,
                col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[0, len(metric_names)-1],
                    y=[stability["cluster_churn"], stability["cluster_churn"]],
                    name="Cluster Churn",
                    mode="lines",
                    line=dict(dash="dash")
                ),
                row=2,
                col=1
            )
        
        # Cohesion distribution
        current = summary["current"]
        cohesions = [
            metrics["cohesion"]
            for metrics in current["cluster_metrics"].values()
        ]
        
        fig.add_trace(
            go.Histogram(
                x=cohesions,
                name="Cohesion Distribution",
                nbinsx=20
            ),
            row=2,
            col=2
        )
        
        # Temporal patterns
        temporal_spreads = [
            metrics["temporal_spread"]
            for metrics in current["cluster_metrics"].values()
        ]
        cluster_sizes = [
            metrics["size"]
            for metrics in current["cluster_metrics"].values()
        ]
        
        fig.add_trace(
            go.Scatter(
                x=temporal_spreads,
                y=cluster_sizes,
                mode="markers",
                name="Temporal vs Size",
                marker=dict(
                    size=10,
                    color=[i for i in range(len(temporal_spreads))],
                    colorscale="Viridis",
                    showscale=True
                )
            ),
            row=3,
            col=1
        )
        
        # Severity distribution
        severity_entropies = [
            metrics["severity_entropy"]
            for metrics in current["cluster_metrics"].values()
        ]
        
        fig.add_trace(
            go.Box(
                y=severity_entropies,
                name="Severity Entropy"
            ),
            row=3,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            showlegend=True,
            title="Cluster Validation Dashboard"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Metric Value", row=1, col=1)
        
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        fig.update_xaxes(title_text="Metric", row=2, col=1)
        fig.update_yaxes(title_text="Variance", row=2, col=1)
        
        fig.update_xaxes(title_text="Cohesion", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        fig.update_xaxes(title_text="Temporal Spread", row=3, col=1)
        fig.update_yaxes(title_text="Cluster Size", row=3, col=1)
        
        fig.update_xaxes(title_text="Cluster", row=3, col=2)
        fig.update_yaxes(title_text="Severity Entropy", row=3, col=2)
        
        return fig
    
    def create_metric_comparison(self) -> go.Figure:
        """Create comparison of different validation metrics."""
        summary = self.validator.get_validation_summary()
        if not summary:
            return go.Figure()
        
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Metric Correlations",
                "Metric vs Cluster Size",
                "Metric Distributions",
                "Metric Stability"
            ]
        )
        
        metrics = summary["trends"]
        metric_names = list(metrics.keys())
        
        # Metric correlations
        for i, m1 in enumerate(metric_names):
            for j, m2 in enumerate(metric_names):
                if i < j:
                    fig.add_trace(
                        go.Scatter(
                            x=metrics[m1],
                            y=metrics[m2],
                            mode="markers",
                            name=f"{m1} vs {m2}",
                            marker=dict(size=8)
                        ),
                        row=1,
                        col=1
                    )
        
        # Metrics vs cluster size
        sizes = summary["cluster_evolution"]["sizes"]
        for metric in metric_names:
            fig.add_trace(
                go.Scatter(
                    x=sizes,
                    y=metrics[metric],
                    mode="markers",
                    name=f"{metric} vs Size",
                    marker=dict(size=8)
                ),
                row=1,
                col=2
            )
        
        # Metric distributions
        for metric in metric_names:
            fig.add_trace(
                go.Box(
                    y=metrics[metric],
                    name=metric
                ),
                row=2,
                col=1
            )
        
        # Metric stability
        if summary["cluster_stability"]:
            variances = list(summary["cluster_stability"]["metric_variance"].values())
            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=variances,
                    name="Stability"
                ),
                row=2,
                col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title="Validation Metric Comparison"
        )
        
        return fig
    
    def create_cluster_evolution_animation(self) -> go.Figure:
        """Create animated visualization of cluster evolution."""
        summary = self.validator.get_validation_summary(window=100)
        if not summary:
            return go.Figure()
        
        frames = []
        times = list(range(len(summary["trends"]["silhouette"])))
        
        for i in range(len(times)):
            frame_data = []
            
            # Metric trends
            for metric in summary["trends"]:
                frame_data.append(
                    go.Scatter(
                        x=times[:i+1],
                        y=summary["trends"][metric][:i+1],
                        name=metric,
                        mode="lines+markers"
                    )
                )
            
            # Cluster evolution
            frame_data.append(
                go.Scatter(
                    x=times[:i+1],
                    y=summary["cluster_evolution"]["sizes"][:i+1],
                    name="Clusters",
                    mode="lines+markers",
                    yaxis="y2"
                )
            )
            
            frames.append(go.Frame(data=frame_data, name=str(i)))
        
        # Create base figure
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=[times[0]],
                    y=[summary["trends"]["silhouette"][0]],
                    name="Initial",
                    mode="markers"
                )
            ],
            frames=frames
        )
        
        # Add slider and buttons
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 100, "redraw": True}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "type": "buttons"
            }],
            sliders=[{
                "currentvalue": {"prefix": "Time: "},
                "steps": [
                    {
                        "args": [[str(i)]],
                        "label": str(i),
                        "method": "animate"
                    }
                    for i in range(len(times))
                ]
            }]
        )
        
        fig.update_layout(
            title="Cluster Evolution Animation",
            xaxis_title="Time",
            yaxis_title="Metric Value",
            yaxis2=dict(
                title="Number of Clusters",
                overlaying="y",
                side="right"
            )
        )
        
        return fig

def create_validation_visualizer(
    validator: ClusterValidator
) -> ValidationVisualizer:
    """Create validation visualizer."""
    return ValidationVisualizer(validator)

if __name__ == "__main__":
    # Example usage
    from .cluster_validation import create_cluster_validator
    from .alert_clustering import create_alert_clusterer
    from .alert_management import create_alert_manager
    from .realtime_anomalies import create_realtime_detector
    from .anomaly_detection import create_anomaly_detector
    from .exploration_trends import create_trend_analyzer
    
    # Create components
    analyzer = create_trend_analyzer()
    detector = create_anomaly_detector(analyzer)
    realtime = create_realtime_detector(detector)
    manager = create_alert_manager(realtime)
    clusterer = create_alert_clusterer(manager)
    validator = create_cluster_validator(clusterer)
    visualizer = create_validation_visualizer(validator)
    
    async def main():
        # Start clustering
        await clusterer.start_clustering()
        
        # Create visualizations
        dashboard = visualizer.create_validation_dashboard()
        dashboard.show()
        
        comparison = visualizer.create_metric_comparison()
        comparison.show()
        
        animation = visualizer.create_cluster_evolution_animation()
        animation.show()
    
    # Run example
    asyncio.run(main())
