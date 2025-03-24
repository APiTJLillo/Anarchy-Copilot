"""Interactive controls for validation visualization."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import numpy as np

from .validation_visualization import ValidationVisualizer
from .cluster_validation import ClusterValidator

class InteractiveValidationControls:
    """Interactive controls for validation visualization."""
    
    def __init__(
        self,
        visualizer: ValidationVisualizer
    ):
        self.visualizer = visualizer
        self.validator = visualizer.validator
        self.filters = {
            "time_window": None,
            "min_cluster_size": None,
            "metric_thresholds": {},
            "severity_levels": set()
        }
    
    def create_interactive_dashboard(self) -> go.Figure:
        """Create interactive validation dashboard."""
        fig = self.visualizer.create_validation_dashboard()
        
        # Add range slider for time window
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        # Add metric threshold controls
        self._add_threshold_controls(fig)
        
        # Add cluster size filter
        self._add_size_filter(fig)
        
        # Add severity filter
        self._add_severity_filter(fig)
        
        # Add filter reset button
        self._add_reset_button(fig)
        
        return fig
    
    def _add_threshold_controls(
        self,
        fig: go.Figure
    ):
        """Add metric threshold controls."""
        metrics = ["silhouette", "calinski_harabasz", "davies_bouldin"]
        
        steps = []
        for metric in metrics:
            values = self._get_metric_range(metric)
            if values:
                min_val, max_val = values
                step = (max_val - min_val) / 100
                
                steps.append({
                    "method": "update",
                    "args": [
                        {"visible": [True]},
                        {
                            "shapes": [{
                                "type": "line",
                                "x0": min_val,
                                "x1": max_val,
                                "y0": 0,
                                "y1": 0,
                                "line": {
                                    "color": "red",
                                    "width": 2,
                                    "dash": "dash"
                                }
                            }]
                        }
                    ],
                    "label": f"{metric} threshold"
                })
        
        if steps:
            fig.update_layout(
                updatemenus=[{
                    "buttons": steps,
                    "direction": "down",
                    "showactive": True,
                    "x": 0.1,
                    "y": 1.1
                }]
            )
    
    def _add_size_filter(
        self,
        fig: go.Figure
    ):
        """Add cluster size filter."""
        sizes = self._get_cluster_sizes()
        if sizes:
            min_size, max_size = min(sizes), max(sizes)
            
            fig.update_layout(
                sliders=[{
                    "active": 0,
                    "currentvalue": {"prefix": "Min Cluster Size: "},
                    "pad": {"t": 50},
                    "steps": [
                        {
                            "method": "update",
                            "args": [{"visible": [size >= min_size for size in sizes]}],
                            "label": str(min_size)
                        }
                        for min_size in range(int(min_size), int(max_size) + 1)
                    ]
                }]
            )
    
    def _add_severity_filter(
        self,
        fig: go.Figure
    ):
        """Add severity level filter."""
        severities = self._get_severity_levels()
        
        buttons = []
        for severity in severities:
            visible = [True] * len(fig.data)
            buttons.append({
                "method": "update",
                "args": [{"visible": visible}],
                "label": severity.name
            })
        
        if buttons:
            fig.update_layout(
                updatemenus=[{
                    "buttons": buttons,
                    "direction": "down",
                    "showactive": True,
                    "x": 0.9,
                    "y": 1.1
                }]
            )
    
    def _add_reset_button(
        self,
        fig: go.Figure
    ):
        """Add filter reset button."""
        fig.update_layout(
            updatemenus=[{
                "buttons": [{
                    "method": "update",
                    "args": [
                        {"visible": [True] * len(fig.data)},
                        {
                            "shapes": [],
                            "sliders": [{"active": 0}],
                            "updatemenus": [{"active": 0}]
                        }
                    ],
                    "label": "Reset Filters"
                }],
                "type": "buttons",
                "x": 0.5,
                "y": 1.1
            }]
        )
    
    def _get_metric_range(
        self,
        metric: str
    ) -> Optional[tuple[float, float]]:
        """Get range of values for metric."""
        summary = self.validator.get_validation_summary()
        if not summary or not summary["trends"].get(metric):
            return None
        
        values = summary["trends"][metric]
        return min(values), max(values)
    
    def _get_cluster_sizes(self) -> List[int]:
        """Get list of cluster sizes."""
        summary = self.validator.get_validation_summary()
        if not summary:
            return []
        
        return [
            metrics["size"]
            for metrics in summary["current"]["cluster_metrics"].values()
        ]
    
    def _get_severity_levels(self) -> Set[Any]:
        """Get unique severity levels."""
        summary = self.validator.get_validation_summary()
        if not summary:
            return set()
        
        return set(summary["severity_distribution"].keys())
    
    def create_interactive_comparison(self) -> go.Figure:
        """Create interactive metric comparison."""
        fig = self.visualizer.create_metric_comparison()
        
        # Add time window selection
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        # Add correlation analysis controls
        self._add_correlation_controls(fig)
        
        # Add trend analysis controls
        self._add_trend_controls(fig)
        
        return fig
    
    def _add_correlation_controls(
        self,
        fig: go.Figure
    ):
        """Add correlation analysis controls."""
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "method": "update",
                        "args": [
                            {
                                "type": "scatter",
                                "mode": "markers",
                                "marker": {
                                    "size": 8,
                                    "color": "blue"
                                }
                            }
                        ],
                        "label": "Raw"
                    },
                    {
                        "method": "update",
                        "args": [
                            {
                                "type": "scatter",
                                "mode": "markers",
                                "marker": {
                                    "size": 8,
                                    "color": "blue"
                                },
                                "line": {
                                    "fit": "lowess"
                                }
                            }
                        ],
                        "label": "Smoothed"
                    }
                ],
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "y": 1.1
            }]
        )
    
    def _add_trend_controls(
        self,
        fig: go.Figure
    ):
        """Add trend analysis controls."""
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "method": "update",
                        "args": [
                            {
                                "type": "scatter",
                                "mode": "lines+markers"
                            }
                        ],
                        "label": "Line"
                    },
                    {
                        "method": "update",
                        "args": [
                            {
                                "type": "scatter",
                                "mode": "markers",
                                "line": {
                                    "shape": "spline"
                                }
                            }
                        ],
                        "label": "Spline"
                    }
                ],
                "direction": "down",
                "showactive": True,
                "x": 0.9,
                "y": 1.1
            }]
        )

def create_interactive_controls(
    visualizer: ValidationVisualizer
) -> InteractiveValidationControls:
    """Create interactive validation controls."""
    return InteractiveValidationControls(visualizer)

if __name__ == "__main__":
    # Example usage
    from .validation_visualization import create_validation_visualizer
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
    controls = create_interactive_controls(visualizer)
    
    async def main():
        # Start clustering
        await clusterer.start_clustering()
        
        # Create interactive visualizations
        dashboard = controls.create_interactive_dashboard()
        dashboard.show()
        
        comparison = controls.create_interactive_comparison()
        comparison.show()
    
    # Run example
    asyncio.run(main())
