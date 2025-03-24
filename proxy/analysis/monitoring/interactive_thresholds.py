"""Interactive controls for threshold visualization."""

import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import logging
import threading
import queue
import time

from .threshold_visualization import ThresholdVisualizer
from .adaptive_thresholds import AdaptiveThresholds, AdaptiveConfig

logger = logging.getLogger(__name__)

class InteractiveThresholds:
    """Interactive dashboard for threshold analysis."""
    
    def __init__(
        self,
        thresholds: AdaptiveThresholds,
        port: int = 8052,
        update_interval: float = 1.0
    ):
        self.thresholds = thresholds
        self.visualizer = ThresholdVisualizer(thresholds)
        self.port = port
        self.update_interval = update_interval
        
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
        self.update_queue: queue.Queue = queue.Queue()
        self._running = False
        self._updater_thread: Optional[threading.Thread] = None
    
    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Interactive Threshold Analysis"),
            
            # Controls
            html.Div([
                html.H3("Controls"),
                
                # Metric selector
                html.Div([
                    html.Label("Metric:"),
                    dcc.Dropdown(
                        id="metric-selector",
                        placeholder="Select metric..."
                    )
                ], style={"margin": "10px 0"}),
                
                # Time range selector
                html.Div([
                    html.Label("Time Range:"),
                    dcc.RangeSlider(
                        id="time-range",
                        min=0,
                        max=168,  # 1 week in hours
                        step=1,
                        value=[24, 168],
                        marks={
                            0: "Now",
                            24: "1d",
                            72: "3d",
                            168: "1w"
                        }
                    )
                ], style={"margin": "20px 0"}),
                
                # Threshold controls
                html.Div([
                    html.Label("Threshold Settings:"),
                    
                    html.Div([
                        html.Label("Sensitivity:"),
                        dcc.Slider(
                            id="sensitivity",
                            min=0.1,
                            max=5.0,
                            step=0.1,
                            value=1.0,
                            marks={
                                0.1: "Low",
                                1.0: "Normal",
                                5.0: "High"
                            }
                        )
                    ]),
                    
                    html.Div([
                        html.Label("Learning Rate:"),
                        dcc.Slider(
                            id="learning-rate",
                            min=0.01,
                            max=1.0,
                            step=0.01,
                            value=0.1,
                            marks={
                                0.01: "Slow",
                                0.1: "Normal",
                                1.0: "Fast"
                            }
                        )
                    ]),
                    
                    html.Div([
                        dcc.Checklist(
                            id="adjustments",
                            options=[
                                {"label": "Seasonal", "value": "seasonal"},
                                {"label": "Trend", "value": "trend"}
                            ],
                            value=["seasonal", "trend"]
                        )
                    ])
                ], style={"margin": "20px 0", "padding": "10px", "border": "1px solid #ddd"}),
                
                # Update controls
                html.Div([
                    html.Button(
                        "Apply Changes",
                        id="apply-button",
                        style={"margin-right": "10px"}
                    ),
                    html.Button(
                        "Reset",
                        id="reset-button"
                    )
                ], style={"margin": "10px 0"})
            ], style={"padding": "20px", "background": "#f8f9fa", "borderRadius": "5px"}),
            
            # Status
            html.Div([
                html.H3("Status"),
                html.Div(id="status-message")
            ], style={"margin": "20px 0"}),
            
            # Visualizations
            html.Div([
                # Evolution plot
                html.Div([
                    html.H3("Threshold Evolution"),
                    dcc.Graph(id="evolution-plot")
                ], style={"margin": "20px 0"}),
                
                # Distribution plot
                html.Div([
                    html.H3("Value Distribution"),
                    dcc.Graph(id="distribution-plot")
                ], style={"margin": "20px 0"}),
                
                # Seasonal pattern
                html.Div([
                    html.H3("Seasonal Pattern"),
                    dcc.Graph(id="seasonal-plot")
                ], style={"margin": "20px 0"})
            ]),
            
            # Hidden data store
            dcc.Store(id="threshold-data"),
            
            # Update interval
            dcc.Interval(
                id="update-interval",
                interval=int(self.update_interval * 1000),
                n_intervals=0
            )
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            [
                Output("metric-selector", "options"),
                Output("metric-selector", "value")
            ],
            [Input("update-interval", "n_intervals")]
        )
        def update_metrics(_):
            """Update available metrics."""
            metrics = []
            for key in self.thresholds.history.keys():
                for metric_type in self.thresholds.history[key].keys():
                    metrics.append({
                        "label": f"{key} - {metric_type}",
                        "value": f"{key}:{metric_type}"
                    })
            
            value = metrics[0]["value"] if metrics else None
            return metrics, value
        
        @self.app.callback(
            [
                Output("evolution-plot", "figure"),
                Output("distribution-plot", "figure"),
                Output("seasonal-plot", "figure"),
                Output("status-message", "children")
            ],
            [
                Input("metric-selector", "value"),
                Input("time-range", "value"),
                Input("update-interval", "n_intervals")
            ]
        )
        def update_plots(metric_value, time_range, _):
            """Update visualization plots."""
            if not metric_value:
                return (
                    self.visualizer._empty_figure("No metric selected"),
                    self.visualizer._empty_figure("No metric selected"),
                    self.visualizer._empty_figure("No metric selected"),
                    "No metric selected"
                )
            
            try:
                metric_key, metric_type = metric_value.split(":")
                time_window = timedelta(hours=time_range[1] - time_range[0])
                
                evolution = self.visualizer.plot_threshold_evolution(
                    metric_key,
                    metric_type,
                    time_window
                )
                
                distribution = self.visualizer.plot_threshold_distribution(
                    metric_key,
                    metric_type
                )
                
                seasonal = self.visualizer.plot_seasonal_heatmap(
                    metric_key,
                    metric_type
                )
                
                stats = self._get_metric_stats(metric_key, metric_type)
                
                return (
                    evolution,
                    distribution,
                    seasonal,
                    stats
                )
                
            except Exception as e:
                logger.error(f"Plot update failed: {e}")
                return (
                    self.visualizer._empty_figure("Update failed"),
                    self.visualizer._empty_figure("Update failed"),
                    self.visualizer._empty_figure("Update failed"),
                    f"Error: {str(e)}"
                )
        
        @self.app.callback(
            Output("threshold-data", "data"),
            [
                Input("apply-button", "n_clicks"),
                Input("reset-button", "n_clicks")
            ],
            [
                State("sensitivity", "value"),
                State("learning-rate", "value"),
                State("adjustments", "value")
            ]
        )
        def update_settings(apply_clicks, reset_clicks, sensitivity, learning_rate, adjustments):
            """Update threshold settings."""
            ctx = dash.callback_context
            if not ctx.triggered:
                return {}
            
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if trigger_id == "reset-button":
                # Reset to defaults
                config = AdaptiveConfig()
            else:
                # Apply new settings
                config = AdaptiveConfig(
                    sensitivity_factor=sensitivity,
                    learning_rate=learning_rate,
                    seasonal_adjust="seasonal" in adjustments,
                    trend_adjust="trend" in adjustments
                )
            
            self.thresholds.config = config
            return {"updated": datetime.now().isoformat()}
    
    def _get_metric_stats(self, metric_key: str, metric_type: str) -> str:
        """Get metric statistics summary."""
        values = self.thresholds.history[metric_key][metric_type]
        if not values:
            return "No data available"
        
        stats = [
            f"Total samples: {len(values)}",
            f"Current value: {values[-1]:.2f}",
            f"Mean: {np.mean(values):.2f}",
            f"Std: {np.std(values):.2f}",
            f"Min: {min(values):.2f}",
            f"Max: {max(values):.2f}"
        ]
        
        if metric_key in self.thresholds.thresholds:
            threshold = self.thresholds.thresholds[metric_key].get(metric_type)
            if threshold is not None:
                stats.append(f"Current threshold: {threshold:.2f}")
        
        return " | ".join(stats)
    
    def run(self):
        """Run interactive dashboard."""
        self.app.run_server(
            port=self.port,
            debug=False,
            use_reloader=False
        )

def run_interactive_thresholds(
    thresholds: AdaptiveThresholds,
    port: int = 8052
) -> None:
    """Run interactive threshold dashboard."""
    dashboard = InteractiveThresholds(thresholds, port)
    dashboard.run()

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
    
    # Run dashboard
    run_interactive_thresholds(thresholds)
