"""Interactive features for profile visualization."""

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
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from .profile_visualization import ProfileVisualizer, VisualizationConfig
from .prediction_profiling import PredictionProfiler

logger = logging.getLogger(__name__)

@dataclass
class InteractiveConfig:
    """Configuration for interactive features."""
    port: int = 8050
    debug: bool = False
    theme: str = "darkly"
    auto_refresh: bool = True
    refresh_interval: float = 5.0
    max_history: int = 1000
    enable_callbacks: bool = True
    output_path: Optional[Path] = None

class InteractiveProfiler:
    """Interactive profiling interface."""
    
    def __init__(
        self,
        visualizer: ProfileVisualizer,
        config: InteractiveConfig
    ):
        self.visualizer = visualizer
        self.config = config
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes[config.theme.upper()]]
        )
        self.current_view = "dashboard"
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup Dash application layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Profile Analysis Dashboard"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button(
                            "Dashboard",
                            id="btn-dashboard",
                            color="primary",
                            className="me-1"
                        ),
                        dbc.Button(
                            "Call Graph",
                            id="btn-call-graph",
                            color="secondary",
                            className="me-1"
                        ),
                        dbc.Button(
                            "Memory Analysis",
                            id="btn-memory",
                            color="info",
                            className="me-1"
                        ),
                        dbc.Button(
                            "Performance",
                            id="btn-performance",
                            color="success",
                            className="me-1"
                        )
                    ])
                ])
            ]),
            
            html.Br(),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="main-graph")
                ])
            ]),
            
            html.Br(),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Time Range"),
                                    dcc.RangeSlider(
                                        id="time-slider",
                                        min=0,
                                        max=100,
                                        value=[0, 100],
                                        marks={
                                            0: "Start",
                                            100: "End"
                                        }
                                    )
                                ])
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Metrics"),
                                    dcc.Dropdown(
                                        id="metric-selector",
                                        options=[
                                            {"label": "Execution Time", "value": "time"},
                                            {"label": "Memory Usage", "value": "memory"},
                                            {"label": "Call Count", "value": "calls"}
                                        ],
                                        value="time",
                                        multi=True
                                    )
                                ])
                            ])
                        ])
                    ])
                ]),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Analysis"),
                        dbc.CardBody([
                            html.Div(id="analysis-output")
                        ])
                    ])
                ])
            ]),
            
            dcc.Interval(
                id="refresh-interval",
                interval=self.config.refresh_interval * 1000,
                disabled=not self.config.auto_refresh
            )
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup Dash callbacks."""
        if not self.config.enable_callbacks:
            return
        
        @self.app.callback(
            Output("main-graph", "figure"),
            [
                Input("btn-dashboard", "n_clicks"),
                Input("btn-call-graph", "n_clicks"),
                Input("btn-memory", "n_clicks"),
                Input("btn-performance", "n_clicks"),
                Input("time-slider", "value"),
                Input("metric-selector", "value"),
                Input("refresh-interval", "n_intervals")
            ]
        )
        def update_graph(
            n1: Optional[int],
            n2: Optional[int],
            n3: Optional[int],
            n4: Optional[int],
            time_range: List[float],
            metrics: List[str],
            _: int
        ) -> go.Figure:
            """Update main graph."""
            ctx = dash.callback_context
            if not ctx.triggered:
                button_id = "btn-dashboard"
            else:
                button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if button_id == "btn-dashboard":
                return self.visualizer.create_dashboard()
            elif button_id == "btn-call-graph":
                return self.visualizer.create_call_graph_visualization()
            elif button_id == "btn-memory":
                return self.visualizer.create_memory_timeline()
            elif button_id == "btn-performance":
                return self.visualizer.create_performance_sunburst()
            else:
                raise PreventUpdate
        
        @self.app.callback(
            Output("analysis-output", "children"),
            [
                Input("main-graph", "selectedData"),
                Input("metric-selector", "value")
            ]
        )
        def update_analysis(
            selected_data: Optional[Dict[str, Any]],
            metrics: List[str]
        ) -> html.Div:
            """Update analysis output."""
            if not selected_data:
                return html.P("Select data points for analysis")
            
            analysis = self._analyze_selection(selected_data, metrics)
            
            return html.Div([
                html.H5("Selection Analysis"),
                html.Br(),
                dbc.Table.from_dataframe(
                    pd.DataFrame(analysis),
                    striped=True,
                    bordered=True,
                    hover=True
                )
            ])
    
    def _analyze_selection(
        self,
        selected_data: Dict[str, Any],
        metrics: List[str]
    ) -> pd.DataFrame:
        """Analyze selected data points."""
        points = selected_data.get("points", [])
        if not points:
            return pd.DataFrame()
        
        analysis = []
        for point in points:
            point_data = {
                "x": point.get("x"),
                "y": point.get("y"),
                "type": point.get("curveNumber", 0)
            }
            
            # Add metric-specific analysis
            if "time" in metrics:
                point_data["execution_time"] = point.get("y", 0)
            if "memory" in metrics:
                point_data["memory_mb"] = point.get("y", 0)
            if "calls" in metrics:
                point_data["call_count"] = point.get("z", 0)
            
            analysis.append(point_data)
        
        return pd.DataFrame(analysis)
    
    def run_server(self, **kwargs):
        """Run Dash server."""
        self.app.run_server(
            port=self.config.port,
            debug=self.config.debug,
            **kwargs
        )
    
    def save_state(self):
        """Save current state."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            state = {
                "current_view": self.current_view,
                "timestamp": datetime.now().isoformat()
            }
            
            state_file = output_path / "interactive_state.json"
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved state to {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load_state(self):
        """Load saved state."""
        if not self.config.output_path:
            return
        
        try:
            state_file = self.config.output_path / "interactive_state.json"
            if not state_file.exists():
                return
            
            with open(state_file) as f:
                state = json.load(f)
            
            self.current_view = state.get("current_view", "dashboard")
            
            logger.info(f"Loaded state from {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

def create_interactive_profiler(
    visualizer: ProfileVisualizer,
    output_path: Optional[Path] = None
) -> InteractiveProfiler:
    """Create interactive profiler."""
    config = InteractiveConfig(output_path=output_path)
    return InteractiveProfiler(visualizer, config)

if __name__ == "__main__":
    # Example usage
    from .profile_visualization import create_profile_visualizer
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
    profile_viz = create_profile_visualizer(profiler)
    interactive = create_interactive_profiler(
        profile_viz,
        output_path=Path("interactive_profile")
    )
    
    # Run interactive dashboard
    interactive.run_server()
