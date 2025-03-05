"""Interactive tools for easing visualizations."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from .easing_visualization import EasingVisualizer, VisualizationConfig
from .easing_transitions import EasingFunctions

logger = logging.getLogger(__name__)

@dataclass
class InteractiveConfig:
    """Configuration for interactive easing tools."""
    port: int = 8050
    debug: bool = False
    theme: str = "darkly"
    auto_refresh: bool = True
    refresh_interval: float = 1.0
    enable_callbacks: bool = True
    output_path: Optional[Path] = None

class InteractiveEasing:
    """Interactive interface for easing functions."""
    
    def __init__(
        self,
        visualizer: EasingVisualizer,
        config: InteractiveConfig
    ):
        self.visualizer = visualizer
        self.config = config
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes[config.theme.upper()]]
        )
        self.setup_layout()
        if config.enable_callbacks:
            self.setup_callbacks()
    
    def setup_layout(self):
        """Setup Dash application layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Interactive Easing Functions"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Easing Selection"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="easing-selector",
                                options=[
                                    {"label": name, "value": name}
                                    for name in self.visualizer.easing.get_easing_function("linear").__code__.co_names
                                    if name.startswith("ease")
                                ],
                                value="ease-in-out-cubic",
                                multi=True
                            )
                        ])
                    ])
                ])
            ]),
            
            html.Br(),
            
            dbc.Row([
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(
                            dcc.Graph(id="curve-comparison"),
                            label="Curve Comparison"
                        ),
                        dbc.Tab(
                            dcc.Graph(id="derivative-analysis"),
                            label="Derivative Analysis"
                        ),
                        dbc.Tab(
                            dcc.Graph(id="animation-preview"),
                            label="Animation Preview"
                        ),
                        dbc.Tab(
                            dcc.Graph(id="property-heatmap"),
                            label="Property Analysis"
                        )
                    ])
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
                                    dbc.Label("Property"),
                                    dcc.Dropdown(
                                        id="property-selector",
                                        options=[
                                            {"label": "Execution Time", "value": "execution_time"},
                                            {"label": "Smoothness", "value": "smoothness"},
                                            {"label": "Overshoots", "value": "overshoots"}
                                        ],
                                        value="execution_time"
                                    )
                                ]),
                                dbc.Col([
                                    dbc.Label("Animation Speed"),
                                    dcc.Slider(
                                        id="animation-speed",
                                        min=0.1,
                                        max=2.0,
                                        step=0.1,
                                        value=1.0,
                                        marks={
                                            0.1: "Slow",
                                            1.0: "Normal",
                                            2.0: "Fast"
                                        }
                                    )
                                ])
                            ])
                        ])
                    ])
                ])
            ]),
            
            html.Br(),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Animation Test"),
                        dbc.CardBody([
                            html.Div(
                                id="animation-box",
                                style={
                                    "width": "50px",
                                    "height": "50px",
                                    "backgroundColor": "blue",
                                    "position": "relative"
                                }
                            ),
                            html.Br(),
                            dbc.Button(
                                "Animate",
                                id="animate-button",
                                color="primary"
                            )
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
            
            dcc.Store(id="animation-state"),
            
            dcc.Interval(
                id="refresh-interval",
                interval=self.config.refresh_interval * 1000,
                disabled=not self.config.auto_refresh
            )
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup Dash callbacks."""
        @self.app.callback(
            Output("curve-comparison", "figure"),
            [Input("easing-selector", "value")]
        )
        def update_comparison(selected_easing: List[str]) -> go.Figure:
            """Update curve comparison plot."""
            return self.visualizer.create_curve_comparison(selected_easing)
        
        @self.app.callback(
            Output("derivative-analysis", "figure"),
            [Input("easing-selector", "value")]
        )
        def update_analysis(selected_easing: List[str]) -> go.Figure:
            """Update derivative analysis plot."""
            if not selected_easing:
                raise PreventUpdate
            return self.visualizer.create_derivative_analysis(selected_easing[0])
        
        @self.app.callback(
            Output("animation-preview", "figure"),
            [Input("easing-selector", "value")]
        )
        def update_preview(selected_easing: List[str]) -> go.Figure:
            """Update animation preview."""
            if not selected_easing:
                raise PreventUpdate
            return self.visualizer.create_animation_preview(selected_easing[0])
        
        @self.app.callback(
            Output("property-heatmap", "figure"),
            [Input("property-selector", "value")]
        )
        def update_heatmap(property_name: str) -> go.Figure:
            """Update property heatmap."""
            return self.visualizer.create_easing_heatmap(property_name)
        
        @self.app.callback(
            Output("animation-box", "style"),
            [
                Input("animate-button", "n_clicks"),
                Input("animation-speed", "value")
            ],
            [
                State("easing-selector", "value"),
                State("animation-box", "style")
            ]
        )
        def animate_box(
            n_clicks: Optional[int],
            speed: float,
            selected_easing: List[str],
            current_style: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Animate test box."""
            if not n_clicks or not selected_easing:
                raise PreventUpdate
            
            easing_func = self.visualizer.easing.get_easing_function(
                selected_easing[0]
            )
            
            progress = easing_func(min(1.0, (n_clicks % 2) * speed))
            left = f"{progress * 300}px"
            
            return {
                **current_style,
                "left": left,
                "transition": f"left {1/speed}s {selected_easing[0]}"
            }
        
        @self.app.callback(
            Output("analysis-output", "children"),
            [
                Input("easing-selector", "value"),
                Input("property-selector", "value")
            ]
        )
        def update_analysis(
            selected_easing: List[str],
            property_name: str
        ) -> html.Div:
            """Update analysis output."""
            if not selected_easing:
                return html.P("Select easing functions to analyze")
            
            analysis = []
            for name in selected_easing:
                t, y = self.visualizer.easing.generate_easing_curve(name)
                
                stats = {
                    "Name": name,
                    "Max Value": float(np.max(y)),
                    "Min Value": float(np.min(y)),
                    "Overshoots": int(np.sum((y < 0) | (y > 1))),
                    "Smoothness": float(-np.mean(np.abs(np.gradient(np.gradient(y, t), t))))
                }
                
                analysis.append(stats)
            
            return html.Div([
                html.H5("Easing Analysis"),
                html.Br(),
                dbc.Table.from_dataframe(
                    pd.DataFrame(analysis),
                    striped=True,
                    bordered=True,
                    hover=True
                )
            ])
    
    def run_server(self, **kwargs):
        """Run Dash server."""
        self.app.run_server(
            port=self.config.port,
            debug=self.config.debug,
            **kwargs
        )

def create_interactive_easing(
    visualizer: EasingVisualizer,
    output_path: Optional[Path] = None
) -> InteractiveEasing:
    """Create interactive easing interface."""
    config = InteractiveConfig(output_path=output_path)
    return InteractiveEasing(visualizer, config)

if __name__ == "__main__":
    # Example usage
    from .easing_visualization import create_easing_visualizer
    from .easing_transitions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    visualizer = create_easing_visualizer(easing)
    interactive = create_interactive_easing(
        visualizer,
        output_path=Path("interactive_easing")
    )
    
    # Run interactive dashboard
    interactive.run_server()
