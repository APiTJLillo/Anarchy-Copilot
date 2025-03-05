"""Interactive sensitivity analysis interface."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import logging
from pathlib import Path
import json

from .sensitivity_analysis import SensitivityAnalyzer, SensitivityConfig
from .power_analysis import PowerAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class InteractiveConfig:
    """Configuration for interactive analysis."""
    port: int = 8050
    host: str = "localhost"
    debug: bool = False
    auto_open: bool = True
    theme: str = "plotly"
    update_interval: float = 1.0
    max_points: int = 1000
    cache_timeout: int = 3600
    output_path: Optional[Path] = None

class InteractiveSensitivity:
    """Interactive interface for sensitivity analysis."""
    
    def __init__(
        self,
        analyzer: SensitivityAnalyzer,
        config: InteractiveConfig
    ):
        self.analyzer = analyzer
        self.config = config
        self.app = dash.Dash(__name__, external_stylesheets=[config.theme])
        self.cache: Dict[str, Any] = {}
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Sensitivity Analysis Dashboard"),
            
            # Parameter controls
            html.Div([
                html.H3("Analysis Parameters"),
                html.Div([
                    html.Label("Test Type"),
                    dcc.Dropdown(
                        id="test-type",
                        options=[
                            {"label": "T-Test", "value": "t_test"},
                            {"label": "F-Test", "value": "f_test"},
                            {"label": "Chi-Square", "value": "chi_square"}
                        ],
                        value="t_test"
                    )
                ], style={"width": "30%", "display": "inline-block"}),
                
                html.Div([
                    html.Label("Parameter Ranges"),
                    dcc.RangeSlider(
                        id="effect-size-range",
                        min=0.1,
                        max=1.0,
                        step=0.1,
                        marks={i/10: str(i/10) for i in range(1, 11)},
                        value=[0.2, 0.8],
                        tooltip={"placement": "bottom"}
                    )
                ], style={"width": "60%", "display": "inline-block"})
            ]),
            
            # Results display
            html.Div([
                html.H3("Analysis Results"),
                dcc.Loading(
                    id="loading-results",
                    children=[
                        dcc.Graph(id="importance-plot"),
                        dcc.Graph(id="sensitivity-plot")
                    ]
                )
            ]),
            
            # Interactive exploration
            html.Div([
                html.H3("Interactive Exploration"),
                html.Div([
                    html.Label("Parameter"),
                    dcc.Dropdown(
                        id="parameter-select",
                        options=[
                            {"label": "Effect Size", "value": "effect_size"},
                            {"label": "Sample Size", "value": "sample_size"},
                            {"label": "Alpha", "value": "alpha"},
                            {"label": "Variance", "value": "variance"}
                        ],
                        value="effect_size"
                    )
                ], style={"width": "30%", "display": "inline-block"}),
                
                dcc.Graph(id="parameter-impact-plot")
            ]),
            
            # Recommendations
            html.Div([
                html.H3("Recommendations"),
                html.Div(id="recommendations-output")
            ]),
            
            # Export controls
            html.Div([
                html.Button("Export Results", id="export-button"),
                dcc.Download(id="download-results")
            ]),
            
            # Real-time updates
            dcc.Interval(
                id="interval-component",
                interval=int(self.config.update_interval * 1000),
                n_intervals=0
            )
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            [
                Output("importance-plot", "figure"),
                Output("sensitivity-plot", "figure")
            ],
            [
                Input("test-type", "value"),
                Input("effect-size-range", "value"),
                Input("interval-component", "n_intervals")
            ]
        )
        def update_plots(test_type: str, effect_range: List[float], _) -> Tuple[go.Figure, go.Figure]:
            """Update main analysis plots."""
            cache_key = f"{test_type}_{effect_range[0]}_{effect_range[1]}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Update analyzer config
            self.analyzer.config.parameter_ranges["effect_size"] = np.linspace(
                effect_range[0],
                effect_range[1],
                5
            ).tolist()
            
            # Run analysis
            results = self.analyzer.analyze_sensitivity(test_type)
            plots = self.analyzer.plot_sensitivity()
            
            # Cache results
            self.cache[cache_key] = (
                plots["importance"],
                plots["sensitivity"]
            )
            
            return plots["importance"], plots["sensitivity"]
        
        @self.app.callback(
            Output("parameter-impact-plot", "figure"),
            [
                Input("parameter-select", "value"),
                Input("test-type", "value")
            ]
        )
        def update_parameter_impact(parameter: str, test_type: str) -> go.Figure:
            """Update parameter impact visualization."""
            # Get parameter range
            param_range = self.analyzer.config.parameter_ranges[parameter]
            
            # Calculate impact across range
            impacts = []
            for value in param_range:
                params = {
                    "effect_size": 0.5,
                    "sample_size": 100,
                    "alpha": 0.05,
                    "variance": 1.0
                }
                params[parameter] = value
                
                result = self.analyzer._analyze_single_combination(
                    params,
                    test_type
                )
                if result:
                    impacts.append(result["power"])
            
            # Create plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=param_range,
                y=impacts,
                mode="lines+markers",
                name="Power"
            ))
            
            fig.update_layout(
                title=f"Impact of {parameter} on Power",
                xaxis_title=parameter,
                yaxis_title="Power",
                showlegend=True
            )
            
            return fig
        
        @self.app.callback(
            Output("recommendations-output", "children"),
            [Input("test-type", "value")]
        )
        def update_recommendations(test_type: str) -> html.Div:
            """Update recommendations display."""
            results = self.analyzer.analyze_sensitivity(test_type)
            recommendations = results["recommendations"]
            
            return html.Div([
                html.Div([
                    html.H4(rec["message"]),
                    html.P(f"Priority: {rec['priority']}")
                ])
                for rec in recommendations
            ])
        
        @self.app.callback(
            Output("download-results", "data"),
            Input("export-button", "n_clicks"),
            prevent_initial_call=True
        )
        def export_results(_) -> Dict[str, Any]:
            """Export analysis results."""
            if not self.config.output_path:
                return {}
            
            try:
                results = {}
                for test_type in ["t_test", "f_test", "chi_square"]:
                    results[test_type] = self.analyzer.analyze_sensitivity(test_type)
                
                return dict(
                    content=json.dumps(results, indent=2),
                    filename="sensitivity_analysis_results.json"
                )
            except Exception as e:
                logger.error(f"Failed to export results: {e}")
                return {}
    
    def run(self):
        """Run interactive dashboard."""
        self.app.run_server(
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug
        )

def create_interactive_analyzer(
    power_analyzer: PowerAnalyzer,
    port: Optional[int] = None,
    output_path: Optional[Path] = None
) -> InteractiveSensitivity:
    """Create interactive sensitivity analyzer."""
    sensitivity_analyzer = SensitivityAnalyzer(
        power_analyzer,
        SensitivityConfig()
    )
    
    config = InteractiveConfig(
        port=port or 8050,
        output_path=output_path
    )
    
    return InteractiveSensitivity(sensitivity_analyzer, config)

if __name__ == "__main__":
    # Example usage
    from .power_analysis import create_analyzer
    
    power_analyzer = create_analyzer()
    interactive = create_interactive_analyzer(
        power_analyzer,
        output_path=Path("sensitivity_results")
    )
    
    # Run dashboard
    interactive.run()
