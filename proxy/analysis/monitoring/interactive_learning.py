"""Interactive interfaces for learning visualization."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from .learning_visualization import LearningVisualizer, VisualizationConfig
from .optimization_learning import OptimizationLearner

logger = logging.getLogger(__name__)

@dataclass
class InteractiveConfig:
    """Configuration for interactive learning visualization."""
    port: int = 8050
    debug: bool = False
    theme: str = "darkly"
    auto_refresh: bool = True
    refresh_interval: float = 1.0
    enable_callbacks: bool = True
    output_path: Optional[Path] = None

class InteractiveLearning:
    """Interactive interface for learning visualization."""
    
    def __init__(
        self,
        visualizer: LearningVisualizer,
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
                    html.H1("Learning Analysis Dashboard"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Analysis Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Time Range"),
                                    dcc.RangeSlider(
                                        id="time-range",
                                        min=0,
                                        max=100,
                                        step=1,
                                        value=[0, 100],
                                        marks={
                                            0: "Start",
                                            25: "25%",
                                            50: "50%",
                                            75: "75%",
                                            100: "End"
                                        }
                                    )
                                ]),
                                dbc.Col([
                                    html.Label("Feature Selection"),
                                    dcc.Dropdown(
                                        id="feature-selector",
                                        multi=True,
                                        placeholder="Select features..."
                                    )
                                ])
                            ]),
                            
                            html.Br(),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Analysis Type"),
                                    dcc.Dropdown(
                                        id="analysis-type",
                                        options=[
                                            {"label": "Impact Analysis", "value": "impact"},
                                            {"label": "Feature Analysis", "value": "feature"},
                                            {"label": "Success Analysis", "value": "success"},
                                            {"label": "Learning Progress", "value": "progress"}
                                        ],
                                        value="impact"
                                    )
                                ]),
                                dbc.Col([
                                    html.Label("View Options"),
                                    dbc.ButtonGroup([
                                        dbc.Button(
                                            "Refresh",
                                            id="refresh-button",
                                            color="primary",
                                            className="me-1"
                                        ),
                                        dbc.Button(
                                            "Export",
                                            id="export-button",
                                            color="secondary",
                                            className="me-1"
                                        ),
                                        dbc.Button(
                                            "Settings",
                                            id="settings-button",
                                            color="info"
                                        )
                                    ])
                                ])
                            ])
                        ])
                    ])
                ])
            ]),
            
            html.Br(),
            
            dbc.Row([
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(
                            dcc.Graph(id="primary-plot"),
                            label="Primary View"
                        ),
                        dbc.Tab(
                            dcc.Graph(id="supplementary-plot"),
                            label="Supplementary View"
                        ),
                        dbc.Tab(
                            dcc.Graph(id="diagnostic-plot"),
                            label="Diagnostics"
                        )
                    ])
                ])
            ]),
            
            html.Br(),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Analysis Details"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id="analysis-stats")
                                ]),
                                dbc.Col([
                                    html.Div(id="analysis-insights")
                                ])
                            ])
                        ])
                    ])
                ])
            ]),
            
            dbc.Modal([
                dbc.ModalHeader("Settings"),
                dbc.ModalBody([
                    dbc.Form([
                        dbc.FormGroup([
                            dbc.Label("Theme"),
                            dcc.Dropdown(
                                id="theme-selector",
                                options=[
                                    {"label": "Light", "value": "light"},
                                    {"label": "Dark", "value": "dark"}
                                ],
                                value=("dark" if self.config.theme == "darkly" else "light")
                            )
                        ]),
                        html.Br(),
                        dbc.FormGroup([
                            dbc.Label("Auto Refresh"),
                            dbc.Switch(
                                id="auto-refresh-toggle",
                                value=self.config.auto_refresh
                            )
                        ]),
                        html.Br(),
                        dbc.FormGroup([
                            dbc.Label("Refresh Interval (s)"),
                            dbc.Input(
                                id="refresh-interval",
                                type="number",
                                value=self.config.refresh_interval,
                                min=0.1,
                                max=60,
                                step=0.1
                            )
                        ])
                    ])
                ]),
                dbc.ModalFooter([
                    dbc.Button(
                        "Close",
                        id="close-settings",
                        className="ml-auto"
                    )
                ])
            ], id="settings-modal"),
            
            dcc.Store(id="analysis-state"),
            
            dcc.Interval(
                id="refresh-interval",
                interval=self.config.refresh_interval * 1000,
                disabled=not self.config.auto_refresh
            )
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup Dash callbacks."""
        @self.app.callback(
            [
                Output("primary-plot", "figure"),
                Output("supplementary-plot", "figure"),
                Output("diagnostic-plot", "figure")
            ],
            [
                Input("analysis-type", "value"),
                Input("time-range", "value"),
                Input("feature-selector", "value"),
                Input("refresh-interval", "n_intervals"),
                Input("refresh-button", "n_clicks")
            ]
        )
        def update_plots(
            analysis_type: str,
            time_range: List[float],
            selected_features: List[str],
            n_intervals: Optional[int],
            n_clicks: Optional[int]
        ) -> Tuple[go.Figure, go.Figure, go.Figure]:
            """Update plot displays."""
            if analysis_type == "impact":
                primary = self.visualizer.create_impact_analysis()
                supplementary = self.visualizer.create_feature_analysis()
                diagnostic = self.visualizer.create_success_analysis()
            elif analysis_type == "feature":
                primary = self.visualizer.create_feature_analysis()
                supplementary = self.visualizer.create_success_analysis()
                diagnostic = self.visualizer.create_impact_analysis()
            elif analysis_type == "success":
                primary = self.visualizer.create_success_analysis()
                supplementary = self.visualizer.create_impact_analysis()
                diagnostic = self.visualizer.create_feature_analysis()
            else:  # progress
                primary = self.visualizer.create_learning_dashboard()
                supplementary = self.visualizer.create_impact_analysis()
                diagnostic = self.visualizer.create_feature_analysis()
            
            return primary, supplementary, diagnostic
        
        @self.app.callback(
            [
                Output("analysis-stats", "children"),
                Output("analysis-insights", "children")
            ],
            [
                Input("analysis-type", "value"),
                Input("refresh-interval", "n_intervals")
            ]
        )
        def update_analysis(
            analysis_type: str,
            n_intervals: Optional[int]
        ) -> Tuple[html.Div, html.Div]:
            """Update analysis details."""
            # Calculate statistics
            stats = self._calculate_stats(analysis_type)
            stats_div = html.Div([
                html.H5("Statistics"),
                html.Br(),
                dbc.Table.from_dataframe(
                    pd.DataFrame(stats),
                    striped=True,
                    bordered=True,
                    hover=True
                )
            ])
            
            # Generate insights
            insights = self._generate_insights(analysis_type, stats)
            insights_div = html.Div([
                html.H5("Insights"),
                html.Br(),
                html.Ul([
                    html.Li(insight) for insight in insights
                ])
            ])
            
            return stats_div, insights_div
        
        @self.app.callback(
            Output("settings-modal", "is_open"),
            [
                Input("settings-button", "n_clicks"),
                Input("close-settings", "n_clicks")
            ],
            [State("settings-modal", "is_open")]
        )
        def toggle_settings(
            n1: Optional[int],
            n2: Optional[int],
            is_open: bool
        ) -> bool:
            """Toggle settings modal."""
            if n1 or n2:
                return not is_open
            return is_open
        
        @self.app.callback(
            [
                Output("refresh-interval", "interval"),
                Output("refresh-interval", "disabled")
            ],
            [
                Input("auto-refresh-toggle", "value"),
                Input("refresh-interval", "value")
            ]
        )
        def update_refresh(
            auto_refresh: bool,
            interval: float
        ) -> Tuple[float, bool]:
            """Update refresh settings."""
            return interval * 1000, not auto_refresh
    
    def _calculate_stats(
        self,
        analysis_type: str
    ) -> List[Dict[str, Any]]:
        """Calculate statistics for analysis type."""
        learner = self.visualizer.learner
        
        if analysis_type == "impact":
            return [
                {
                    "Metric": "Mean Impact",
                    "Value": f"{learner.impact_model.score(np.array([[0]]), np.array([0])):.3f}"
                },
                {
                    "Metric": "Impact Variance",
                    "Value": f"{np.var(learner.impact_model.feature_importances_):.3f}"
                }
            ]
        elif analysis_type == "feature":
            return [
                {
                    "Metric": "Feature Count",
                    "Value": str(len(learner.feature_importances))
                },
                {
                    "Metric": "Top Feature",
                    "Value": max(
                        learner.feature_importances.items(),
                        key=lambda x: x[1]
                    )[0]
                }
            ]
        elif analysis_type == "success":
            return [
                {
                    "Metric": "Success Rate",
                    "Value": f"{learner.success_model.score(np.array([[0]]), np.array([0])):.3f}"
                },
                {
                    "Metric": "Sample Count",
                    "Value": str(learner.samples)
                }
            ]
        else:  # progress
            return [
                {
                    "Metric": "Training Count",
                    "Value": str(learner.samples)
                },
                {
                    "Metric": "Last Training",
                    "Value": learner.last_training.isoformat() if learner.last_training else "Never"
                }
            ]
    
    def _generate_insights(
        self,
        analysis_type: str,
        stats: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate insights for analysis type."""
        learner = self.visualizer.learner
        
        insights = []
        if analysis_type == "impact":
            impact_score = float(stats[0]["Value"])
            if impact_score > 0.8:
                insights.append("Impact predictions are highly accurate")
            elif impact_score > 0.6:
                insights.append("Impact predictions show good accuracy")
            else:
                insights.append("Impact predictions need improvement")
        
        elif analysis_type == "feature":
            feature_count = int(stats[0]["Value"])
            if feature_count < 5:
                insights.append("Consider adding more features")
            elif feature_count > 20:
                insights.append("Consider feature selection to reduce complexity")
        
        elif analysis_type == "success":
            success_rate = float(stats[0]["Value"])
            if success_rate > 0.8:
                insights.append("Success predictions are reliable")
            elif success_rate > 0.6:
                insights.append("Success predictions are moderately reliable")
            else:
                insights.append("Success predictions need more training data")
        
        else:  # progress
            sample_count = int(stats[0]["Value"])
            if sample_count < learner.config.min_samples:
                insights.append(f"Need {learner.config.min_samples - sample_count} more samples")
            else:
                insights.append("Sufficient training data available")
        
        return insights
    
    def run_server(self, **kwargs):
        """Run Dash server."""
        self.app.run_server(
            port=self.config.port,
            debug=self.config.debug,
            **kwargs
        )

def create_interactive_learning(
    visualizer: LearningVisualizer,
    output_path: Optional[Path] = None
) -> InteractiveLearning:
    """Create interactive learning interface."""
    config = InteractiveConfig(output_path=output_path)
    return InteractiveLearning(visualizer, config)

if __name__ == "__main__":
    # Example usage
    from .learning_visualization import create_learning_visualizer
    from .optimization_learning import create_optimization_learner
    from .composition_optimization import create_composition_optimizer
    from .composition_analysis import create_composition_analysis
    from .pattern_composition import create_pattern_composer
    from .scheduling_patterns import create_scheduling_pattern
    from .event_scheduler import create_event_scheduler
    from .animation_events import create_event_manager
    from .animation_controls import create_animation_controls
    from .interactive_easing import create_interactive_easing
    from .easing_visualization import create_easing_visualizer
    from .easing_transitions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    visualizer = create_easing_visualizer(easing)
    interactive = create_interactive_easing(visualizer)
    controls = create_animation_controls(interactive)
    events = create_event_manager(controls)
    scheduler = create_event_scheduler(events)
    pattern = create_scheduling_pattern(scheduler)
    composer = create_pattern_composer(pattern)
    analyzer = create_composition_analysis(composer)
    optimizer = create_composition_optimizer(analyzer)
    learner = create_optimization_learner(optimizer)
    viz = create_learning_visualizer(learner)
    interactive_learning = create_interactive_learning(
        viz,
        output_path=Path("interactive_learning")
    )
    
    # Run interactive dashboard
    interactive_learning.run_server()
