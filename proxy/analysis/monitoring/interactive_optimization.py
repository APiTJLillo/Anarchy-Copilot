"""Interactive exploration of multi-objective optimization results."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from .multi_objective_optimization import MultiObjectiveOptimizer, MultiObjectiveConfig
from .simulation_optimization import SimulationOptimizer

logger = logging.getLogger(__name__)

@dataclass
class InteractionConfig:
    """Configuration for interactive exploration."""
    update_interval: int = 1000
    max_history: int = 100
    highlight_threshold: float = 0.1
    preference_learning_rate: float = 0.1
    output_path: Optional[Path] = None

class InteractiveExplorer:
    """Interactive exploration of optimization results."""
    
    def __init__(
        self,
        optimizer: MultiObjectiveOptimizer,
        config: InteractionConfig
    ):
        self.optimizer = optimizer
        self.config = config
        
        # State management
        self.current_view = None
        self.selection_history = []
        self.preference_weights = self.optimizer.config.objective_weights.copy()
        self.highlighted_solutions = set()
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup Dash app layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Multi-objective Optimization Explorer"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                # Visualization panel
                dbc.Col([
                    dcc.Graph(
                        id="pareto-plot",
                        style={"height": "600px"}
                    ),
                    dcc.Graph(
                        id="trade-off-plot",
                        style={"height": "400px"}
                    )
                ], width=8),
                
                # Control panel
                dbc.Col([
                    html.H4("Controls"),
                    html.Hr(),
                    
                    # Objective weights
                    html.H5("Objective Weights"),
                    *[
                        dbc.Row([
                            dbc.Col([
                                html.Label(obj),
                                dcc.Slider(
                                    id=f"weight-{obj}",
                                    min=0,
                                    max=1,
                                    step=0.1,
                                    value=weight,
                                    marks={i/10: str(i/10) for i in range(11)}
                                )
                            ])
                        ])
                        for obj, weight in self.preference_weights.items()
                    ],
                    
                    html.Hr(),
                    
                    # View controls
                    html.H5("View Options"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id="view-mode",
                                options=[
                                    {"label": "3D Pareto", "value": "3d"},
                                    {"label": "2D Projection", "value": "2d"},
                                    {"label": "Parallel Coordinates", "value": "parallel"}
                                ],
                                value="3d"
                            )
                        ])
                    ]),
                    
                    html.Hr(),
                    
                    # Selection info
                    html.H5("Selected Solution"),
                    html.Div(id="selection-info"),
                    
                    html.Hr(),
                    
                    # Action buttons
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Update Preferences",
                                id="update-button",
                                color="primary",
                                className="mr-2"
                            ),
                            dbc.Button(
                                "Reset View",
                                id="reset-button",
                                color="secondary"
                            )
                        ])
                    ])
                ], width=4)
            ]),
            
            dbc.Row([
                # History panel
                dbc.Col([
                    html.H4("Exploration History"),
                    html.Hr(),
                    dcc.Graph(
                        id="history-plot",
                        style={"height": "300px"}
                    )
                ])
            ])
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Setup Dash app callbacks."""
        @self.app.callback(
            [
                Output("pareto-plot", "figure"),
                Output("trade-off-plot", "figure"),
                Output("selection-info", "children"),
                Output("history-plot", "figure")
            ],
            [
                Input("view-mode", "value"),
                Input("update-button", "n_clicks"),
                Input("reset-button", "n_clicks")
            ] + [
                Input(f"weight-{obj}", "value")
                for obj in self.preference_weights.keys()
            ],
            [
                State("pareto-plot", "clickData")
            ]
        )
        def update_visualization(
            view_mode,
            update_clicks,
            reset_clicks,
            *args
        ):
            ctx = dash.callback_context
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            click_data = args[-1]
            weights = args[:-1]
            
            # Update preferences if requested
            if trigger == "update-button":
                self._update_preferences(
                    {
                        obj: weight
                        for obj, weight in zip(self.preference_weights.keys(), weights)
                    }
                )
            
            # Reset if requested
            elif trigger == "reset-button":
                self._reset_view()
            
            # Update current view
            self.current_view = view_mode
            
            # Create visualizations
            pareto_fig = self._create_pareto_plot(view_mode)
            trade_off_fig = self._create_trade_off_plot()
            history_fig = self._create_history_plot()
            
            # Update selection info
            selection_info = self._get_selection_info(click_data)
            
            return pareto_fig, trade_off_fig, selection_info, history_fig
    
    def _create_pareto_plot(
        self,
        view_mode: str
    ) -> go.Figure:
        """Create Pareto front visualization."""
        if view_mode == "3d":
            return self._create_3d_pareto()
        elif view_mode == "2d":
            return self._create_2d_pareto()
        else:
            return self._create_parallel_pareto()
    
    def _create_3d_pareto(self) -> go.Figure:
        """Create 3D Pareto front plot."""
        fig = go.Figure()
        
        # Get Pareto solutions
        solutions = self.optimizer.optimization_results["pareto_front"]["solutions"]
        
        # Plot points
        obj_names = list(self.preference_weights.keys())
        x = [s["objectives"][obj_names[0]] for s in solutions]
        y = [s["objectives"][obj_names[1]] for s in solutions]
        z = [s["objectives"][obj_names[2]] for s in solutions]
        
        colors = [
            "red" if i in self.highlighted_solutions else "blue"
            for i in range(len(solutions))
        ]
        
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=5,
                    color=colors,
                    opacity=0.8
                ),
                text=[f"Solution {s['id']}" for s in solutions],
                hoverinfo="text"
            )
        )
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title=obj_names[0],
                yaxis_title=obj_names[1],
                zaxis_title=obj_names[2]
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        
        return fig
    
    def _create_2d_pareto(self) -> go.Figure:
        """Create 2D Pareto front projections."""
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Obj1 vs Obj2",
                "Obj2 vs Obj3",
                "Obj1 vs Obj3"
            ]
        )
        
        # Get Pareto solutions
        solutions = self.optimizer.optimization_results["pareto_front"]["solutions"]
        obj_names = list(self.preference_weights.keys())
        
        # Create projections
        for i, (obj1, obj2) in enumerate([
            (0, 1),
            (1, 2),
            (0, 2)
        ]):
            x = [s["objectives"][obj_names[obj1]] for s in solutions]
            y = [s["objectives"][obj_names[obj2]] for s in solutions]
            
            colors = [
                "red" if i in self.highlighted_solutions else "blue"
                for i in range(len(solutions))
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(
                        color=colors,
                        size=5
                    ),
                    text=[f"Solution {s['id']}" for s in solutions],
                    hoverinfo="text",
                    showlegend=False
                ),
                row=1,
                col=i+1
            )
        
        return fig
    
    def _create_parallel_pareto(self) -> go.Figure:
        """Create parallel coordinates plot."""
        fig = go.Figure()
        
        # Get Pareto solutions
        solutions = self.optimizer.optimization_results["pareto_front"]["solutions"]
        obj_names = list(self.preference_weights.keys())
        
        # Create parallel coordinates
        for i, sol in enumerate(solutions):
            y = [sol["objectives"][obj] for obj in obj_names]
            
            fig.add_trace(
                go.Scatter(
                    x=obj_names,
                    y=y,
                    mode="lines",
                    line=dict(
                        color="red" if i in self.highlighted_solutions else "blue",
                        width=2
                    ),
                    name=f"Solution {sol['id']}"
                )
            )
        
        return fig
    
    def _create_trade_off_plot(self) -> go.Figure:
        """Create trade-off analysis plot."""
        fig = go.Figure()
        
        # Get trade-off data
        trade_offs = self.optimizer.optimization_results["trade_offs"]
        
        # Create heatmap
        fig.add_trace(
            go.Heatmap(
                z=trade_offs["correlation_matrix"],
                x=trade_offs["objectives"],
                y=trade_offs["objectives"],
                colorscale="RdBu"
            )
        )
        
        return fig
    
    def _create_history_plot(self) -> go.Figure:
        """Create exploration history plot."""
        fig = go.Figure()
        
        if self.selection_history:
            # Plot preference evolution
            for obj in self.preference_weights:
                weights = [h[obj] for h in self.selection_history]
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(weights))),
                        y=weights,
                        mode="lines+markers",
                        name=f"{obj} Weight"
                    )
                )
        
        fig.update_layout(
            xaxis_title="Step",
            yaxis_title="Weight",
            showlegend=True
        )
        
        return fig
    
    def _get_selection_info(
        self,
        click_data: Optional[Dict[str, Any]]
    ) -> List[html.Div]:
        """Get information about selected solution."""
        if not click_data:
            return [html.Div("No solution selected")]
        
        # Get selected solution
        point_index = click_data["points"][0]["pointIndex"]
        solution = self.optimizer.optimization_results["pareto_front"]["solutions"][point_index]
        
        return [
            html.Div([
                html.H6(f"Solution {solution['id']}"),
                html.P("Objectives:"),
                html.Ul([
                    html.Li(f"{obj}: {val:.4f}")
                    for obj, val in solution["objectives"].items()
                ]),
                html.P("Parameters:"),
                html.Ul([
                    html.Li(f"{param}: {val:.4f}")
                    for param, val in solution["parameters"].items()
                ])
            ])
        ]
    
    def _update_preferences(
        self,
        new_weights: Dict[str, float]
    ):
        """Update preference weights."""
        self.preference_weights = new_weights
        
        # Normalize weights
        total = sum(new_weights.values())
        if total > 0:
            self.preference_weights = {
                k: v/total for k, v in new_weights.items()
            }
        
        # Update history
        self.selection_history.append(self.preference_weights.copy())
        if len(self.selection_history) > self.config.max_history:
            self.selection_history.pop(0)
        
        # Update highlighted solutions
        self._update_highlights()
    
    def _update_highlights(self):
        """Update highlighted solutions based on preferences."""
        self.highlighted_solutions.clear()
        
        solutions = self.optimizer.optimization_results["pareto_front"]["solutions"]
        
        # Calculate weighted scores
        scores = []
        for sol in solutions:
            score = sum(
                self.preference_weights[obj] * val
                for obj, val in sol["objectives"].items()
            )
            scores.append(score)
        
        # Highlight best solutions
        threshold = np.percentile(scores, (1 - self.config.highlight_threshold) * 100)
        self.highlighted_solutions.update(
            i for i, score in enumerate(scores)
            if score >= threshold
        )
    
    def _reset_view(self):
        """Reset view to initial state."""
        self.current_view = "3d"
        self.selection_history.clear()
        self.preference_weights = self.optimizer.config.objective_weights.copy()
        self.highlighted_solutions.clear()
    
    def run_server(
        self,
        host: str = "localhost",
        port: int = 8050,
        debug: bool = False
    ):
        """Run the interactive explorer server."""
        self.app.run_server(
            host=host,
            port=port,
            debug=debug
        )
    
    def save_session(
        self,
        output_path: Optional[Path] = None
    ):
        """Save exploration session."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            session_data = {
                "preference_history": self.selection_history,
                "final_preferences": self.preference_weights,
                "highlighted_solutions": list(self.highlighted_solutions)
            }
            
            with open(path / "exploration_session.json", "w") as f:
                json.dump(session_data, f, indent=2)
            
            logger.info(f"Saved exploration session to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

def create_interactive_explorer(
    optimizer: MultiObjectiveOptimizer,
    output_path: Optional[Path] = None
) -> InteractiveExplorer:
    """Create interactive explorer."""
    config = InteractionConfig(output_path=output_path)
    return InteractiveExplorer(optimizer, config)

if __name__ == "__main__":
    # Example usage
    from .multi_objective_optimization import create_multi_objective_optimizer
    from .simulation_optimization import create_simulation_optimizer
    from .monte_carlo_power import create_monte_carlo_analyzer
    from .power_analysis import create_chain_power_analyzer
    from .statistical_comparison import create_chain_statistician
    from .comparison_animation import create_chain_comparator
    from .chain_animation import create_chain_animator
    from .chain_visualization import create_chain_visualizer
    from .filter_chaining import create_filter_chain
    from .learning_filters import create_learning_filter
    from .interactive_learning import create_interactive_learning
    
    # Create components
    filters = create_learning_filter()
    chain = create_filter_chain(filters)
    chain_viz = create_chain_visualizer(chain)
    animator = create_chain_animator(chain_viz)
    comparator = create_chain_comparator(animator)
    statistician = create_chain_statistician(comparator)
    power_analyzer = create_chain_power_analyzer(statistician)
    mc_analyzer = create_monte_carlo_analyzer(power_analyzer)
    sim_optimizer = create_simulation_optimizer(mc_analyzer)
    mo_optimizer = create_multi_objective_optimizer(sim_optimizer)
    explorer = create_interactive_explorer(
        mo_optimizer,
        output_path=Path("interactive_exploration")
    )
    
    # Run interactive explorer
    explorer.run_server(debug=True)
