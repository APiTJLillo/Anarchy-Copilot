"""Interactive coverage visualization with filtering capabilities."""

import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta

from .coverage_visualization import CoverageVisualizer

logger = logging.getLogger(__name__)

class InteractiveCoverageDashboard:
    """Interactive dashboard for coverage analysis."""
    
    def __init__(
        self,
        port: int = 8051,
        history_file: Optional[Path] = None
    ):
        self.app = dash.Dash(__name__)
        self.port = port
        self.history_file = history_file
        self.visualizer = CoverageVisualizer()
        
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Interactive Coverage Analysis"),
            
            # Filters section
            html.Div([
                html.H3("Filters"),
                
                # Coverage threshold filter
                html.Div([
                    html.Label("Coverage Threshold:"),
                    dcc.Slider(
                        id="coverage-threshold",
                        min=0,
                        max=1,
                        step=0.1,
                        value=0.7,
                        marks={i/10: f"{i*10}%" for i in range(11)}
                    )
                ], style={"margin": "10px 0"}),
                
                # Method filter
                html.Div([
                    html.Label("Filter Methods:"),
                    dcc.Dropdown(
                        id="method-filter",
                        multi=True,
                        placeholder="Select methods..."
                    )
                ], style={"margin": "10px 0"}),
                
                # Mutation type filter
                html.Div([
                    html.Label("Mutation Types:"),
                    dcc.Dropdown(
                        id="mutation-filter",
                        multi=True,
                        placeholder="Select mutation types..."
                    )
                ], style={"margin": "10px 0"}),
                
                # Date range filter
                html.Div([
                    html.Label("Date Range:"),
                    dcc.DateRangePickerSingle(
                        id="date-filter",
                        min_date_allowed=datetime.now() - timedelta(days=90),
                        max_date_allowed=datetime.now(),
                        initial_visible_month=datetime.now()
                    )
                ], style={"margin": "10px 0"})
            ], style={"padding": "20px", "background": "#f8f9fa", "borderRadius": "5px"}),
            
            # Stats summary
            html.Div([
                html.H3("Coverage Summary"),
                html.Div(id="coverage-stats")
            ], style={"margin": "20px 0"}),
            
            # Main visualizations
            html.Div([
                # Coverage heatmap
                html.Div([
                    html.H3("Method Coverage"),
                    dcc.Graph(id="coverage-heatmap")
                ], style={"margin": "20px 0"}),
                
                # Mutation effectiveness
                html.Div([
                    html.H3("Mutation Effectiveness"),
                    dcc.Graph(id="mutation-effectiveness")
                ], style={"margin": "20px 0"}),
                
                # Field coverage
                html.Div([
                    html.H3("Field Coverage"),
                    dcc.Graph(id="field-coverage")
                ], style={"margin": "20px 0"}),
                
                # Error distribution
                html.Div([
                    html.H3("Error Distribution"),
                    dcc.Graph(id="error-distribution")
                ], style={"margin": "20px 0"})
            ]),
            
            # Timeline (if history available)
            html.Div([
                html.H3("Coverage Timeline"),
                dcc.Graph(id="coverage-timeline")
            ], style={"margin": "20px 0"}),
            
            # Hidden data store
            dcc.Store(id="coverage-data"),
            
            # Update interval
            dcc.Interval(
                id="update-interval",
                interval=30000,  # 30 seconds
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            [
                Output("coverage-heatmap", "figure"),
                Output("mutation-effectiveness", "figure"),
                Output("field-coverage", "figure"),
                Output("error-distribution", "figure"),
                Output("coverage-timeline", "figure"),
                Output("coverage-stats", "children"),
                Output("method-filter", "options"),
                Output("mutation-filter", "options")
            ],
            [
                Input("coverage-threshold", "value"),
                Input("method-filter", "value"),
                Input("mutation-filter", "value"),
                Input("date-filter", "start_date"),
                Input("date-filter", "end_date"),
                Input("update-interval", "n_intervals")
            ],
            [State("coverage-data", "data")]
        )
        def update_visualizations(
            threshold: float,
            methods: List[str],
            mutations: List[str],
            start_date: str,
            end_date: str,
            n_intervals: int,
            data: Dict[str, Any]
        ) -> Tuple[go.Figure, ...]:
            """Update visualizations based on filters."""
            if not data:
                return self._empty_figures()
            
            # Filter data
            filtered_data = self._filter_data(
                data,
                threshold,
                methods,
                mutations,
                start_date,
                end_date
            )
            
            # Create figures
            heatmap = self.visualizer.create_coverage_heatmap(filtered_data)
            effectiveness = self.visualizer.create_mutation_effectiveness(filtered_data)
            field_cov = self.visualizer.create_field_coverage_sunburst(filtered_data)
            error_dist = self.visualizer.create_error_distribution(filtered_data)
            
            # Create timeline if history available
            timeline = None
            if self.history_file:
                timeline = self.visualizer.create_coverage_timeline(self.history_file)
            
            # Create stats summary
            stats = self._create_stats_summary(filtered_data)
            
            # Update filter options
            method_options = self._get_method_options(data)
            mutation_options = self._get_mutation_options(data)
            
            return (
                heatmap,
                effectiveness,
                field_cov,
                error_dist,
                timeline or go.Figure(),
                stats,
                method_options,
                mutation_options
            )
    
    def _filter_data(
        self,
        data: Dict[str, Any],
        threshold: float,
        methods: List[str],
        mutations: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Filter analysis data based on criteria."""
        filtered = data.copy()
        
        # Filter by coverage threshold
        filtered["validation_coverage"]["methods"] = {
            m: v for m, v in data["validation_coverage"]["methods"].items()
            if v["coverage_percent"] >= threshold
        }
        
        # Filter by methods
        if methods:
            filtered["validation_coverage"]["methods"] = {
                m: v for m, v in filtered["validation_coverage"]["methods"].items()
                if m in methods
            }
        
        # Filter by mutation types
        if mutations:
            filtered["mutation_stats"] = {
                k: v for k, v in data["mutation_stats"].items()
                if any(m in k for m in mutations)
            }
        
        # Filter by date range
        if start_date and end_date:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            
            filtered["history"] = [
                entry for entry in data.get("history", [])
                if start <= datetime.fromisoformat(entry["timestamp"]) <= end
            ]
        
        return filtered
    
    def _create_stats_summary(self, data: Dict[str, Any]) -> html.Div:
        """Create coverage statistics summary."""
        coverage = data["validation_coverage"]
        
        return html.Div([
            html.Div([
                html.Strong("Overall Coverage: "),
                f"{coverage['overall_coverage']:.1%}"
            ]),
            html.Div([
                html.Strong("Methods Covered: "),
                f"{len(coverage['methods'])}"
            ]),
            html.Div([
                html.Strong("Uncovered Methods: "),
                f"{len(coverage['uncovered'])}"
            ]),
            html.Div([
                html.Strong("Total Mutations: "),
                f"{data['mutation_stats']['total_mutations']}"
            ])
        ])
    
    def _get_method_options(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get options for method filter."""
        methods = list(data["validation_coverage"]["methods"].keys())
        return [{"label": m, "value": m} for m in methods]
    
    def _get_mutation_options(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get options for mutation filter."""
        mutations = [
            k for k in data["mutation_stats"].keys()
            if isinstance(k, str)
        ]
        return [{"label": m, "value": m} for m in mutations]
    
    def _empty_figures(self) -> Tuple[go.Figure, ...]:
        """Create empty figures when no data available."""
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[{
                "text": "No data available",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return (empty_fig,) * 5
    
    def run(self):
        """Run the dashboard server."""
        self.app.run_server(
            port=self.port,
            debug=False,
            use_reloader=False
        )

def run_interactive_dashboard(
    analysis: Dict[str, Any],
    history_file: Optional[Path] = None,
    port: int = 8051
):
    """Run interactive coverage dashboard."""
    dashboard = InteractiveCoverageDashboard(port, history_file)
    dashboard.app.layout.children[-2].data = analysis  # Update data store
    dashboard.run()

if __name__ == "__main__":
    # Example usage
    with open("coverage_analysis.json") as f:
        analysis = json.load(f)
    run_interactive_dashboard(analysis)
