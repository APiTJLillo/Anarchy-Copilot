"""Visualization tools for mutation coverage analysis."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CoverageVisualizer:
    """Create visualizations for mutation coverage analysis."""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None
    ):
        self.output_dir = output_dir or Path("coverage_visualizations")
        self.output_dir.mkdir(exist_ok=True)
    
    def create_coverage_heatmap(
        self,
        analysis: Dict[str, Any]
    ) -> go.Figure:
        """Create heatmap of validation coverage."""
        validation_coverage = analysis["validation_coverage"]["methods"]
        
        methods = list(validation_coverage.keys())
        coverage_values = [
            v["coverage_percent"]
            for v in validation_coverage.values()
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=[coverage_values],
            x=methods,
            y=["Coverage"],
            colorscale="RdYlGn",
            text=[[f"{v:.1%}" for v in coverage_values]],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Validation Method Coverage",
            xaxis_title="Method",
            yaxis_title="Coverage Level",
            height=200
        )
        
        return fig
    
    def create_mutation_effectiveness(
        self,
        analysis: Dict[str, Any]
    ) -> go.Figure:
        """Create visualization of mutation effectiveness."""
        stats = analysis["mutation_stats"]
        
        # Prepare data
        mutations = list(stats["success_rate"].keys())
        success_rates = list(stats["success_rate"].values())
        error_rates = list(stats["error_rate"].values())
        
        # Create figure
        fig = go.Figure()
        
        # Add success rate bars
        fig.add_trace(go.Bar(
            name="Success Rate",
            x=mutations,
            y=success_rates,
            marker_color="green",
            opacity=0.6
        ))
        
        # Add error rate bars
        fig.add_trace(go.Bar(
            name="Error Rate",
            x=mutations,
            y=error_rates,
            marker_color="red",
            opacity=0.6
        ))
        
        fig.update_layout(
            title="Mutation Effectiveness",
            barmode="stack",
            xaxis_title="Mutation Type",
            yaxis_title="Rate",
            height=400
        )
        
        return fig
    
    def create_field_coverage_sunburst(
        self,
        analysis: Dict[str, Any]
    ) -> go.Figure:
        """Create sunburst diagram of field coverage."""
        field_coverage = analysis["field_coverage"]["fields"]
        
        # Prepare data
        labels = []
        parents = []
        values = []
        colors = []
        
        # Add root
        labels.append("Fields")
        parents.append("")
        values.append(100)
        colors.append("lightgrey")
        
        # Add categories
        categories = ["AnimationStyle", "PlaybackConfig", "InteractionConfig"]
        for category in categories:
            labels.append(category)
            parents.append("Fields")
            values.append(70)
            colors.append("grey")
        
        # Add fields
        for field, stats in field_coverage.items():
            # Determine parent category
            if "color" in field or "font" in field or "size" in field:
                parent = "AnimationStyle"
            elif "duration" in field or "mode" in field:
                parent = "PlaybackConfig"
            else:
                parent = "InteractionConfig"
            
            labels.append(field)
            parents.append(parent)
            values.append(stats["mutations"])
            coverage = stats["validations"] / stats["mutations"] if stats["mutations"] > 0 else 0
            colors.append(f"rgba(0, 255, 0, {coverage})")
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            branchvalues="total"
        ))
        
        fig.update_layout(
            title="Field Coverage Distribution",
            width=800,
            height=800
        )
        
        return fig
    
    def create_error_distribution(
        self,
        analysis: Dict[str, Any]
    ) -> go.Figure:
        """Create error distribution visualization."""
        error_dist = analysis["error_distribution"]["error_types"]
        
        # Sort by frequency
        sorted_errors = sorted(
            error_dist.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        fig = go.Figure(data=go.Bar(
            x=[e[0] for e in sorted_errors],
            y=[e[1] for e in sorted_errors],
            marker_color="red",
            opacity=0.6
        ))
        
        fig.update_layout(
            title="Error Distribution",
            xaxis_title="Error Type",
            yaxis_title="Count",
            height=400
        )
        
        return fig
    
    def create_coverage_timeline(
        self,
        history_file: Path
    ) -> go.Figure:
        """Create timeline of coverage changes."""
        if not history_file.exists():
            return None
        
        with open(history_file) as f:
            history = json.load(f)
        
        # Prepare data
        dates = []
        overall_coverage = []
        method_coverage = defaultdict(list)
        
        for entry in history:
            dates.append(entry["timestamp"])
            overall_coverage.append(
                entry["validation_coverage"]["overall_coverage"]
            )
            
            for method, stats in entry["validation_coverage"]["methods"].items():
                method_coverage[method].append(stats["coverage_percent"])
        
        # Create figure
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Overall Coverage Trend",
                "Method Coverage Trends"
            )
        )
        
        # Add overall coverage line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=overall_coverage,
                mode="lines+markers",
                name="Overall Coverage"
            ),
            row=1,
            col=1
        )
        
        # Add method coverage lines
        for method, coverage in method_coverage.items():
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=coverage,
                    mode="lines",
                    name=method,
                    opacity=0.6
                ),
                row=2,
                col=1
            )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title="Coverage Evolution"
        )
        
        return fig
    
    def create_dashboard(
        self,
        analysis: Dict[str, Any],
        history_file: Optional[Path] = None
    ):
        """Create comprehensive coverage dashboard."""
        # Create visualizations
        figures = {
            "coverage_heatmap": self.create_coverage_heatmap(analysis),
            "mutation_effectiveness": self.create_mutation_effectiveness(analysis),
            "field_coverage": self.create_field_coverage_sunburst(analysis),
            "error_distribution": self.create_error_distribution(analysis)
        }
        
        if history_file:
            figures["coverage_timeline"] = self.create_coverage_timeline(history_file)
        
        # Save individual visualizations
        for name, fig in figures.items():
            if fig is not None:
                fig.write_html(
                    str(self.output_dir / f"{name}.html")
                )
        
        # Create index page
        self._create_index_page(
            [name for name, fig in figures.items() if fig is not None]
        )
    
    def _create_index_page(self, visualizations: List[str]):
        """Create HTML index page."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Coverage Analysis Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 2em; }
                .viz-link {
                    display: block;
                    margin: 1em 0;
                    padding: 1em;
                    background: #f0f0f0;
                    border-radius: 4px;
                    text-decoration: none;
                    color: #333;
                }
                .viz-link:hover {
                    background: #e0e0e0;
                }
            </style>
        </head>
        <body>
            <h1>Coverage Analysis Dashboard</h1>
            <div id="visualizations">
        """
        
        for viz in visualizations:
            html += f"""
                <a class="viz-link" href="{viz}.html">
                    {viz.replace("_", " ").title()}
                </a>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / "index.html", "w") as f:
            f.write(html)

def visualize_coverage(
    analysis: Dict[str, Any],
    history_file: Optional[Path] = None,
    output_dir: Optional[Path] = None
):
    """Generate coverage visualizations."""
    visualizer = CoverageVisualizer(output_dir)
    visualizer.create_dashboard(analysis, history_file)

if __name__ == "__main__":
    # Example usage
    with open("coverage_analysis.json") as f:
        analysis = json.load(f)
    visualize_coverage(analysis)
