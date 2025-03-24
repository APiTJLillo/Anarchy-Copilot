"""Visualization tools for mutation testing coverage."""

import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json

from .state_migration import StateMigrator, MigrationMetadata
from tests.proxy.analysis.monitoring.test_mutation_coverage import (
    MutationTestResult,
    MigrationMutator
)

@dataclass
class VisualizationConfig:
    """Configuration for mutation visualization."""
    plot_height: int = 800
    plot_width: int = 1200
    export_path: Optional[Path] = None
    theme: str = "plotly"
    animation: bool = True
    interactive: bool = True

class MutationVisualizer:
    """Visualize mutation testing results."""
    
    def __init__(
        self,
        config: VisualizationConfig = None
    ):
        self.config = config or VisualizationConfig()
    
    def create_summary_plot(
        self,
        result: MutationTestResult
    ) -> go.Figure:
        """Create summary visualization."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Mutation Status",
                "Operator Performance",
                "Error Distribution",
                "Coverage Over Time"
            ],
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # Mutation status pie chart
        fig.add_trace(
            go.Pie(
                labels=["Killed", "Survived", "Error"],
                values=[
                    result.killed_mutations,
                    result.survived_mutations,
                    len(result.errors)
                ],
                hole=0.4,
                name="Status"
            ),
            row=1,
            col=1
        )
        
        # Operator performance bar chart
        operators = []
        killed = []
        survived = []
        errors = []
        
        for op, stats in result.operator_stats.items():
            operators.append(op)
            killed.append(stats["killed"])
            survived.append(stats["survived"])
            errors.append(stats.get("errors", 0))
        
        fig.add_trace(
            go.Bar(
                x=operators,
                y=killed,
                name="Killed",
                marker_color="green"
            ),
            row=1,
            col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=operators,
                y=survived,
                name="Survived",
                marker_color="red"
            ),
            row=1,
            col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=operators,
                y=errors,
                name="Errors",
                marker_color="gray"
            ),
            row=1,
            col=2
        )
        
        # Error distribution
        error_counts = defaultdict(int)
        for error in result.errors:
            error_type = error.split(":")[0]
            error_counts[error_type] += 1
        
        fig.add_trace(
            go.Bar(
                x=list(error_counts.keys()),
                y=list(error_counts.values()),
                name="Errors"
            ),
            row=2,
            col=1
        )
        
        # Coverage trend (mock data - replace with real historical data)
        dates = pd.date_range(end=datetime.now(), periods=10)
        scores = np.linspace(0.5, result.mutation_score, 10)
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=scores,
                mode="lines+markers",
                name="Coverage"
            ),
            row=2,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            height=self.config.plot_height,
            width=self.config.plot_width,
            template=self.config.theme,
            showlegend=True,
            barmode="stack"
        )
        
        return fig
    
    def create_detail_plot(
        self,
        result: MutationTestResult
    ) -> go.Figure:
        """Create detailed visualization."""
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[
                "Mutation Survival Rate by Operator",
                "Error Types by Operator",
                "Mutation Coverage Map"
            ],
            row_heights=[0.3, 0.3, 0.4]
        )
        
        # Survival rate heatmap
        survival_data = []
        operators = []
        categories = ["killed", "survived", "errors"]
        
        for op, stats in result.operator_stats.items():
            operators.append(op)
            total = sum(stats.values())
            if total > 0:
                survival_data.append([
                    stats[cat] / total * 100
                    for cat in categories
                ])
            else:
                survival_data.append([0] * len(categories))
        
        fig.add_trace(
            go.Heatmap(
                z=survival_data,
                x=categories,
                y=operators,
                colorscale="RdYlGn",
                name="Survival Rate"
            ),
            row=1,
            col=1
        )
        
        # Error type breakdown
        error_types = defaultdict(lambda: defaultdict(int))
        for op, stats in result.operator_stats.items():
            for error in stats.get("error_details", []):
                error_type = error.split(":")[0]
                error_types[op][error_type] += 1
        
        error_data = []
        error_labels = sorted(set(
            error_type
            for op_errors in error_types.values()
            for error_type in op_errors
        ))
        
        for op in operators:
            error_data.append([
                error_types[op][label]
                for label in error_labels
            ])
        
        fig.add_trace(
            go.Heatmap(
                z=error_data,
                x=error_labels,
                y=operators,
                colorscale="Viridis",
                name="Error Types"
            ),
            row=2,
            col=1
        )
        
        # Coverage map
        coverage_data = np.random.random((len(operators), 10))  # Mock data
        fig.add_trace(
            go.Heatmap(
                z=coverage_data,
                x=[f"Test {i+1}" for i in range(10)],
                y=operators,
                colorscale="YlOrRd",
                name="Coverage"
            ),
            row=3,
            col=1
        )
        
        # Update layout
        fig.update_layout(
            height=self.config.plot_height * 1.5,
            width=self.config.plot_width,
            template=self.config.theme,
            showlegend=False
        )
        
        return fig
    
    async def create_animation(
        self,
        results: List[MutationTestResult]
    ) -> go.Figure:
        """Create animated visualization."""
        if not self.config.animation:
            return None
        
        fig = go.Figure()
        
        # Prepare frame data
        frames = []
        for i, result in enumerate(results):
            frame_data = []
            
            # Add mutation status
            frame_data.append(
                go.Pie(
                    labels=["Killed", "Survived", "Error"],
                    values=[
                        result.killed_mutations,
                        result.survived_mutations,
                        len(result.errors)
                    ],
                    hole=0.4,
                    domain={"x": [0, 0.5], "y": [0.5, 1]}
                )
            )
            
            # Add operator performance
            operators = []
            stats = []
            for op, op_stats in result.operator_stats.items():
                operators.append(op)
                stats.append([
                    op_stats["killed"],
                    op_stats["survived"],
                    op_stats.get("errors", 0)
                ])
            
            frame_data.append(
                go.Bar(
                    x=operators,
                    y=[s[0] for s in stats],
                    name="Killed",
                    marker_color="green",
                    domain={"x": [0.6, 1], "y": [0.5, 1]}
                )
            )
            
            frames.append(go.Frame(
                data=frame_data,
                name=f"frame_{i}"
            ))
        
        # Add frames to figure
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "type": "buttons"
            }],
            sliders=[{
                "currentvalue": {"prefix": "Frame: "},
                "steps": [
                    {
                        "args": [[f"frame_{i}"]],
                        "label": str(i),
                        "method": "animate"
                    }
                    for i in range(len(frames))
                ]
            }]
        )
        
        return fig
    
    async def save_visualizations(
        self,
        result: MutationTestResult,
        prefix: str = "mutation"
    ):
        """Save visualizations to files."""
        if not self.config.export_path:
            return
        
        export_dir = Path(self.config.export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary plot
        summary_fig = self.create_summary_plot(result)
        summary_fig.write_html(
            export_dir / f"{prefix}_summary.html"
        )
        
        # Save detail plot
        detail_fig = self.create_detail_plot(result)
        detail_fig.write_html(
            export_dir / f"{prefix}_detail.html"
        )
        
        # Save raw data
        with open(export_dir / f"{prefix}_data.json", "w") as f:
            json.dump({
                "total_mutations": result.total_mutations,
                "killed_mutations": result.killed_mutations,
                "survived_mutations": result.survived_mutations,
                "operator_stats": result.operator_stats,
                "errors": result.errors,
                "mutation_score": result.mutation_score
            }, f, indent=2)

def create_mutation_visualizer(
    config: Optional[VisualizationConfig] = None
) -> MutationVisualizer:
    """Create mutation visualizer."""
    return MutationVisualizer(config)

if __name__ == "__main__":
    # Example usage
    from tests.proxy.analysis.monitoring.test_mutation_coverage import (
        test_mutation_detection,
        mutator,
        test_state
    )
    
    async def main():
        # Run mutation tests
        result = await test_mutation_detection(mutator, test_state)
        
        # Create visualizations
        visualizer = create_mutation_visualizer(
            VisualizationConfig(
                export_path=Path("mutation_reports")
            )
        )
        
        # Generate and save visualizations
        await visualizer.save_visualizations(result)
        
        print("Visualizations saved to mutation_reports/")
    
    asyncio.run(main())
