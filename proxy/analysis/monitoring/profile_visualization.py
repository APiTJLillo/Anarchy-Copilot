"""Visualization tools for profiling results."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, linkage

from .prediction_profiling import PredictionProfiler, ProfilingConfig

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for profile visualization."""
    width: int = 1200
    height: int = 800
    dark_mode: bool = False
    interactive: bool = True
    animation_duration: int = 500
    show_tooltips: bool = True
    max_nodes: int = 50
    output_path: Optional[Path] = None

class ProfileVisualizer:
    """Visualize profiling results."""
    
    def __init__(
        self,
        profiler: PredictionProfiler,
        config: VisualizationConfig
    ):
        self.profiler = profiler
        self.config = config
    
    def create_dashboard(self) -> go.Figure:
        """Create comprehensive profiling dashboard."""
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "Execution Time Distribution",
                "Memory Usage",
                "Call Graph",
                "Function Heatmap",
                "Performance Timeline",
                "Bottleneck Analysis"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter3d"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # Add execution time distribution
        self._add_time_distribution(fig, row=1, col=1)
        
        # Add memory usage visualization
        self._add_memory_usage(fig, row=1, col=2)
        
        # Add call graph visualization
        self._add_call_graph(fig, row=2, col=1)
        
        # Add function heatmap
        self._add_function_heatmap(fig, row=2, col=2)
        
        # Add performance timeline
        self._add_performance_timeline(fig, row=3, col=1)
        
        # Add bottleneck analysis
        self._add_bottleneck_analysis(fig, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            showlegend=True
        )
        
        return fig
    
    def create_call_graph_visualization(self) -> go.Figure:
        """Create interactive call graph visualization."""
        stats = self.profiler.function_stats
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for func_name, func_stats in stats.items():
            G.add_node(
                func_name,
                calls=func_stats["call_count"],
                time=func_stats["execution_time"]
            )
        
        # Add edges
        if self.profiler.current_profile:
            for func, (_, _, _, _, callers) in self.profiler.current_profile.stats.items():
                for caller in callers:
                    G.add_edge(
                        f"{caller[2]}:{caller[0]}({caller[1]})",
                        f"{func[2]}:{func[0]}({func[1]})"
                    )
        
        # Get layout
        pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()))
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                name="Calls"
            )
        )
        
        # Add nodes
        node_x = []
        node_y = []
        node_color = []
        node_size = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_color.append(G.nodes[node]["time"])
            node_size.append(np.sqrt(G.nodes[node]["calls"]) * 10)
            node_text.append(
                f"{node}<br>"
                f"Calls: {G.nodes[node]['calls']}<br>"
                f"Time: {G.nodes[node]['time']:.2f}s"
            )
        
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hoverinfo="text",
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale="YlOrRd",
                    color=node_color,
                    size=node_size,
                    colorbar=dict(
                        thickness=15,
                        title="Execution Time (s)",
                        xanchor="left",
                        titleside="right"
                    )
                ),
                name="Functions"
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Function Call Graph",
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=800
        )
        
        return fig
    
    def create_memory_timeline(self) -> go.Figure:
        """Create memory usage timeline visualization."""
        fig = go.Figure()
        
        if not self.profiler.memory_snapshots:
            return fig
        
        # Extract memory data
        timestamps = []
        total_memory = []
        allocation_counts = []
        
        for snapshot in self.profiler.memory_snapshots:
            stats = snapshot.statistics("traceback")
            timestamps.append(snapshot.timestamp)
            total_memory.append(sum(stat.size for stat in stats))
            allocation_counts.append(len(stats))
        
        # Add total memory line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=[m/1024/1024 for m in total_memory],
                mode="lines",
                name="Total Memory (MB)",
                line=dict(color="blue", width=2)
            )
        )
        
        # Add allocation count
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=allocation_counts,
                mode="lines",
                name="Allocation Count",
                yaxis="y2",
                line=dict(color="red", width=2)
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Memory Usage Timeline",
            xaxis_title="Time",
            yaxis_title="Memory (MB)",
            yaxis2=dict(
                title="Allocation Count",
                overlaying="y",
                side="right"
            ),
            showlegend=True
        )
        
        return fig
    
    def create_performance_sunburst(self) -> go.Figure:
        """Create hierarchical performance visualization."""
        stats = self.profiler.function_stats
        
        # Prepare data
        data = []
        for func_name, func_stats in stats.items():
            module, func = func_name.split(":")
            data.append({
                "id": func_name,
                "parent": module,
                "value": func_stats["execution_time"],
                "color": func_stats["call_count"]
            })
        
        # Add module nodes
        modules = set(d["parent"] for d in data)
        data.extend([
            {"id": module, "parent": "", "value": 0}
            for module in modules
        ])
        
        # Create figure
        fig = px.sunburst(
            pd.DataFrame(data),
            ids="id",
            parents="parent",
            values="value",
            color="color",
            title="Function Performance Hierarchy",
            color_continuous_scale="Viridis"
        )
        
        return fig
    
    def _add_time_distribution(
        self,
        fig: go.Figure,
        row: int,
        col: int
    ):
        """Add execution time distribution subplot."""
        times = [
            stats["execution_time"]
            for stats in self.profiler.function_stats.values()
        ]
        
        if not times:
            return
        
        fig.add_trace(
            go.Histogram(
                x=times,
                nbinsx=30,
                name="Execution Time"
            ),
            row=row,
            col=col
        )
        
        fig.update_xaxes(title_text="Time (s)", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)
    
    def _add_memory_usage(
        self,
        fig: go.Figure,
        row: int,
        col: int
    ):
        """Add memory usage subplot."""
        if not self.profiler.memory_snapshots:
            return
        
        stats = self.profiler._get_memory_stats()
        allocations = stats.get("top_allocations", [])
        
        if not allocations:
            return
        
        sizes = [a["size"]/1024/1024 for a in allocations]  # Convert to MB
        locations = [a["traceback"][0] for a in allocations]
        
        fig.add_trace(
            go.Bar(
                x=locations,
                y=sizes,
                name="Memory Usage"
            ),
            row=row,
            col=col
        )
        
        fig.update_xaxes(
            title_text="Location",
            tickangle=45,
            row=row,
            col=col
        )
        fig.update_yaxes(title_text="Memory (MB)", row=row, col=col)
    
    def _add_call_graph(
        self,
        fig: go.Figure,
        row: int,
        col: int
    ):
        """Add call graph subplot."""
        stats = self.profiler.function_stats
        if not stats:
            return
        
        # Create graph
        G = nx.DiGraph()
        for name, stat in stats.items():
            G.add_node(name, **stat)
        
        pos = nx.spring_layout(G)
        
        # Add nodes
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [
            f"{node}<br>"
            f"Calls: {G.nodes[node]['call_count']}<br>"
            f"Time: {G.nodes[node]['execution_time']:.2f}s"
            for node in G.nodes()
        ]
        
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=list(G.nodes()),
                textposition="bottom center",
                hovertext=node_text,
                name="Functions",
                marker=dict(
                    size=10,
                    color=[G.nodes[node]["execution_time"] for node in G.nodes()],
                    colorscale="Viridis",
                    showscale=True
                )
            ),
            row=row,
            col=col
        )
    
    def _add_function_heatmap(
        self,
        fig: go.Figure,
        row: int,
        col: int
    ):
        """Add function performance heatmap subplot."""
        stats = self.profiler.function_stats
        if not stats:
            return
        
        # Prepare data
        names = list(stats.keys())
        metrics = ["call_count", "execution_time", "memory_delta"]
        data = np.zeros((len(names), len(metrics)))
        
        for i, name in enumerate(names):
            for j, metric in enumerate(metrics):
                data[i, j] = stats[name].get(metric, 0)
        
        # Normalize data
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        
        fig.add_trace(
            go.Heatmap(
                z=data,
                x=metrics,
                y=names,
                colorscale="RdBu",
                name="Performance"
            ),
            row=row,
            col=col
        )
    
    def _add_performance_timeline(
        self,
        fig: go.Figure,
        row: int,
        col: int
    ):
        """Add performance timeline subplot."""
        stats = self.profiler.function_stats
        if not stats:
            return
        
        # Create timeline
        names = []
        start_times = []
        durations = []
        
        for name, stat in stats.items():
            if "timestamp" in stat:
                names.append(name)
                start_times.append(
                    pd.to_datetime(stat["timestamp"]).timestamp()
                )
                durations.append(stat["execution_time"])
        
        if not names:
            return
        
        fig.add_trace(
            go.Scatter(
                x=start_times,
                y=durations,
                mode="markers+lines",
                name="Performance",
                text=names,
                hovertemplate=(
                    "Function: %{text}<br>"
                    "Time: %{y:.2f}s<br>"
                    "<extra></extra>"
                )
            ),
            row=row,
            col=col
        )
        
        fig.update_xaxes(title_text="Time", row=row, col=col)
        fig.update_yaxes(title_text="Duration (s)", row=row, col=col)
    
    def _add_bottleneck_analysis(
        self,
        fig: go.Figure,
        row: int,
        col: int
    ):
        """Add bottleneck analysis subplot."""
        bottlenecks = self.profiler._identify_bottlenecks()
        if not bottlenecks:
            return
        
        # Prepare data
        types = []
        values = []
        thresholds = []
        locations = []
        
        for bottleneck in bottlenecks:
            types.append(bottleneck["type"])
            values.append(bottleneck["value"])
            thresholds.append(bottleneck.get("threshold", 0))
            locations.append(
                bottleneck.get("function", bottleneck.get("location", "unknown"))
            )
        
        fig.add_trace(
            go.Bar(
                x=locations,
                y=values,
                name="Actual",
                marker_color="red"
            ),
            row=row,
            col=col
        )
        
        fig.add_trace(
            go.Bar(
                x=locations,
                y=thresholds,
                name="Threshold",
                marker_color="green"
            ),
            row=row,
            col=col
        )
        
        fig.update_xaxes(
            title_text="Location",
            tickangle=45,
            row=row,
            col=col
        )
        fig.update_yaxes(title_text="Value", row=row, col=col)
    
    def save_visualizations(self):
        """Save all visualizations."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save dashboard
            dashboard = self.create_dashboard()
            dashboard.write_html(str(output_path / "profile_dashboard.html"))
            
            # Save call graph
            call_graph = self.create_call_graph_visualization()
            call_graph.write_html(str(output_path / "call_graph.html"))
            
            # Save memory timeline
            memory_timeline = self.create_memory_timeline()
            memory_timeline.write_html(str(output_path / "memory_timeline.html"))
            
            # Save performance sunburst
            sunburst = self.create_performance_sunburst()
            sunburst.write_html(str(output_path / "performance_sunburst.html"))
            
            logger.info(f"Saved visualizations to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save visualizations: {e}")

def create_profile_visualizer(
    profiler: PredictionProfiler,
    output_path: Optional[Path] = None
) -> ProfileVisualizer:
    """Create profile visualizer."""
    config = VisualizationConfig(output_path=output_path)
    return ProfileVisualizer(profiler, config)

if __name__ == "__main__":
    # Example usage
    from .prediction_profiling import create_prediction_profiler
    from .prediction_performance import create_prediction_performance
    from .realtime_prediction import create_realtime_prediction
    from .prediction_controls import create_interactive_controls
    from .prediction_visualization import create_prediction_visualizer
    from .easing_prediction import create_easing_predictor
    from .easing_statistics import create_easing_statistics
    from .easing_metrics import create_easing_metrics
    from .easing_functions import create_easing_functions
    
    async def main():
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
        profile_viz = create_profile_visualizer(
            profiler,
            output_path=Path("profile_viz")
        )
        
        # Generate visualizations
        await profile_viz.save_visualizations()
    
    asyncio.run(main())
