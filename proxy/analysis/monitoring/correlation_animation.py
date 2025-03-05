"""Animation tools for correlation visualization."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats
import networkx as nx
import logging
from pathlib import Path
import json

from .correlation_visualization import CorrelationVisualizer
from .metric_correlation import MetricCorrelation, CorrelationConfig

logger = logging.getLogger(__name__)

class CorrelationAnimator:
    """Create animated correlation visualizations."""
    
    def __init__(
        self,
        window_size: timedelta = timedelta(hours=24),
        step_size: timedelta = timedelta(minutes=30),
        output_dir: Optional[Path] = None
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.visualizer = CorrelationVisualizer(output_dir)
    
    def create_correlation_animation(
        self,
        metrics: Dict[str, pd.Series],
        min_correlation: float = 0.5
    ) -> go.Figure:
        """Create animated correlation matrix."""
        # Align timestamps and resample data
        df = pd.DataFrame(metrics)
        df = df.resample("1min").mean()
        
        # Calculate time windows
        end_time = df.index[-1]
        start_time = df.index[0]
        current = start_time + self.window_size
        
        frames = []
        while current <= end_time:
            window_data = df[current - self.window_size:current]
            correlation = window_data.corr()
            
            frame = go.Frame(
                data=[go.Heatmap(
                    z=correlation.values,
                    x=correlation.columns,
                    y=correlation.index,
                    colorscale="RdBu",
                    zmid=0,
                    text=np.round(correlation.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10}
                )],
                name=str(current)
            )
            frames.append(frame)
            current += self.step_size
        
        # Create base figure
        initial_corr = df[start_time:start_time + self.window_size].corr()
        fig = go.Figure(
            data=[go.Heatmap(
                z=initial_corr.values,
                x=initial_corr.columns,
                y=initial_corr.index,
                colorscale="RdBu",
                zmid=0,
                text=np.round(initial_corr.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10}
            )],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title="Correlation Evolution",
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
                "currentvalue": {
                    "prefix": "Time: ",
                    "visible": True
                },
                "steps": [
                    {
                        "args": [[str(f.name)], {"frame": {"duration": 0, "redraw": True}}],
                        "label": str(f.name),
                        "method": "animate"
                    }
                    for f in frames
                ]
            }]
        )
        
        return fig
    
    def create_network_animation(
        self,
        metrics: Dict[str, pd.Series],
        min_correlation: float = 0.5
    ) -> go.Figure:
        """Create animated correlation network."""
        df = pd.DataFrame(metrics)
        df = df.resample("1min").mean()
        
        # Calculate time windows
        end_time = df.index[-1]
        start_time = df.index[0]
        current = start_time + self.window_size
        
        # Calculate initial layout using all data
        full_corr = df.corr()
        G = nx.Graph()
        for i in range(len(full_corr.index)):
            for j in range(i + 1, len(full_corr.columns)):
                corr = full_corr.iloc[i, j]
                if abs(corr) >= min_correlation:
                    G.add_edge(
                        full_corr.index[i],
                        full_corr.columns[j],
                        weight=abs(corr)
                    )
        
        pos = nx.spring_layout(G)
        
        frames = []
        while current <= end_time:
            window_data = df[current - self.window_size:current]
            correlation = window_data.corr()
            
            # Create graph for this window
            G = nx.Graph()
            edge_x = []
            edge_y = []
            edge_colors = []
            
            for i in range(len(correlation.index)):
                for j in range(i + 1, len(correlation.columns)):
                    corr = correlation.iloc[i, j]
                    if abs(corr) >= min_correlation:
                        n1 = correlation.index[i]
                        n2 = correlation.columns[j]
                        G.add_edge(n1, n2, weight=abs(corr))
                        
                        x0, y0 = pos[n1]
                        x1, y1 = pos[n2]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        edge_colors.extend([abs(corr), abs(corr), None])
            
            # Create node traces
            node_x = []
            node_y = []
            node_sizes = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_sizes.append(G.degree(node) * 10)
            
            frame = go.Frame(
                data=[
                    go.Scatter(  # Edges
                        x=edge_x,
                        y=edge_y,
                        mode="lines",
                        line=dict(
                            width=1,
                            color=edge_colors,
                            colorscale="Viridis"
                        ),
                        hoverinfo="none"
                    ),
                    go.Scatter(  # Nodes
                        x=node_x,
                        y=node_y,
                        mode="markers+text",
                        text=list(G.nodes()),
                        textposition="top center",
                        marker=dict(
                            size=node_sizes,
                            color="lightblue",
                            line=dict(width=2)
                        )
                    )
                ],
                name=str(current)
            )
            frames.append(frame)
            current += self.step_size
        
        # Create base figure with initial state
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title="Correlation Network Evolution",
            showlegend=False,
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
                "currentvalue": {
                    "prefix": "Time: ",
                    "visible": True
                },
                "steps": [
                    {
                        "args": [[str(f.name)], {"frame": {"duration": 0, "redraw": True}}],
                        "label": str(f.name),
                        "method": "animate"
                    }
                    for f in frames
                ]
            }]
        )
        
        return fig
    
    def create_cluster_animation(
        self,
        metrics: Dict[str, pd.Series]
    ) -> go.Figure:
        """Create animated cluster evolution."""
        df = pd.DataFrame(metrics)
        df = df.resample("1min").mean()
        
        # Calculate time windows
        end_time = df.index[-1]
        start_time = df.index[0]
        current = start_time + self.window_size
        
        frames = []
        while current <= end_time:
            window_data = df[current - self.window_size:current]
            correlation = window_data.corr()
            
            # Create graph
            G = nx.Graph()
            for i in range(len(correlation.index)):
                for j in range(i + 1, len(correlation.columns)):
                    corr = correlation.iloc[i, j]
                    if abs(corr) >= 0.5:  # Threshold for clustering
                        G.add_edge(
                            correlation.index[i],
                            correlation.columns[j],
                            weight=abs(corr)
                        )
            
            # Find communities
            try:
                import community
                partition = community.best_partition(G)
                clusters = defaultdict(list)
                for node, cluster_id in partition.items():
                    clusters[cluster_id].append(node)
                
                # Create cluster visualization
                x_positions = []
                y_positions = []
                colors = []
                texts = []
                sizes = []
                
                for cluster_id, nodes in clusters.items():
                    angle = 2 * np.pi * cluster_id / len(clusters)
                    radius = 1
                    base_x = radius * np.cos(angle)
                    base_y = radius * np.sin(angle)
                    
                    for node in nodes:
                        x_positions.append(base_x + np.random.normal(0, 0.1))
                        y_positions.append(base_y + np.random.normal(0, 0.1))
                        colors.append(cluster_id)
                        texts.append(node)
                        sizes.append(G.degree(node) * 10)
                
                frame = go.Frame(
                    data=[go.Scatter(
                        x=x_positions,
                        y=y_positions,
                        mode="markers+text",
                        text=texts,
                        textposition="top center",
                        marker=dict(
                            size=sizes,
                            color=colors,
                            colorscale="Viridis",
                            showscale=True
                        ),
                        hoverinfo="text"
                    )],
                    name=str(current)
                )
                frames.append(frame)
                
            except ImportError:
                logger.warning("python-louvain package not found, skipping clustering")
                break
            
            current += self.step_size
        
        if not frames:
            return self.visualizer._empty_figure("No clusters found")
        
        # Create base figure
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title="Cluster Evolution",
            showlegend=False,
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
                "currentvalue": {
                    "prefix": "Time: ",
                    "visible": True
                },
                "steps": [
                    {
                        "args": [[str(f.name)], {"frame": {"duration": 0, "redraw": True}}],
                        "label": str(f.name),
                        "method": "animate"
                    }
                    for f in frames
                ]
            }]
        )
        
        return fig

def create_correlation_animations(
    metrics: Dict[str, pd.Series],
    window_size: timedelta = timedelta(hours=24),
    output_dir: Optional[Path] = None
) -> Dict[str, go.Figure]:
    """Create all correlation animations."""
    animator = CorrelationAnimator(window_size, output_dir=output_dir)
    
    return {
        "matrix": animator.create_correlation_animation(metrics),
        "network": animator.create_network_animation(metrics),
        "clusters": animator.create_cluster_animation(metrics)
    }

if __name__ == "__main__":
    # Example usage
    from .metric_correlation import analyze_metric_correlations
    from .adaptive_thresholds import create_adaptive_thresholds
    
    # Create sample data
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=7),
        end=datetime.now(),
        freq="1min"
    )
    
    metrics = {}
    for hour in range(len(timestamps)):
        cpu_value = 50 + 10 * np.sin(hour * np.pi / 12)
        mem_value = cpu_value * 0.8 + 20
        io_value = 100 + 5 * np.sin(hour * np.pi / 8)
        
        metrics["cpu"] = pd.Series(
            cpu_value + np.random.normal(0, 2),
            index=timestamps
        )
        metrics["memory"] = pd.Series(
            mem_value + np.random.normal(0, 2),
            index=timestamps
        )
        metrics["io"] = pd.Series(
            io_value + np.random.normal(0, 5),
            index=timestamps
        )
    
    # Create animations
    animations = create_correlation_animations(metrics)
    
    # Save animations
    output_dir = Path("correlation_animations")
    output_dir.mkdir(exist_ok=True)
    
    for name, fig in animations.items():
        fig.write_html(str(output_dir / f"{name}_animation.html"))
