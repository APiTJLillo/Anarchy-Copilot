"""Visualization tools for metric correlations."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import logging
from pathlib import Path

from .metric_correlation import MetricCorrelation, CorrelationConfig

logger = logging.getLogger(__name__)

class CorrelationVisualizer:
    """Visualize metric correlations."""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None
    ):
        self.output_dir = output_dir or Path("correlation_visualizations")
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_correlation_matrix(
        self,
        correlation_matrix: pd.DataFrame
    ) -> go.Figure:
        """Create correlation matrix heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale="RdBu",
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Metric Correlation Matrix",
            xaxis_title="Metric",
            yaxis_title="Metric",
            height=800,
            width=800
        )
        
        return fig
    
    def plot_correlation_network(
        self,
        graph: nx.Graph,
        min_weight: float = 0.3
    ) -> go.Figure:
        """Create correlation network visualization."""
        # Filter edges by weight
        edges = [
            (u, v) for (u, v, d) in graph.edges(data=True)
            if d["weight"] >= min_weight
        ]
        
        # Create filtered graph
        filtered_graph = nx.Graph()
        filtered_graph.add_edges_from(edges)
        
        # Calculate layout
        pos = nx.spring_layout(filtered_graph, k=1/np.sqrt(len(graph)))
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in filtered_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(graph[edge[0]][edge[1]]["weight"])
        
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(
                width=1,
                color=edge_weights,
                colorscale="Viridis"
            ),
            hoverinfo="none",
            mode="lines"
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in filtered_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            node_size.append(filtered_graph.degree(node) * 10)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(
                size=node_size,
                color="lightblue",
                line=dict(width=2)
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Metric Correlation Network",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                height=800,
                width=800
            )
        )
        
        return fig
    
    def plot_causality_graph(
        self,
        causal_graph: nx.DiGraph,
        min_correlation: float = 0.5
    ) -> go.Figure:
        """Create causality graph visualization."""
        # Filter edges by correlation
        edges = [
            (u, v) for (u, v, d) in causal_graph.edges(data=True)
            if abs(d["correlation"]) >= min_correlation
        ]
        
        # Create filtered graph
        filtered_graph = nx.DiGraph()
        filtered_graph.add_edges_from(edges)
        
        # Calculate layout
        pos = nx.spring_layout(filtered_graph)
        
        # Create edge traces
        edge_traces = []
        
        for edge in filtered_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Calculate arrow position
            dx = x1 - x0
            dy = y1 - y0
            dist = np.sqrt(dx*dx + dy*dy)
            if dist == 0:
                continue
                
            # Shorten arrow to not overlap with nodes
            x1 = x0 + dx * 0.8
            y1 = y0 + dy * 0.8
            
            edge_trace = go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines+markers",
                line=dict(
                    width=2,
                    color="gray"
                ),
                marker=dict(
                    size=10,
                    symbol="arrow",
                    angle=np.arctan2(dy, dx) * 180 / np.pi,
                    color="gray"
                ),
                hoverinfo="text",
                hovertext=f"{edge[0]} → {edge[1]}"
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_trace = go.Scatter(
            x=[pos[node][0] for node in filtered_graph.nodes()],
            y=[pos[node][1] for node in filtered_graph.nodes()],
            mode="markers+text",
            text=list(filtered_graph.nodes()),
            textposition="top center",
            marker=dict(
                size=20,
                color="lightblue",
                line=dict(width=2)
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[*edge_traces, node_trace],
            layout=go.Layout(
                title="Metric Causality Graph",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                height=800,
                width=800
            )
        )
        
        return fig
    
    def plot_correlation_clusters(
        self,
        clusters: List[Dict[str, Any]]
    ) -> go.Figure:
        """Create cluster visualization."""
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Cluster Sizes",
                "Cluster Densities"
            )
        )
        
        # Add size bars
        fig.add_trace(
            go.Bar(
                x=[f"Cluster {c['id']}" for c in clusters],
                y=[c["size"] for c in clusters],
                name="Size"
            ),
            row=1,
            col=1
        )
        
        # Add density bars
        fig.add_trace(
            go.Bar(
                x=[f"Cluster {c['id']}" for c in clusters],
                y=[c["density"] for c in clusters],
                name="Density"
            ),
            row=2,
            col=1
        )
        
        fig.update_layout(
            title="Correlation Clusters",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_temporal_correlations(
        self,
        metrics: Dict[str, pd.Series],
        window: str = "1h"
    ) -> go.Figure:
        """Create temporal correlation visualization."""
        # Calculate rolling correlations
        df = pd.DataFrame(metrics)
        correlations = df.rolling(window).corr()
        
        # Create figure
        fig = go.Figure()
        
        # Add correlation lines for each pair
        for i in range(len(df.columns)):
            for j in range(i + 1, len(df.columns)):
                metric1 = df.columns[i]
                metric2 = df.columns[j]
                
                corr = correlations[metric1][metric2].dropna()
                
                fig.add_trace(go.Scatter(
                    x=corr.index,
                    y=corr.values,
                    name=f"{metric1} vs {metric2}",
                    mode="lines"
                ))
        
        fig.update_layout(
            title=f"Rolling Correlations (Window: {window})",
            xaxis_title="Time",
            yaxis_title="Correlation",
            height=400
        )
        
        return fig
    
    def create_dashboard(
        self,
        correlation_analysis: Dict[str, Any],
        metrics: Optional[Dict[str, pd.Series]] = None
    ) -> None:
        """Create comprehensive correlation dashboard."""
        # Create visualizations
        matrix_fig = self.plot_correlation_matrix(
            pd.DataFrame(correlation_analysis["correlation_matrix"])
        )
        
        network_fig = self.plot_correlation_network(
            correlation_analysis["graph"]
        )
        
        causality_fig = self.plot_causality_graph(
            correlation_analysis["causal_graph"]
        )
        
        cluster_fig = self.plot_correlation_clusters(
            correlation_analysis["correlation_clusters"]
        )
        
        # Create temporal correlation plot if metrics provided
        temporal_fig = None
        if metrics:
            temporal_fig = self.plot_temporal_correlations(metrics)
        
        # Save figures
        matrix_fig.write_html(str(self.output_dir / "correlation_matrix.html"))
        network_fig.write_html(str(self.output_dir / "correlation_network.html"))
        causality_fig.write_html(str(self.output_dir / "causality_graph.html"))
        cluster_fig.write_html(str(self.output_dir / "correlation_clusters.html"))
        
        if temporal_fig:
            temporal_fig.write_html(str(self.output_dir / "temporal_correlations.html"))
        
        # Create index page
        self._create_index_page(temporal_fig is not None)
    
    def _create_index_page(self, has_temporal: bool):
        """Create HTML index page."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Metric Correlation Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 2em; }
                iframe {
                    width: 100%;
                    height: 800px;
                    border: none;
                    margin: 1em 0;
                }
            </style>
        </head>
        <body>
            <h1>Metric Correlation Analysis</h1>
            
            <h2>Correlation Matrix</h2>
            <iframe src="correlation_matrix.html"></iframe>
            
            <h2>Correlation Network</h2>
            <iframe src="correlation_network.html"></iframe>
            
            <h2>Causality Graph</h2>
            <iframe src="causality_graph.html"></iframe>
            
            <h2>Correlation Clusters</h2>
            <iframe src="correlation_clusters.html"></iframe>
        """
        
        if has_temporal:
            html += """
            <h2>Temporal Correlations</h2>
            <iframe src="temporal_correlations.html"></iframe>
            """
        
        html += """
        </body>
        </html>
        """
        
        with open(self.output_dir / "index.html", "w") as f:
            f.write(html)

def visualize_correlations(
    correlation_analysis: Dict[str, Any],
    metrics: Optional[Dict[str, pd.Series]] = None,
    output_dir: Optional[Path] = None
) -> None:
    """Create correlation visualizations."""
    visualizer = CorrelationVisualizer(output_dir)
    visualizer.create_dashboard(correlation_analysis, metrics)

if __name__ == "__main__":
    # Example usage
    from .metric_correlation import analyze_metric_correlations
    from .adaptive_thresholds import create_adaptive_thresholds
    
    thresholds = create_adaptive_thresholds()
    
    # Add sample data
    for hour in range(24 * 7):
        cpu_value = 50 + 10 * np.sin(hour * np.pi / 12)
        mem_value = cpu_value * 0.8 + 20
        io_value = 100 + 5 * np.sin(hour * np.pi / 8)
        
        metrics = {
            "cpu": cpu_value + np.random.normal(0, 2),
            "memory": mem_value + np.random.normal(0, 2),
            "io": io_value + np.random.normal(0, 5)
        }
        
        timestamp = datetime.now() - timedelta(hours=24*7-hour)
        for name, value in metrics.items():
            thresholds.add_value(name, "value", value, timestamp)
    
    # Analyze and visualize
    analysis = analyze_metric_correlations(thresholds)
    visualize_correlations(analysis)
