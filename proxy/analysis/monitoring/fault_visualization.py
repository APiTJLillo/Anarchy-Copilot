"""Visualization tools for fault correlation analysis."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
from dataclasses import asdict

from .test_fault_correlation import FaultEvent, FaultCorrelationAnalyzer

logger = logging.getLogger(__name__)

class FaultVisualizer:
    """Create visualizations for fault analysis."""
    
    def __init__(
        self,
        analyzer: FaultCorrelationAnalyzer,
        output_dir: Optional[Path] = None
    ):
        self.analyzer = analyzer
        self.output_dir = output_dir or Path("fault_visualizations")
        self.output_dir.mkdir(exist_ok=True)
    
    def create_correlation_heatmap(self) -> go.Figure:
        """Create correlation heatmap between faults."""
        # Get fault time series
        fault_series = self.analyzer._create_fault_series()
        
        if fault_series.empty:
            return None
        
        # Calculate correlation matrix
        corr_matrix = fault_series.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Fault Correlation Heatmap",
            height=800,
            width=800
        )
        
        return fig
    
    def create_fault_timeline(self) -> go.Figure:
        """Create timeline of fault occurrences."""
        if not self.analyzer.fault_history:
            return None
        
        # Sort events by timestamp
        events = sorted(
            self.analyzer.fault_history,
            key=lambda e: e.timestamp
        )
        
        # Create figure
        fig = go.Figure()
        
        fault_types = set(e.fault_type for e in events)
        colors = px.colors.qualitative.Set3
        color_map = dict(zip(fault_types, colors[:len(fault_types)]))
        
        for fault_type in fault_types:
            fault_events = [
                e for e in events
                if e.fault_type == fault_type
            ]
            
            # Add trace for each fault type
            fig.add_trace(go.Scatter(
                x=[e.timestamp for e in fault_events],
                y=[fault_type for _ in fault_events],
                mode="markers",
                name=fault_type,
                marker=dict(
                    size=10,
                    color=color_map[fault_type],
                    symbol="square"
                ),
                customdata=[
                    {
                        "duration": str(e.duration),
                        "impact": e.impact_metrics,
                        "related": e.related_faults,
                        "depth": e.cascade_depth
                    }
                    for e in fault_events
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Time: %{x}<br>"
                    "Duration: %{customdata.duration}<br>"
                    "Impact: %{customdata.impact}<br>"
                    "Related: %{customdata.related}<br>"
                    "Cascade Depth: %{customdata.depth}"
                )
            ))
            
            # Add bars for duration
            fig.add_trace(go.Bar(
                x=[e.timestamp for e in fault_events],
                y=[fault_type for _ in fault_events],
                width=[e.duration.total_seconds() / 60 for e in fault_events],  # Minutes
                marker=dict(
                    color=color_map[fault_type],
                    opacity=0.3
                ),
                showlegend=False
            ))
        
        fig.update_layout(
            title="Fault Timeline",
            xaxis_title="Time",
            yaxis_title="Fault Type",
            height=400,
            barmode="overlay",
            showlegend=True
        )
        
        return fig
    
    def create_cascade_graph(self) -> go.Figure:
        """Create graph visualization of fault cascades."""
        if not self.analyzer.fault_graph:
            return None
        
        # Create layout
        pos = nx.spring_layout(self.analyzer.fault_graph)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in self.analyzer.fault_graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"Weight: {edge[2]['weight']}")
        
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="text",
            text=edge_text,
            mode="lines"
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in self.analyzer.fault_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Fault: {node}")
            node_size.append(
                self.analyzer.fault_graph.nodes[node]["events"] * 10
            )
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=node_text,
            textposition="bottom center",
            marker=dict(
                showscale=True,
                colorscale="YlOrRd",
                size=node_size,
                color=[
                    self.analyzer.fault_graph.nodes[node]["events"]
                    for node in self.analyzer.fault_graph.nodes()
                ],
                line_width=2
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Fault Cascade Graph",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
    
    def create_impact_analysis(self) -> go.Figure:
        """Create impact analysis visualization."""
        if not self.analyzer.fault_history:
            return None
        
        # Calculate impacts for different metrics
        metrics = set()
        for event in self.analyzer.fault_history:
            metrics.update(event.impact_metrics.keys())
        
        # Create subplots
        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            subplot_titles=[f"{m} Impact by Fault Type" for m in metrics]
        )
        
        for i, metric in enumerate(metrics, 1):
            impacts = self.analyzer.calculate_fault_impact(metric)
            
            fig.add_trace(
                go.Bar(
                    x=list(impacts.keys()),
                    y=list(impacts.values()),
                    name=metric
                ),
                row=i,
                col=1
            )
        
        fig.update_layout(
            height=300 * len(metrics),
            showlegend=False,
            title="Fault Impact Analysis"
        )
        
        return fig
    
    def create_pattern_sankey(self) -> go.Figure:
        """Create Sankey diagram of fault patterns."""
        patterns = self.analyzer.detect_cascading_patterns()
        
        if not patterns:
            return None
        
        # Create nodes and links
        nodes = set()
        links = []
        
        for pattern in patterns:
            for i in range(len(pattern) - 1):
                nodes.add(pattern[i])
                nodes.add(pattern[i + 1])
                links.append((pattern[i], pattern[i + 1]))
        
        # Convert to indices
        node_map = {node: i for i, node in enumerate(nodes)}
        
        # Count link occurrences
        link_counts = {}
        for src, tgt in links:
            key = (node_map[src], node_map[tgt])
            link_counts[key] = link_counts.get(key, 0) + 1
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(nodes),
                color="blue"
            ),
            link=dict(
                source=[src for (src, _), _ in link_counts.items()],
                target=[tgt for (_, tgt), _ in link_counts.items()],
                value=[count for count in link_counts.values()]
            )
        )])
        
        fig.update_layout(
            title="Fault Pattern Flow",
            height=600
        )
        
        return fig
    
    def save_visualizations(self):
        """Save all visualizations."""
        visualizations = {
            "correlation": self.create_correlation_heatmap(),
            "timeline": self.create_fault_timeline(),
            "cascade": self.create_cascade_graph(),
            "impact": self.create_impact_analysis(),
            "pattern": self.create_pattern_sankey()
        }
        
        # Save individual visualizations
        for name, fig in visualizations.items():
            if fig is not None:
                fig.write_html(
                    str(self.output_dir / f"{name}_visualization.html")
                )
        
        # Create index page
        self._create_index_page(
            [name for name, fig in visualizations.items() if fig is not None]
        )
    
    def _create_index_page(self, visualizations: List[str]):
        """Create HTML index page."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fault Analysis Dashboard</title>
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
            <h1>Fault Analysis Dashboard</h1>
            <div id="visualizations">
        """
        
        for viz in visualizations:
            html += f"""
                <a class="viz-link" href="{viz}_visualization.html">
                    {viz.title()} Analysis
                </a>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / "index.html", "w") as f:
            f.write(html)

def create_fault_visualizations(analyzer: FaultCorrelationAnalyzer):
    """Generate fault analysis visualizations."""
    visualizer = FaultVisualizer(analyzer)
    visualizer.save_visualizations()

if __name__ == "__main__":
    # Example usage
    analyzer = FaultCorrelationAnalyzer()
    # Add fault events...
    create_fault_visualizations(analyzer)
