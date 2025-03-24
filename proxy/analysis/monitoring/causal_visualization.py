"""Visualization tools for causal analysis."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
import json
from pathlib import Path
import logging

from .fault_causality import (
    FaultCausalityAnalyzer,
    CausalLink,
    RootCause
)

logger = logging.getLogger(__name__)

class CausalVisualizer:
    """Create visualizations for causal analysis."""
    
    def __init__(
        self,
        analyzer: FaultCausalityAnalyzer,
        output_dir: Optional[Path] = None
    ):
        self.analyzer = analyzer
        self.output_dir = output_dir or Path("causal_visualizations")
        self.output_dir.mkdir(exist_ok=True)
    
    def create_causal_graph(self) -> go.Figure:
        """Create interactive causal graph visualization."""
        if not self.analyzer.causal_graph:
            return None
        
        # Create layout
        pos = nx.spring_layout(self.analyzer.causal_graph)
        
        # Create node traces by impact level
        node_traces = []
        
        # Group nodes by impact level
        impact_levels = {}
        for node in self.analyzer.causal_graph.nodes():
            impacts = []
            for event in self.analyzer.fault_history:
                if event.fault_type == node:
                    impacts.extend(event.impact_metrics.values())
            
            avg_impact = np.mean(impacts) if impacts else 0
            if avg_impact < 0.3:
                impact_levels.setdefault("low", []).append(node)
            elif avg_impact < 0.7:
                impact_levels.setdefault("medium", []).append(node)
            else:
                impact_levels.setdefault("high", []).append(node)
        
        colors = {"low": "green", "medium": "orange", "high": "red"}
        
        for level, nodes in impact_levels.items():
            x = []
            y = []
            text = []
            
            for node in nodes:
                x.append(pos[node][0])
                y.append(pos[node][1])
                text.append(self._create_node_hover_text(node))
            
            node_traces.append(go.Scatter(
                x=x,
                y=y,
                mode="markers+text",
                marker=dict(
                    size=20,
                    color=colors[level],
                    line=dict(width=2)
                ),
                text=[n.split("_")[-1] for n in nodes],
                textposition="bottom center",
                name=f"{level.title()} Impact",
                hovertext=text,
                hoverinfo="text"
            ))
        
        # Create edge traces by confidence
        edge_traces = []
        edges = self.analyzer.find_causal_links()
        
        for confidence_range in [(0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
            edge_x = []
            edge_y = []
            edge_text = []
            
            filtered_edges = [
                e for e in edges
                if confidence_range[0] <= e.confidence < confidence_range[1]
            ]
            
            for edge in filtered_edges:
                x0, y0 = pos[edge.source]
                x1, y1 = pos[edge.target]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_text.append(self._create_edge_hover_text(edge))
            
            if edge_x:
                edge_traces.append(go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(
                        width=2,
                        color=f"rgba(100,100,100,{confidence_range[0]})"
                    ),
                    hovertext=edge_text,
                    hoverinfo="text",
                    name=f"Confidence {confidence_range[0]:.1f}-{confidence_range[1]:.1f}"
                ))
        
        # Create figure
        fig = go.Figure(
            data=[*node_traces, *edge_traces],
            layout=go.Layout(
                title="Fault Causal Graph",
                titlefont_size=16,
                showlegend=True,
                hovermode="closest",
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
        )
        
        return fig
    
    def create_root_cause_sunburst(self) -> go.Figure:
        """Create sunburst diagram of root causes and effects."""
        root_causes = self.analyzer.identify_root_causes()
        
        if not root_causes:
            return None
        
        # Prepare data for sunburst
        labels = []
        parents = []
        values = []
        colors = []
        
        # Add root causes
        for rc in root_causes:
            labels.append(rc.fault_type)
            parents.append("")
            values.append(rc.probability * 100)
            colors.append("red")
            
            # Add cascading effects
            for effect in rc.cascading_effects:
                labels.append(effect)
                parents.append(rc.fault_type)
                
                # Find impact value
                impact = 0
                for event in self.analyzer.fault_history:
                    if event.fault_type == effect:
                        impact = max(impact, sum(event.impact_metrics.values()))
                
                values.append(impact * 100)
                colors.append("orange")
                
                # Add metrics affected
                for metric, value in rc.impact_metrics.items():
                    metric_label = f"{effect}_{metric}"
                    labels.append(metric_label)
                    parents.append(effect)
                    values.append(value * 100)
                    colors.append("yellow")
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            branchvalues="total"
        ))
        
        fig.update_layout(
            title="Root Causes and Cascading Effects",
            width=800,
            height=800
        )
        
        return fig
    
    def create_temporal_pattern_view(self) -> go.Figure:
        """Create temporal view of causal patterns."""
        if not self.analyzer.fault_history:
            return None
        
        fig = make_subplots(rows=2, cols=1)
        
        # Create time series of fault occurrences
        events_df = pd.DataFrame([
            {
                "timestamp": e.timestamp,
                "fault_type": e.fault_type,
                "impact": sum(e.impact_metrics.values())
            }
            for e in sorted(
                self.analyzer.fault_history,
                key=lambda x: x.timestamp
            )
        ])
        
        # Plot fault occurrences
        for fault_type in events_df["fault_type"].unique():
            fault_events = events_df[
                events_df["fault_type"] == fault_type
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=fault_events["timestamp"],
                    y=fault_events["impact"],
                    name=fault_type,
                    mode="markers+lines",
                    marker=dict(size=10)
                ),
                row=1, col=1
            )
        
        # Add causal connections
        causal_links = self.analyzer.find_causal_links()
        y_positions = np.linspace(0, 1, len(causal_links))
        
        for i, link in enumerate(causal_links):
            source_events = events_df[
                events_df["fault_type"] == link.source
            ]
            target_events = events_df[
                events_df["fault_type"] == link.target
            ]
            
            for _, source in source_events.iterrows():
                # Find corresponding target events
                related_targets = target_events[
                    (target_events["timestamp"] > source["timestamp"]) &
                    (target_events["timestamp"] <= source["timestamp"] + link.time_lag)
                ]
                
                for _, target in related_targets.iterrows():
                    fig.add_trace(
                        go.Scatter(
                            x=[source["timestamp"], target["timestamp"]],
                            y=[y_positions[i], y_positions[i]],
                            mode="lines",
                            line=dict(
                                width=1,
                                color=f"rgba(100,100,100,{link.confidence})"
                            ),
                            name=f"{link.source} -> {link.target}",
                            showlegend=False
                        ),
                        row=2, col=1
                    )
        
        fig.update_layout(
            title="Temporal Fault Patterns",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_impact_flow_sankey(self) -> go.Figure:
        """Create Sankey diagram of fault impact flow."""
        causal_links = self.analyzer.find_causal_links()
        
        if not causal_links:
            return None
        
        # Create nodes list
        nodes = set()
        for link in causal_links:
            nodes.add(link.source)
            nodes.add(link.target)
        
        node_indices = {node: i for i, node in enumerate(nodes)}
        
        # Prepare Sankey data
        link_values = []
        link_sources = []
        link_targets = []
        link_colors = []
        
        for link in causal_links:
            source_idx = node_indices[link.source]
            target_idx = node_indices[link.target]
            
            link_sources.append(source_idx)
            link_targets.append(target_idx)
            link_values.append(link.probability * 100)
            link_colors.append(f"rgba(100,100,100,{link.confidence})")
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(nodes),
                color="blue"
            ),
            link=dict(
                source=link_sources,
                target=link_targets,
                value=link_values,
                color=link_colors
            )
        )])
        
        fig.update_layout(
            title="Fault Impact Flow",
            font_size=10,
            height=600
        )
        
        return fig
    
    def _create_node_hover_text(self, node: str) -> str:
        """Create hover text for node."""
        node_events = [
            e for e in self.analyzer.fault_history
            if e.fault_type == node
        ]
        
        if not node_events:
            return node
        
        avg_metrics = defaultdict(list)
        for event in node_events:
            for metric, value in event.impact_metrics.items():
                avg_metrics[metric].append(value)
        
        text = [f"<b>{node}</b>"]
        text.append(f"Occurrences: {len(node_events)}")
        
        for metric, values in avg_metrics.items():
            text.append(f"{metric}: {np.mean(values):.2f}")
        
        return "<br>".join(text)
    
    def _create_edge_hover_text(self, edge: CausalLink) -> str:
        """Create hover text for edge."""
        text = [
            f"<b>{edge.source} -> {edge.target}</b>",
            f"Strength: {edge.strength:.2f}",
            f"Confidence: {edge.confidence:.2f}",
            f"Time Lag: {edge.time_lag}",
            f"Probability: {edge.probability:.2f}"
        ]
        
        if edge.evidence:
            text.append("<br><b>Evidence:</b>")
            text.extend(edge.evidence)
        
        return "<br>".join(text)
    
    def save_visualizations(self):
        """Save all visualizations."""
        visualizations = {
            "causal_graph": self.create_causal_graph(),
            "root_cause_sunburst": self.create_root_cause_sunburst(),
            "temporal_pattern": self.create_temporal_pattern_view(),
            "impact_flow": self.create_impact_flow_sankey()
        }
        
        # Save individual visualizations
        for name, fig in visualizations.items():
            if fig is not None:
                fig.write_html(
                    str(self.output_dir / f"{name}.html")
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
            <title>Causal Analysis Dashboard</title>
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
            <h1>Causal Analysis Dashboard</h1>
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

def visualize_causality(analyzer: FaultCausalityAnalyzer):
    """Generate causal analysis visualizations."""
    visualizer = CausalVisualizer(analyzer)
    visualizer.save_visualizations()

if __name__ == "__main__":
    # Example usage
    analyzer = FaultCausalityAnalyzer()
    visualize_causality(analyzer)
