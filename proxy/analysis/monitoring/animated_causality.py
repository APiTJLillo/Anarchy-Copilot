"""Animated visualizations of causal relationships."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
import logging
from pathlib import Path
from collections import defaultdict
import math

from .causal_visualization import CausalVisualizer
from .fault_causality import FaultCausalityAnalyzer, CausalLink, RootCause

logger = logging.getLogger(__name__)

class AnimatedCausalVisualizer:
    """Create animated visualizations of causal relationships."""
    
    def __init__(
        self,
        analyzer: FaultCausalityAnalyzer,
        output_dir: Optional[Path] = None,
        frame_duration: int = 500  # milliseconds
    ):
        self.analyzer = analyzer
        self.output_dir = output_dir or Path("animated_causality")
        self.output_dir.mkdir(exist_ok=True)
        self.frame_duration = frame_duration
        self.base_visualizer = CausalVisualizer(analyzer)
    
    def create_causal_cascade_animation(self) -> go.Figure:
        """Create animation showing fault cascades developing."""
        if not self.analyzer.fault_history:
            return None
        
        # Sort events by timestamp
        events = sorted(
            self.analyzer.fault_history,
            key=lambda e: e.timestamp
        )
        
        # Create frames for each event
        frames = []
        node_positions = {}
        current_nodes = set()
        current_edges = set()
        
        for event in events:
            # Add new node
            current_nodes.add(event.fault_type)
            
            # Calculate node positions if needed
            if event.fault_type not in node_positions:
                angle = len(node_positions) * (2 * math.pi / len(events))
                node_positions[event.fault_type] = (
                    math.cos(angle),
                    math.sin(angle)
                )
            
            # Add causal edges
            for related in event.related_faults:
                if related in current_nodes:
                    current_edges.add((related, event.fault_type))
            
            # Create frame data
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            
            edge_x = []
            edge_y = []
            
            for node in current_nodes:
                x, y = node_positions[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                # Color based on impact
                impact = sum(
                    e.impact_metrics.values()
                    for e in events
                    if e.fault_type == node
                )
                node_color.append(impact)
            
            for src, tgt in current_edges:
                sx, sy = node_positions[src]
                tx, ty = node_positions[tgt]
                edge_x.extend([sx, tx, None])
                edge_y.extend([sy, ty, None])
            
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode="markers+text",
                        text=node_text,
                        marker=dict(
                            size=20,
                            color=node_color,
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Impact")
                        ),
                        textposition="bottom center",
                        name="Nodes"
                    ),
                    go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        mode="lines",
                        line=dict(width=1, color="gray"),
                        name="Edges"
                    )
                ],
                name=str(event.timestamp)
            )
            frames.append(frame)
        
        # Create figure with initial state
        fig = go.Figure(
            data=frames[0].data,
            frames=frames,
            layout=go.Layout(
                title="Fault Cascade Evolution",
                showlegend=False,
                hovermode="closest",
                xaxis=dict(range=[-1.5, 1.5], showgrid=False),
                yaxis=dict(range=[-1.5, 1.5], showgrid=False),
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": self.frame_duration}}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "type": "buttons"
                }],
                sliders=[{
                    "currentvalue": {
                        "prefix": "Time: "
                    },
                    "steps": [
                        {
                            "args": [[frame.name]],
                            "label": frame.name,
                            "method": "animate"
                        }
                        for frame in frames
                    ]
                }]
            )
        )
        
        return fig
    
    def create_impact_propagation_animation(self) -> go.Figure:
        """Create animation showing impact propagation over time."""
        if not self.analyzer.fault_history:
            return None
        
        # Create timeline
        events = sorted(
            self.analyzer.fault_history,
            key=lambda e: e.timestamp
        )
        start_time = events[0].timestamp
        end_time = events[-1].timestamp
        time_range = pd.date_range(start_time, end_time, freq="1min")
        
        # Create frames
        frames = []
        metrics = set()
        for event in events:
            metrics.update(event.impact_metrics.keys())
        
        for current_time in time_range:
            # Get active faults at current time
            active_faults = [
                e for e in events
                if (
                    e.timestamp <= current_time and
                    e.timestamp + timedelta(minutes=30) >= current_time
                )
            ]
            
            # Calculate impact propagation
            impact_data = defaultdict(list)
            for fault in active_faults:
                for metric, value in fault.impact_metrics.items():
                    impact_data[metric].append(value)
            
            # Create frame
            traces = []
            for metric in metrics:
                values = impact_data.get(metric, [0])
                traces.append(go.Bar(
                    x=[metric],
                    y=[sum(values)],
                    name=metric
                ))
            
            frame = go.Frame(
                data=traces,
                name=str(current_time)
            )
            frames.append(frame)
        
        # Create figure
        fig = go.Figure(
            data=frames[0].data,
            frames=frames,
            layout=go.Layout(
                title="Impact Propagation Over Time",
                yaxis_title="Cumulative Impact",
                barmode="stack",
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": self.frame_duration}}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "type": "buttons"
                }],
                sliders=[{
                    "currentvalue": {
                        "prefix": "Time: "
                    },
                    "steps": [
                        {
                            "args": [[frame.name]],
                            "label": frame.name,
                            "method": "animate"
                        }
                        for frame in frames
                    ]
                }]
            )
        )
        
        return fig
    
    def create_root_cause_evolution_animation(self) -> go.Figure:
        """Create animation showing root cause probability changes."""
        root_causes = self.analyzer.identify_root_causes()
        if not root_causes:
            return None
        
        # Create timeline
        events = sorted(
            self.analyzer.fault_history,
            key=lambda e: e.timestamp
        )
        time_range = pd.date_range(
            events[0].timestamp,
            events[-1].timestamp,
            freq="1min"
        )
        
        # Create frames
        frames = []
        for current_time in time_range:
            # Calculate probabilities up to current time
            current_events = [
                e for e in events
                if e.timestamp <= current_time
            ]
            
            if not current_events:
                continue
            
            # Calculate evolving probabilities
            root_probs = defaultdict(float)
            for rc in root_causes:
                rc_events = [
                    e for e in current_events
                    if e.fault_type == rc.fault_type
                ]
                if rc_events:
                    root_probs[rc.fault_type] = len(rc_events) / len(current_events)
            
            frame = go.Frame(
                data=[go.Bar(
                    x=list(root_probs.keys()),
                    y=list(root_probs.values()),
                    marker_color="red"
                )],
                name=str(current_time)
            )
            frames.append(frame)
        
        # Create figure
        fig = go.Figure(
            data=frames[0].data,
            frames=frames,
            layout=go.Layout(
                title="Root Cause Probability Evolution",
                yaxis_title="Probability",
                yaxis=dict(range=[0, 1]),
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": self.frame_duration}}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "type": "buttons"
                }],
                sliders=[{
                    "currentvalue": {
                        "prefix": "Time: "
                    },
                    "steps": [
                        {
                            "args": [[frame.name]],
                            "label": frame.name,
                            "method": "animate"
                        }
                        for frame in frames
                    ]
                }]
            )
        )
        
        return fig
    
    def save_animations(self):
        """Save all animations."""
        animations = {
            "causal_cascade": self.create_causal_cascade_animation(),
            "impact_propagation": self.create_impact_propagation_animation(),
            "root_cause_evolution": self.create_root_cause_evolution_animation()
        }
        
        # Save individual animations
        for name, fig in animations.items():
            if fig is not None:
                fig.write_html(
                    str(self.output_dir / f"{name}.html")
                )
        
        # Create index page
        self._create_index_page(
            [name for name, fig in animations.items() if fig is not None]
        )
    
    def _create_index_page(self, animations: List[str]):
        """Create HTML index page."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Causal Animation Dashboard</title>
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
            <h1>Causal Animation Dashboard</h1>
            <div id="animations">
        """
        
        for anim in animations:
            html += f"""
                <a class="viz-link" href="{anim}.html">
                    {anim.replace("_", " ").title()}
                </a>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / "index.html", "w") as f:
            f.write(html)

def animate_causality(analyzer: FaultCausalityAnalyzer):
    """Generate causal analysis animations."""
    animator = AnimatedCausalVisualizer(analyzer)
    animator.save_animations()

if __name__ == "__main__":
    # Example usage
    analyzer = FaultCausalityAnalyzer()
    animate_causality(analyzer)
