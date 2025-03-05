"""Visualization tools for easing functions."""

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

from .easing_transitions import EasingFunctions, EasingConfig

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for easing visualizations."""
    width: int = 1200
    height: int = 800
    dark_mode: bool = False
    show_derivatives: bool = True
    show_grid: bool = True
    show_legend: bool = True
    animation_fps: int = 30
    output_path: Optional[Path] = None

class EasingVisualizer:
    """Visualize easing functions."""
    
    def __init__(
        self,
        easing: EasingFunctions,
        config: VisualizationConfig
    ):
        self.easing = easing
        self.config = config
    
    def create_curve_comparison(
        self,
        easing_names: Optional[List[str]] = None
    ) -> go.Figure:
        """Create easing curve comparison visualization."""
        fig = go.Figure()
        
        # Get easing functions to compare
        if easing_names is None:
            easing_names = [
                name for name in self.easing.get_easing_function("linear").__code__.co_names
                if name.startswith("ease")
            ]
        
        # Add curves
        for name in easing_names:
            t, y = self.easing.generate_easing_curve(name)
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=y,
                    name=name,
                    mode="lines",
                    line=dict(width=2)
                )
            )
        
        # Update layout
        fig.update_layout(
            title="Easing Function Comparison",
            xaxis_title="Progress",
            yaxis_title="Value",
            width=self.config.width,
            height=self.config.height,
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            showlegend=self.config.show_legend,
            xaxis=dict(showgrid=self.config.show_grid),
            yaxis=dict(showgrid=self.config.show_grid)
        )
        
        return fig
    
    def create_derivative_analysis(
        self,
        easing_name: str
    ) -> go.Figure:
        """Create derivative analysis visualization."""
        t, y = self.easing.generate_easing_curve(easing_name)
        
        # Calculate derivatives
        dy = np.gradient(y, t)  # First derivative
        d2y = np.gradient(dy, t)  # Second derivative
        
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[
                "Value",
                "First Derivative (Velocity)",
                "Second Derivative (Acceleration)"
            ]
        )
        
        # Add curves
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y,
                name="Value",
                line=dict(width=2)
            ),
            row=1,
            col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=t,
                y=dy,
                name="Velocity",
                line=dict(width=2)
            ),
            row=2,
            col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=t,
                y=d2y,
                name="Acceleration",
                line=dict(width=2)
            ),
            row=3,
            col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"{easing_name} Analysis",
            width=self.config.width,
            height=self.config.height,
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            showlegend=self.config.show_legend
        )
        
        fig.update_xaxes(showgrid=self.config.show_grid)
        fig.update_yaxes(showgrid=self.config.show_grid)
        
        return fig
    
    def create_animation_preview(
        self,
        easing_name: str
    ) -> go.Figure:
        """Create animated preview of easing function."""
        t, y = self.easing.generate_easing_curve(easing_name)
        
        # Create frames
        frames = []
        for i in range(len(t)):
            frame = go.Frame(
                data=[
                    # Full curve (faded)
                    go.Scatter(
                        x=t,
                        y=y,
                        mode="lines",
                        line=dict(
                            color="rgba(150,150,150,0.3)",
                            width=1
                        ),
                        showlegend=False
                    ),
                    # Current progress
                    go.Scatter(
                        x=t[:i+1],
                        y=y[:i+1],
                        mode="lines",
                        line=dict(width=2),
                        name="Progress"
                    ),
                    # Current point
                    go.Scatter(
                        x=[t[i]],
                        y=[y[i]],
                        mode="markers",
                        marker=dict(size=10),
                        name="Current"
                    )
                ],
                name=f"frame{i}"
            )
            frames.append(frame)
        
        # Create figure
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=[t[0]],
                    y=[y[0]],
                    mode="markers",
                    marker=dict(size=10)
                )
            ],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": 1000 / self.config.animation_fps,
                                    "redraw": True
                                },
                                "fromcurrent": True
                            }
                        ],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0},
                                "mode": "immediate"
                            }
                        ],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top"
            }]
        )
        
        # Add slider
        fig.update_layout(
            sliders=[{
                "currentvalue": {"prefix": "Frame: "},
                "steps": [
                    {
                        "args": [
                            [f"frame{i}"],
                            {
                                "frame": {"duration": 0},
                                "mode": "immediate"
                            }
                        ],
                        "label": str(i),
                        "method": "animate"
                    }
                    for i in range(len(frames))
                ]
            }]
        )
        
        # Update layout
        fig.update_layout(
            title=f"{easing_name} Animation",
            xaxis_title="Progress",
            yaxis_title="Value",
            width=self.config.width,
            height=self.config.height,
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            showlegend=self.config.show_legend,
            xaxis=dict(showgrid=self.config.show_grid),
            yaxis=dict(showgrid=self.config.show_grid)
        )
        
        return fig
    
    def create_easing_heatmap(
        self,
        property_name: str = "execution_time"
    ) -> go.Figure:
        """Create heatmap of easing function characteristics."""
        easing_names = [
            name for name in self.easing.get_easing_function("linear").__code__.co_names
            if name.startswith("ease")
        ]
        
        # Calculate properties
        properties = {}
        for name in easing_names:
            t, y = self.easing.generate_easing_curve(name)
            
            if property_name == "execution_time":
                # Measure execution time
                start_time = datetime.now()
                for _ in range(1000):
                    self.easing.get_easing_function(name)(0.5)
                properties[name] = (datetime.now() - start_time).total_seconds()
            
            elif property_name == "smoothness":
                # Calculate curve smoothness
                dy = np.gradient(y, t)
                properties[name] = -np.mean(np.abs(np.gradient(dy, t)))
            
            elif property_name == "overshoots":
                # Count overshoots
                properties[name] = np.sum(
                    (y < 0) | (y > 1)
                )
            
            else:
                properties[name] = 0
        
        # Create categorized data
        data = []
        categories = ["in", "out", "in-out"]
        types = ["quad", "cubic", "quart", "quint", "sine", "expo", "circ", "back", "elastic", "bounce"]
        
        z = np.zeros((len(types), len(categories)))
        for i, type_ in enumerate(types):
            for j, category in enumerate(categories):
                name = f"ease-{category}-{type_}"
                if name in properties:
                    z[i, j] = properties[name]
        
        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=categories,
                y=types,
                colorscale="Viridis",
                colorbar=dict(title=property_name.title())
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"Easing Function {property_name.title()} Comparison",
            width=self.config.width,
            height=self.config.height,
            template="plotly_dark" if self.config.dark_mode else "plotly_white"
        )
        
        return fig
    
    def save_visualizations(
        self,
        easing_name: Optional[str] = None
    ):
        """Save visualization results."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save curve comparison
            comparison = self.create_curve_comparison()
            comparison.write_html(
                str(output_path / "easing_comparison.html")
            )
            
            if easing_name:
                # Save derivative analysis
                analysis = self.create_derivative_analysis(easing_name)
                analysis.write_html(
                    str(output_path / f"{easing_name}_analysis.html")
                )
                
                # Save animation preview
                preview = self.create_animation_preview(easing_name)
                preview.write_html(
                    str(output_path / f"{easing_name}_preview.html")
                )
            
            # Save heatmaps
            for property_name in ["execution_time", "smoothness", "overshoots"]:
                heatmap = self.create_easing_heatmap(property_name)
                heatmap.write_html(
                    str(output_path / f"easing_{property_name}_heatmap.html")
                )
            
            logger.info(f"Saved visualizations to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save visualizations: {e}")

def create_easing_visualizer(
    easing: EasingFunctions,
    output_path: Optional[Path] = None
) -> EasingVisualizer:
    """Create easing visualizer."""
    config = VisualizationConfig(output_path=output_path)
    return EasingVisualizer(easing, config)

if __name__ == "__main__":
    # Example usage
    from .easing_transitions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    visualizer = create_easing_visualizer(
        easing,
        output_path=Path("easing_viz")
    )
    
    # Generate visualizations
    visualizer.save_visualizations("ease-in-out-elastic")
