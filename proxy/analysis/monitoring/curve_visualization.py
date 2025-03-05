"""Visualization tools for easing curves and transitions."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime

from .easing_functions import EasingFunctions, EasingConfig

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for curve visualization."""
    width: int = 800
    height: int = 600
    samples: int = 200
    show_grid: bool = True
    show_control_points: bool = True
    show_velocity: bool = True
    dark_mode: bool = False
    interactive: bool = True
    output_path: Optional[Path] = None

class CurveVisualizer:
    """Visualize easing curves and transitions."""
    
    def __init__(
        self,
        easing: EasingFunctions,
        config: VisualizationConfig
    ):
        self.easing = easing
        self.config = config
    
    def visualize_easing(self, name: str) -> go.Figure:
        """Visualize single easing function."""
        t = np.linspace(0, 1, self.config.samples)
        easing_func = self.easing.get_easing_function(name)
        y = [easing_func(x) for x in t]
        
        # Calculate velocity (first derivative)
        velocity = np.gradient(y, t)
        
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[
                "Easing Curve",
                "Velocity Profile"
            ] if self.config.show_velocity else ["Easing Curve"],
            row_heights=[0.7, 0.3] if self.config.show_velocity else [1.0]
        )
        
        # Add easing curve
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y,
                mode="lines",
                name="Easing",
                line=dict(width=2, color="blue")
            ),
            row=1,
            col=1
        )
        
        # Add reference line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Linear",
                line=dict(
                    dash="dash",
                    color="gray",
                    width=1
                )
            ),
            row=1,
            col=1
        )
        
        # Add velocity profile
        if self.config.show_velocity:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=velocity,
                    mode="lines",
                    name="Velocity",
                    line=dict(color="red")
                ),
                row=2,
                col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"Easing Function: {name}",
            width=self.config.width,
            height=self.config.height,
            showlegend=True,
            template="plotly_dark" if self.config.dark_mode else "plotly_white"
        )
        
        # Add grid
        if self.config.show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="gray")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="gray")
        
        return fig
    
    def compare_easings(
        self,
        names: List[str]
    ) -> go.Figure:
        """Compare multiple easing functions."""
        t = np.linspace(0, 1, self.config.samples)
        
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Easing Curves",
                "Velocity Profiles",
                "Phase Portrait",
                "Acceleration"
            ]
        )
        
        for name in names:
            easing_func = self.easing.get_easing_function(name)
            y = [easing_func(x) for x in t]
            velocity = np.gradient(y, t)
            acceleration = np.gradient(velocity, t)
            
            # Add easing curve
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=y,
                    mode="lines",
                    name=name
                ),
                row=1,
                col=1
            )
            
            # Add velocity profile
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=velocity,
                    mode="lines",
                    name=f"{name} velocity"
                ),
                row=1,
                col=2
            )
            
            # Add phase portrait
            fig.add_trace(
                go.Scatter(
                    x=y,
                    y=velocity,
                    mode="lines",
                    name=f"{name} phase"
                ),
                row=2,
                col=1
            )
            
            # Add acceleration profile
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=acceleration,
                    mode="lines",
                    name=f"{name} acceleration"
                ),
                row=2,
                col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Easing Function Comparison",
            width=self.config.width * 1.5,
            height=self.config.height * 1.5,
            showlegend=True,
            template="plotly_dark" if self.config.dark_mode else "plotly_white"
        )
        
        return fig
    
    def visualize_custom_curve(
        self,
        name: str,
        show_construction: bool = True
    ) -> go.Figure:
        """Visualize custom easing curve."""
        if name not in self.easing.custom_curves:
            raise ValueError(f"Custom curve not found: {name}")
        
        curve = np.array(self.easing.custom_curves[name])
        
        fig = go.Figure()
        
        # Add curve
        fig.add_trace(
            go.Scatter(
                x=curve[:, 0],
                y=curve[:, 1],
                mode="lines",
                name="Curve",
                line=dict(color="blue", width=2)
            )
        )
        
        # Add control points
        if self.config.show_control_points:
            x = [p[0] for p in curve[::len(curve)//10]]
            y = [p[1] for p in curve[::len(curve)//10]]
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    name="Control Points",
                    marker=dict(
                        size=8,
                        color="red",
                        symbol="circle"
                    )
                )
            )
        
        # Add construction lines
        if show_construction:
            for i in range(len(x) - 1):
                fig.add_trace(
                    go.Scatter(
                        x=[x[i], x[i+1]],
                        y=[y[i], y[i+1]],
                        mode="lines",
                        name=f"Segment {i}",
                        line=dict(
                            dash="dot",
                            color="gray",
                            width=1
                        )
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=f"Custom Curve: {name}",
            width=self.config.width,
            height=self.config.height,
            showlegend=True,
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            xaxis_title="Time",
            yaxis_title="Value"
        )
        
        if self.config.show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="gray")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="gray")
        
        return fig
    
    def create_easing_gallery(self) -> go.Figure:
        """Create gallery of all available easing functions."""
        standard_easings = [
            "linear",
            "ease-in-quad",
            "ease-out-quad",
            "ease-in-out-quad",
            "ease-in-cubic",
            "ease-out-cubic",
            "ease-in-out-cubic",
            "ease-elastic",
            "ease-bounce",
            "ease-spring"
        ]
        
        rows = int(np.ceil(len(standard_easings) / 3))
        
        fig = make_subplots(
            rows=rows,
            cols=3,
            subplot_titles=standard_easings
        )
        
        t = np.linspace(0, 1, self.config.samples)
        
        for i, name in enumerate(standard_easings):
            row = i // 3 + 1
            col = i % 3 + 1
            
            easing_func = self.easing.get_easing_function(name)
            y = [easing_func(x) for x in t]
            
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=y,
                    mode="lines",
                    name=name,
                    showlegend=False
                ),
                row=row,
                col=col
            )
        
        # Update layout
        fig.update_layout(
            title="Easing Function Gallery",
            width=self.config.width * 1.2,
            height=self.config.height * 1.2,
            template="plotly_dark" if self.config.dark_mode else "plotly_white"
        )
        
        return fig
    
    def save_visualizations(
        self,
        name: Optional[str] = None
    ):
        """Save visualizations."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            if name:
                # Save single easing visualization
                fig = self.visualize_easing(name)
                fig.write_html(str(output_path / f"{name}_easing.html"))
                
                if name in self.easing.custom_curves:
                    fig = self.visualize_custom_curve(name)
                    fig.write_html(str(output_path / f"{name}_curve.html"))
            else:
                # Save gallery
                fig = self.create_easing_gallery()
                fig.write_html(str(output_path / "easing_gallery.html"))
            
            logger.info(f"Saved visualizations to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save visualizations: {e}")

def create_curve_visualizer(
    easing: EasingFunctions,
    output_path: Optional[Path] = None
) -> CurveVisualizer:
    """Create curve visualizer."""
    config = VisualizationConfig(output_path=output_path)
    return CurveVisualizer(easing, config)

if __name__ == "__main__":
    # Example usage
    from .easing_functions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    visualizer = create_curve_visualizer(
        easing,
        output_path=Path("curve_viz")
    )
    
    # Create custom curve
    control_points = [
        (0.0, 0.0),
        (0.2, 0.1),
        (0.4, 0.8),
        (0.8, 0.9),
        (1.0, 1.0)
    ]
    easing.create_custom_curve("custom1", control_points)
    
    # Generate and save visualizations
    visualizer.save_visualizations("ease-in-out-cubic")
    visualizer.save_visualizations("custom1")
    visualizer.save_visualizations()  # Save gallery
    
    # Compare multiple easings
    fig = visualizer.compare_easings([
        "ease-in-quad",
        "ease-out-quad",
        "ease-elastic",
        "ease-bounce"
    ])
    fig.write_html("easing_comparison.html")
