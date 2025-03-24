"""Advanced visualizations for sensitivity analysis."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
import logging
from pathlib import Path
import json
import pandas as pd

from .sensitivity_analysis import SensitivityAnalyzer
from .interactive_sensitivity import InteractiveSensitivity

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for advanced visualizations."""
    colormap: str = "viridis"
    animation_duration: int = 500
    resolution: int = 100
    contour_levels: int = 20
    surface_quality: int = 50
    smoothing: bool = True
    interactive_legend: bool = True
    output_path: Optional[Path] = None
    height: int = 800
    width: int = 1200

class AdvancedVisualizer:
    """Advanced visualization tools for sensitivity analysis."""
    
    def __init__(
        self,
        sensitivity: SensitivityAnalyzer,
        config: VisualizationConfig
    ):
        self.sensitivity = sensitivity
        self.config = config
        self.cached_data: Dict[str, Any] = {}
    
    def create_sensitivity_heatmap(
        self,
        param1: str,
        param2: str,
        metric: str = "power"
    ) -> go.Figure:
        """Create 2D sensitivity heatmap."""
        # Generate parameter combinations
        p1_range = np.linspace(
            min(self.sensitivity.config.parameter_ranges[param1]),
            max(self.sensitivity.config.parameter_ranges[param1]),
            self.config.resolution
        )
        p2_range = np.linspace(
            min(self.sensitivity.config.parameter_ranges[param2]),
            max(self.sensitivity.config.parameter_ranges[param2]),
            self.config.resolution
        )
        
        # Calculate sensitivity values
        values = np.zeros((len(p1_range), len(p2_range)))
        for i, v1 in enumerate(p1_range):
            for j, v2 in enumerate(p2_range):
                params = {
                    "effect_size": 0.5,
                    "sample_size": 100,
                    "alpha": 0.05,
                    "variance": 1.0
                }
                params[param1] = v1
                params[param2] = v2
                
                result = self.sensitivity._analyze_single_combination(
                    params,
                    "t_test"
                )
                if result:
                    values[i, j] = result[metric]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=values,
            x=p2_range,
            y=p1_range,
            colorscale=self.config.colormap,
            colorbar=dict(title=metric.capitalize()),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"Sensitivity Heatmap: {param1} vs {param2}",
            xaxis_title=param2,
            yaxis_title=param1,
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def create_sensitivity_surface(
        self,
        param1: str,
        param2: str,
        metric: str = "power"
    ) -> go.Figure:
        """Create 3D sensitivity surface."""
        # Reuse data from heatmap
        p1_range = np.linspace(
            min(self.sensitivity.config.parameter_ranges[param1]),
            max(self.sensitivity.config.parameter_ranges[param1]),
            self.config.surface_quality
        )
        p2_range = np.linspace(
            min(self.sensitivity.config.parameter_ranges[param2]),
            max(self.sensitivity.config.parameter_ranges[param2]),
            self.config.surface_quality
        )
        
        values = np.zeros((len(p1_range), len(p2_range)))
        for i, v1 in enumerate(p1_range):
            for j, v2 in enumerate(p2_range):
                params = {
                    "effect_size": 0.5,
                    "sample_size": 100,
                    "alpha": 0.05,
                    "variance": 1.0
                }
                params[param1] = v1
                params[param2] = v2
                
                result = self.sensitivity._analyze_single_combination(
                    params,
                    "t_test"
                )
                if result:
                    values[i, j] = result[metric]
        
        # Create surface plot
        fig = go.Figure(data=[go.Surface(
            z=values,
            x=p2_range,
            y=p1_range,
            colorscale=self.config.colormap,
            colorbar=dict(title=metric.capitalize())
        )])
        
        fig.update_layout(
            title=f"Sensitivity Surface: {param1} vs {param2}",
            scene=dict(
                xaxis_title=param2,
                yaxis_title=param1,
                zaxis_title=metric
            ),
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def create_parallel_coordinates(
        self,
        results: List[Dict[str, Any]]
    ) -> go.Figure:
        """Create parallel coordinates plot."""
        # Convert results to DataFrame
        df = pd.DataFrame([
            {**r["parameters"], "power": r["power"]}
            for r in results
        ])
        
        # Create parallel coordinates
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=df["power"],
                colorscale=self.config.colormap
            ),
            dimensions=[
                dict(range=[df[col].min(), df[col].max()],
                     label=col,
                     values=df[col])
                for col in df.columns
            ]
        ))
        
        fig.update_layout(
            title="Parameter Interactions",
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def create_contour_plot(
        self,
        param1: str,
        param2: str,
        metric: str = "power"
    ) -> go.Figure:
        """Create contour plot with isolines."""
        # Reuse data generation
        p1_range = np.linspace(
            min(self.sensitivity.config.parameter_ranges[param1]),
            max(self.sensitivity.config.parameter_ranges[param1]),
            self.config.resolution
        )
        p2_range = np.linspace(
            min(self.sensitivity.config.parameter_ranges[param2]),
            max(self.sensitivity.config.parameter_ranges[param2]),
            self.config.resolution
        )
        
        values = np.zeros((len(p1_range), len(p2_range)))
        for i, v1 in enumerate(p1_range):
            for j, v2 in enumerate(p2_range):
                params = {
                    "effect_size": 0.5,
                    "sample_size": 100,
                    "alpha": 0.05,
                    "variance": 1.0
                }
                params[param1] = v1
                params[param2] = v2
                
                result = self.sensitivity._analyze_single_combination(
                    params,
                    "t_test"
                )
                if result:
                    values[i, j] = result[metric]
        
        # Create contour plot
        fig = go.Figure(data=go.Contour(
            z=values,
            x=p2_range,
            y=p1_range,
            colorscale=self.config.colormap,
            contours=dict(
                start=np.min(values),
                end=np.max(values),
                size=(np.max(values) - np.min(values)) / self.config.contour_levels
            ),
            colorbar=dict(title=metric.capitalize()),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"Sensitivity Contours: {param1} vs {param2}",
            xaxis_title=param2,
            yaxis_title=param1,
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def create_interaction_plot(
        self,
        results: List[Dict[str, Any]]
    ) -> go.Figure:
        """Create interaction effect plot."""
        df = pd.DataFrame([
            {**r["parameters"], "power": r["power"]}
            for r in results
        ])
        
        # Create subplot for each parameter pair
        params = list(df.columns[:-1])  # Exclude power
        n_params = len(params)
        
        fig = make_subplots(
            rows=n_params-1,
            cols=n_params-1,
            subplot_titles=[
                f"{p1} × {p2}"
                for p1 in params[:-1]
                for p2 in params[1:]
                if p1 != p2
            ]
        )
        
        row = 1
        col = 1
        for p1 in params[:-1]:
            for p2 in params[1:]:
                if p1 != p2:
                    # Add scatter plot for interaction
                    fig.add_trace(
                        go.Scatter(
                            x=df[p1],
                            y=df[p2],
                            mode="markers",
                            marker=dict(
                                color=df["power"],
                                colorscale=self.config.colormap,
                                showscale=True if row == 1 and col == 1 else False
                            ),
                            name=f"{p1} × {p2}"
                        ),
                        row=row,
                        col=col
                    )
                    
                    col += 1
                    if col > n_params - 1:
                        col = 1
                        row += 1
        
        fig.update_layout(
            title="Parameter Interactions Matrix",
            showlegend=False,
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def create_sensitivity_animation(
        self,
        param1: str,
        param2: str,
        param3: str,
        metric: str = "power"
    ) -> go.Figure:
        """Create animated sensitivity plot."""
        frames = []
        p3_range = self.sensitivity.config.parameter_ranges[param3]
        
        for value in p3_range:
            # Create frame for each value of param3
            params = {
                "effect_size": 0.5,
                "sample_size": 100,
                "alpha": 0.05,
                "variance": 1.0
            }
            params[param3] = value
            
            # Generate 2D heatmap for params 1 and 2
            p1_range = np.linspace(
                min(self.sensitivity.config.parameter_ranges[param1]),
                max(self.sensitivity.config.parameter_ranges[param1]),
                self.config.resolution
            )
            p2_range = np.linspace(
                min(self.sensitivity.config.parameter_ranges[param2]),
                max(self.sensitivity.config.parameter_ranges[param2]),
                self.config.resolution
            )
            
            values = np.zeros((len(p1_range), len(p2_range)))
            for i, v1 in enumerate(p1_range):
                for j, v2 in enumerate(p2_range):
                    params[param1] = v1
                    params[param2] = v2
                    
                    result = self.sensitivity._analyze_single_combination(
                        params,
                        "t_test"
                    )
                    if result:
                        values[i, j] = result[metric]
            
            frame = go.Frame(
                data=[go.Heatmap(
                    z=values,
                    x=p2_range,
                    y=p1_range,
                    colorscale=self.config.colormap
                )],
                name=str(value)
            )
            frames.append(frame)
        
        # Create base figure
        fig = go.Figure(
            data=[frames[0].data[0]],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title=f"Sensitivity Animation: {param1} vs {param2} by {param3}",
            xaxis_title=param2,
            yaxis_title=param1,
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": self.config.animation_duration}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ]
            }],
            sliders=[{
                "currentvalue": {"prefix": f"{param3}: "},
                "steps": [
                    {
                        "args": [[f.name], {"frame": {"duration": 0}}],
                        "label": f.name,
                        "method": "animate"
                    }
                    for f in frames
                ]
            }],
            width=self.config.width,
            height=self.config.height
        )
        
        return fig

def create_advanced_visualizer(
    sensitivity: SensitivityAnalyzer,
    output_path: Optional[Path] = None
) -> AdvancedVisualizer:
    """Create advanced visualizer."""
    config = VisualizationConfig(output_path=output_path)
    return AdvancedVisualizer(sensitivity, config)

if __name__ == "__main__":
    # Example usage
    from .sensitivity_analysis import create_sensitivity_analyzer
    from .power_analysis import create_analyzer
    
    power_analyzer = create_analyzer()
    sensitivity = create_sensitivity_analyzer(power_analyzer)
    visualizer = create_advanced_visualizer(
        sensitivity,
        output_path=Path("sensitivity_viz")
    )
    
    # Create visualizations
    heatmap = visualizer.create_sensitivity_heatmap(
        "effect_size",
        "sample_size"
    )
    surface = visualizer.create_sensitivity_surface(
        "effect_size",
        "sample_size"
    )
    
    # Save visualizations
    heatmap.write_html("sensitivity_heatmap.html")
    surface.write_html("sensitivity_surface.html")
