"""Interactive controls for sensitivity visualizations."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import logging
from pathlib import Path
import json

from .advanced_sensitivity_viz import AdvancedVisualizer, VisualizationConfig
from .sensitivity_analysis import SensitivityAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class ControlConfig:
    """Configuration for visualization controls."""
    initial_metric: str = "power"
    slider_steps: int = 100
    update_interval: float = 0.5
    debounce_delay: float = 0.2
    color_presets: List[str] = None
    dimension_options: List[str] = None
    layout_templates: List[str] = None
    output_path: Optional[Path] = None
    
    def __post_init__(self):
        if self.color_presets is None:
            self.color_presets = [
                "viridis", "magma", "plasma", "inferno",
                "RdBu", "RdYlBu", "Spectral", "coolwarm"
            ]
        if self.dimension_options is None:
            self.dimension_options = ["2D", "3D", "Multi"]
        if self.layout_templates is None:
            self.layout_templates = [
                "plotly", "plotly_white", "plotly_dark",
                "seaborn", "simple_white", "ggplot2"
            ]

class VisualizationControls:
    """Interactive controls for sensitivity visualizations."""
    
    def __init__(
        self,
        visualizer: AdvancedVisualizer,
        config: ControlConfig
    ):
        self.visualizer = visualizer
        self.config = config
        self.app = dash.Dash(__name__)
        self.current_settings: Dict[str, Any] = {}
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup the control panel layout."""
        self.app.layout = html.Div([
            html.Div([  # Control Panel
                html.H3("Visualization Controls"),
                
                # Visualization Type
                html.Div([
                    html.Label("Visualization Type"),
                    dcc.Dropdown(
                        id="viz-type",
                        options=[
                            {"label": "Heatmap", "value": "heatmap"},
                            {"label": "Surface", "value": "surface"},
                            {"label": "Contour", "value": "contour"},
                            {"label": "Parallel", "value": "parallel"},
                            {"label": "Interaction", "value": "interaction"},
                            {"label": "Animation", "value": "animation"}
                        ],
                        value="heatmap"
                    )
                ], className="control-group"),
                
                # Parameters
                html.Div([
                    html.Label("Primary Parameter"),
                    dcc.Dropdown(
                        id="param1",
                        options=[
                            {"label": "Effect Size", "value": "effect_size"},
                            {"label": "Sample Size", "value": "sample_size"},
                            {"label": "Alpha", "value": "alpha"},
                            {"label": "Variance", "value": "variance"}
                        ],
                        value="effect_size"
                    )
                ], className="control-group"),
                
                html.Div([
                    html.Label("Secondary Parameter"),
                    dcc.Dropdown(
                        id="param2",
                        options=[
                            {"label": "Sample Size", "value": "sample_size"},
                            {"label": "Effect Size", "value": "effect_size"},
                            {"label": "Alpha", "value": "alpha"},
                            {"label": "Variance", "value": "variance"}
                        ],
                        value="sample_size"
                    )
                ], className="control-group"),
                
                # Appearance
                html.Div([
                    html.Label("Color Scheme"),
                    dcc.Dropdown(
                        id="colormap",
                        options=[
                            {"label": cmap, "value": cmap}
                            for cmap in self.config.color_presets
                        ],
                        value=self.visualizer.config.colormap
                    )
                ], className="control-group"),
                
                html.Div([
                    html.Label("Plot Template"),
                    dcc.Dropdown(
                        id="template",
                        options=[
                            {"label": template, "value": template}
                            for template in self.config.layout_templates
                        ],
                        value="plotly"
                    )
                ], className="control-group"),
                
                # Advanced Settings
                html.Div([
                    html.Label("Resolution"),
                    dcc.Slider(
                        id="resolution",
                        min=20,
                        max=200,
                        step=10,
                        value=self.visualizer.config.resolution,
                        marks={i: str(i) for i in range(20, 201, 20)}
                    )
                ], className="control-group"),
                
                html.Div([
                    html.Label("Animation Speed"),
                    dcc.Slider(
                        id="animation-speed",
                        min=100,
                        max=2000,
                        step=100,
                        value=self.visualizer.config.animation_duration,
                        marks={i: f"{i}ms" for i in range(100, 2001, 100)}
                    )
                ], className="control-group"),
                
                # Export Controls
                html.Div([
                    html.Button("Export Plot", id="export-button"),
                    dcc.Download(id="download-plot"),
                    html.Button("Save Settings", id="save-settings"),
                    dcc.Download(id="download-settings")
                ], className="control-group")
            ], className="control-panel"),
            
            # Visualization Display
            html.Div([
                dcc.Loading(
                    id="loading-output",
                    children=[
                        dcc.Graph(id="visualization-output")
                    ]
                )
            ], className="visualization-panel")
        ])
    
    def _setup_callbacks(self):
        """Setup the control callbacks."""
        @self.app.callback(
            Output("visualization-output", "figure"),
            [
                Input("viz-type", "value"),
                Input("param1", "value"),
                Input("param2", "value"),
                Input("colormap", "value"),
                Input("template", "value"),
                Input("resolution", "value"),
                Input("animation-speed", "value")
            ]
        )
        def update_visualization(
            viz_type: str,
            param1: str,
            param2: str,
            colormap: str,
            template: str,
            resolution: int,
            animation_speed: int
        ) -> go.Figure:
            """Update visualization based on control settings."""
            # Update visualizer config
            self.visualizer.config.colormap = colormap
            self.visualizer.config.resolution = resolution
            self.visualizer.config.animation_duration = animation_speed
            
            # Store current settings
            self.current_settings = {
                "viz_type": viz_type,
                "param1": param1,
                "param2": param2,
                "colormap": colormap,
                "template": template,
                "resolution": resolution,
                "animation_speed": animation_speed
            }
            
            # Create visualization
            if viz_type == "heatmap":
                fig = self.visualizer.create_sensitivity_heatmap(param1, param2)
            elif viz_type == "surface":
                fig = self.visualizer.create_sensitivity_surface(param1, param2)
            elif viz_type == "contour":
                fig = self.visualizer.create_contour_plot(param1, param2)
            elif viz_type == "parallel":
                fig = self.visualizer.create_parallel_coordinates(
                    self.visualizer.sensitivity.analysis_results
                )
            elif viz_type == "interaction":
                fig = self.visualizer.create_interaction_plot(
                    self.visualizer.sensitivity.analysis_results
                )
            else:  # animation
                fig = self.visualizer.create_sensitivity_animation(
                    param1,
                    param2,
                    "alpha"  # Third parameter for animation
                )
            
            # Update template
            fig.update_layout(template=template)
            
            return fig
        
        @self.app.callback(
            Output("download-plot", "data"),
            Input("export-button", "n_clicks"),
            prevent_initial_call=True
        )
        def export_plot(_) -> Dict[str, Any]:
            """Export current plot."""
            if not self.visualizer.config.output_path:
                return {}
            
            try:
                # Get current figure
                fig = self.app.get_component("visualization-output").figure
                
                # Save plot
                output_path = self.visualizer.config.output_path
                filename = (
                    f"sensitivity_{self.current_settings['viz_type']}_"
                    f"{self.current_settings['param1']}_"
                    f"{self.current_settings['param2']}.html"
                )
                fig.write_html(str(output_path / filename))
                
                return {
                    "content": "Plot exported successfully",
                    "filename": filename
                }
                
            except Exception as e:
                logger.error(f"Failed to export plot: {e}")
                return {}
        
        @self.app.callback(
            Output("download-settings", "data"),
            Input("save-settings", "n_clicks"),
            prevent_initial_call=True
        )
        def save_settings(_) -> Dict[str, Any]:
            """Save current visualization settings."""
            if not self.config.output_path:
                return {}
            
            try:
                return {
                    "content": json.dumps(self.current_settings, indent=2),
                    "filename": "visualization_settings.json"
                }
                
            except Exception as e:
                logger.error(f"Failed to save settings: {e}")
                return {}
    
    def run(self, host: str = "localhost", port: int = 8050):
        """Run the control panel."""
        self.app.run_server(host=host, port=port)

def create_visualization_controls(
    visualizer: AdvancedVisualizer,
    output_path: Optional[Path] = None
) -> VisualizationControls:
    """Create visualization controls."""
    config = ControlConfig(output_path=output_path)
    return VisualizationControls(visualizer, config)

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
    controls = create_visualization_controls(
        visualizer,
        output_path=Path("viz_controls")
    )
    
    # Run control panel
    controls.run()
