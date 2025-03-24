"""Interactive controls for prediction visualization."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime

from .prediction_visualization import PredictionVisualizer, VisualizationConfig
from .easing_prediction import EasingPredictor

logger = logging.getLogger(__name__)

@dataclass
class ControlConfig:
    """Configuration for interactive controls."""
    width: int = 300
    height: int = 800
    show_tooltips: bool = True
    advanced_mode: bool = False
    default_theme: str = "light"
    animation_duration: int = 500
    auto_update: bool = True
    output_path: Optional[Path] = None

class InteractiveControls:
    """Interactive controls for prediction visualization."""
    
    def __init__(
        self,
        visualizer: PredictionVisualizer,
        config: ControlConfig
    ):
        self.visualizer = visualizer
        self.config = config
        self.current_state: Dict[str, Any] = {}
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    def create_control_panel(self) -> go.Figure:
        """Create interactive control panel."""
        fig = go.Figure()
        
        # Add control groups
        self._add_prediction_controls(fig)
        self._add_visualization_controls(fig)
        self._add_analysis_controls(fig)
        self._add_export_controls(fig)
        
        # Update layout
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            margin=dict(l=20, r=20, t=40, b=20),
            template=self._get_theme(),
            showlegend=False
        )
        
        return fig
    
    def create_interactive_dashboard(
        self,
        name: str,
        t: np.ndarray
    ) -> Tuple[go.Figure, go.Figure]:
        """Create interactive dashboard with controls."""
        # Create main visualization
        main_fig = self.visualizer.create_prediction_dashboard(name, t)
        
        # Create control panel
        control_fig = self.create_control_panel()
        
        # Link interactions
        self._link_interactions(main_fig, control_fig)
        
        return main_fig, control_fig
    
    def _add_prediction_controls(self, fig: go.Figure):
        """Add prediction control elements."""
        buttons = [
            {
                "method": "restyle",
                "args": ["visible", [True]],
                "label": "Show All"
            },
            {
                "method": "restyle",
                "args": ["visible", "legendonly"],
                "label": "Hide All"
            }
        ]
        
        # Add model selection
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(size=0),
                showlegend=False,
                hoverinfo="none",
                visible=False,
                name="model_selector",
                customdata=[{
                    "type": "dropdown",
                    "options": ["RF", "GP", "NN", "Ensemble"],
                    "value": "Ensemble",
                    "callback": "update_model"
                }]
            )
        )
        
        # Add confidence interval control
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(size=0),
                showlegend=False,
                hoverinfo="none",
                visible=False,
                name="confidence_control",
                customdata=[{
                    "type": "slider",
                    "min": 0.8,
                    "max": 0.99,
                    "step": 0.01,
                    "value": 0.95,
                    "callback": "update_confidence"
                }]
            )
        )
    
    def _add_visualization_controls(self, fig: go.Figure):
        """Add visualization control elements."""
        # Add theme selector
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(size=0),
                showlegend=False,
                hoverinfo="none",
                visible=False,
                name="theme_selector",
                customdata=[{
                    "type": "radio",
                    "options": ["light", "dark", "custom"],
                    "value": self.config.default_theme,
                    "callback": "update_theme"
                }]
            )
        )
        
        # Add component visibility toggles
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(size=0),
                showlegend=False,
                hoverinfo="none",
                visible=False,
                name="component_toggles",
                customdata=[{
                    "type": "checklist",
                    "options": [
                        "Show Confidence Intervals",
                        "Show Components",
                        "Show Annotations"
                    ],
                    "value": [
                        "Show Confidence Intervals",
                        "Show Components"
                    ],
                    "callback": "update_visibility"
                }]
            )
        )
    
    def _add_analysis_controls(self, fig: go.Figure):
        """Add analysis control elements."""
        # Add analysis type selector
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(size=0),
                showlegend=False,
                hoverinfo="none",
                visible=False,
                name="analysis_selector",
                customdata=[{
                    "type": "dropdown",
                    "options": [
                        "Point Predictions",
                        "Uncertainty Analysis",
                        "Trend Analysis",
                        "Anomaly Detection"
                    ],
                    "value": "Point Predictions",
                    "callback": "update_analysis"
                }]
            )
        )
        
        # Add advanced settings if enabled
        if self.config.advanced_mode:
            self._add_advanced_controls(fig)
    
    def _add_export_controls(self, fig: go.Figure):
        """Add export control elements."""
        buttons = [
            {
                "method": "relayout",
                "args": ["showlegend", True],
                "label": "Export Data"
            },
            {
                "method": "relayout",
                "args": ["showlegend", True],
                "label": "Save Image"
            }
        ]
        
        fig.update_layout(
            updatemenus=[{
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "y": 1.1,
                "xanchor": "left",
                "yanchor": "top"
            }]
        )
    
    def _add_advanced_controls(self, fig: go.Figure):
        """Add advanced analysis controls."""
        # Add parameter tuning
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(size=0),
                showlegend=False,
                hoverinfo="none",
                visible=False,
                name="parameter_tuning",
                customdata=[{
                    "type": "group",
                    "elements": [
                        {
                            "type": "slider",
                            "label": "Window Size",
                            "min": 10,
                            "max": 100,
                            "value": 50,
                            "callback": "update_window"
                        },
                        {
                            "type": "slider",
                            "label": "Threshold",
                            "min": 0.1,
                            "max": 5.0,
                            "value": 2.0,
                            "callback": "update_threshold"
                        }
                    ]
                }]
            )
        )
    
    def _link_interactions(
        self,
        main_fig: go.Figure,
        control_fig: go.Figure
    ):
        """Link interactive elements between figures."""
        # Add crossfiltering
        for trace in main_fig.data:
            trace.update(
                selected=dict(marker=dict(color="red")),
                unselected=dict(marker=dict(color="gray"))
            )
        
        # Add click callbacks
        main_fig.update_layout(
            clickmode="event+select"
        )
        
        # Add hover interactions
        if self.config.show_tooltips:
            main_fig.update_traces(
                hoverinfo="all",
                hovertemplate=(
                    "Time: %{x}<br>" +
                    "Value: %{y:.2f}<br>" +
                    "<extra></extra>"
                )
            )
    
    def register_callback(
        self,
        event: str,
        callback: Callable
    ):
        """Register callback for interactive events."""
        self.callbacks[event].append(callback)
    
    def _get_theme(self) -> str:
        """Get current theme template."""
        themes = {
            "light": "plotly_white",
            "dark": "plotly_dark",
            "custom": "custom_theme"
        }
        return themes.get(
            self.config.default_theme,
            "plotly_white"
        )
    
    def save_state(self):
        """Save current control state."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            state_file = output_path / "control_state.json"
            with open(state_file, "w") as f:
                json.dump(self.current_state, f, indent=2)
            
            logger.info(f"Saved control state to {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load_state(self):
        """Load saved control state."""
        if not self.config.output_path:
            return
        
        try:
            state_file = self.config.output_path / "control_state.json"
            if not state_file.exists():
                return
            
            with open(state_file) as f:
                self.current_state = json.load(f)
            
            logger.info(f"Loaded control state from {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

def create_interactive_controls(
    visualizer: PredictionVisualizer,
    output_path: Optional[Path] = None
) -> InteractiveControls:
    """Create interactive controls."""
    config = ControlConfig(output_path=output_path)
    return InteractiveControls(visualizer, config)

if __name__ == "__main__":
    # Example usage
    from .prediction_visualization import create_prediction_visualizer
    from .easing_prediction import create_easing_predictor
    from .easing_statistics import create_easing_statistics
    from .easing_metrics import create_easing_metrics
    from .easing_functions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    metrics = create_easing_metrics(easing)
    stats = create_easing_statistics(metrics)
    predictor = create_easing_predictor(stats)
    visualizer = create_prediction_visualizer(predictor)
    controls = create_interactive_controls(
        visualizer,
        output_path=Path("prediction_controls")
    )
    
    # Create interactive dashboard
    t = np.linspace(0, 1, 100)
    main_fig, control_fig = controls.create_interactive_dashboard(
        "ease-in-out-cubic",
        t
    )
    
    # Save figures
    main_fig.write_html("prediction_dashboard.html")
    control_fig.write_html("control_panel.html")
    
    # Save state
    controls.save_state()
