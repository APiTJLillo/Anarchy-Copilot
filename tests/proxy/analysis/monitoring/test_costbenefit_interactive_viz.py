"""Interactive controls for explanation visualizations."""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

import pytest
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .test_costbenefit_explanation_viz import (
    ExplanationVisualizer,
    ExplanationVisualizationConfig
)

@dataclass
class InteractiveControlConfig:
    """Configuration for interactive visualization controls."""
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_selection: bool = True
    enable_hover: bool = True
    enable_reset: bool = True
    enable_compare: bool = True
    enable_filters: bool = True
    add_rangesliders: bool = True
    enable_animation: bool = True
    animation_duration: int = 500
    animation_easing: str = "cubic-in-out"
    hover_template: str = (
        "<b>%{x}</b><br>" +
        "Value: %{y:.3f}<br>" +
        "<extra></extra>"
    )
    margin: Dict[str, int] = field(default_factory=lambda: {
        "l": 50, "r": 50, "t": 50, "b": 50, "pad": 4
    })

class InteractiveVisualizer:
    """Add interactive controls to explanation visualizations."""

    def __init__(
        self,
        base_visualizer: ExplanationVisualizer,
        config: InteractiveControlConfig
    ):
        self.base_visualizer = base_visualizer
        self.config = config
        self.click_handlers: Dict[str, List[Callable]] = {}
        self.hover_handlers: Dict[str, List[Callable]] = {}
        self.selection_handlers: Dict[str, List[Callable]] = {}

    def enhance_feature_importance_plot(
        self,
        fig: go.Figure,
        importance_data: Dict[str, Any]
    ) -> go.Figure:
        """Add interactive controls to feature importance plot."""
        # Add filter buttons
        if self.config.enable_filters:
            filter_buttons = [
                dict(
                    label="All Features",
                    method="update",
                    args=[{"visible": [True] * len(fig.data)}]
                ),
                dict(
                    label="Top 5 Features",
                    method="update",
                    args=[{"visible": [i < 5 for i in range(len(fig.data))]}]
                ),
                dict(
                    label="Significant Only",
                    method="update",
                    args=[{"visible": [
                        "p=" in trace.text[0] if hasattr(trace, "text") else True
                        for trace in fig.data
                    ]}]
                )
            ]
            
            fig.update_layout(
                updatemenus=[dict(
                    type="dropdown",
                    buttons=filter_buttons,
                    x=0,
                    y=1.1,
                    xanchor="left",
                    yanchor="top"
                )]
            )
        
        # Add range sliders
        if self.config.add_rangesliders:
            fig.update_layout(
                xaxis=dict(rangeslider=dict(visible=True)),
                xaxis2=dict(rangeslider=dict(visible=True))
            )
        
        # Add interactive features
        if self.config.enable_zoom:
            fig.update_layout(
                dragmode="zoom",
                modebar=dict(
                    add=["zoomIn2d", "zoomOut2d", "resetScale2d"]
                )
            )
        
        # Add hover template
        if self.config.enable_hover:
            fig.update_traces(
                hovertemplate=self.config.hover_template
            )
        
        # Add animation settings
        if self.config.enable_animation:
            fig.update_layout(
                transition={
                    "duration": self.config.animation_duration,
                    "easing": self.config.animation_easing
                },
                frame={
                    "duration": self.config.animation_duration
                }
            )
        
        return fig

    def enhance_shap_summary_plot(
        self,
        fig: go.Figure,
        explanation: Dict[str, Any]
    ) -> go.Figure:
        """Add interactive controls to SHAP summary plot."""
        # Add comparison controls
        if self.config.enable_compare:
            compare_buttons = [
                dict(
                    label="Absolute Values",
                    method="update",
                    args=[{"y": [[abs(v) for v in trace.y] for trace in fig.data]}]
                ),
                dict(
                    label="Raw Values",
                    method="update",
                    args=[{"y": [trace.y for trace in fig.data]}]
                )
            ]
            
            fig.update_layout(
                updatemenus=[dict(
                    type="buttons",
                    buttons=compare_buttons,
                    x=0,
                    y=1.1,
                    xanchor="left",
                    yanchor="top"
                )]
            )
        
        # Add selection tools
        if self.config.enable_selection:
            fig.update_layout(
                dragmode="select",
                clickmode="event+select"
            )
            
            # Add selection handler
            fig.update_traces(
                selectedpoints=None,
                selected=dict(
                    marker=dict(color="red", size=8)
                ),
                unselected=dict(
                    marker=dict(color="gray", opacity=0.5)
                )
            )
        
        return fig

    def enhance_lime_explanation_plot(
        self,
        fig: go.Figure,
        explanation: Dict[str, Any]
    ) -> go.Figure:
        """Add interactive controls to LIME explanation plot."""
        # Add confidence interval toggle
        if "lime" in explanation:
            ci_buttons = [
                dict(
                    label="Show Confidence Intervals",
                    method="update",
                    args=[{
                        "error_y": [{
                            "type": "data",
                            "array": [0.1] * len(fig.data[0].y),
                            "visible": True
                        }]
                    }]
                ),
                dict(
                    label="Hide Confidence Intervals",
                    method="update",
                    args=[{"error_y": [{"visible": False}]}]
                )
            ]
            
            fig.update_layout(
                updatemenus=[dict(
                    type="buttons",
                    buttons=ci_buttons,
                    x=0.1,
                    y=1.1,
                    xanchor="left",
                    yanchor="top"
                )]
            )
        
        # Add feature highlighting
        if self.config.enable_hover:
            fig.update_traces(
                hoverinfo="x+y+text",
                hoverlabel=dict(bgcolor="white"),
                hovertemplate=(
                    "<b>%{x}</b><br>" +
                    "Contribution: %{y:.3f}<br>" +
                    "<extra></extra>"
                )
            )
        
        return fig

    def register_event_handler(
        self,
        event_type: str,
        plot_type: str,
        handler: Callable
    ) -> None:
        """Register event handler for interactive elements."""
        if event_type == "click":
            handlers = self.click_handlers
        elif event_type == "hover":
            handlers = self.hover_handlers
        elif event_type == "selection":
            handlers = self.selection_handlers
        else:
            raise ValueError(f"Unsupported event type: {event_type}")
        
        if plot_type not in handlers:
            handlers[plot_type] = []
        handlers[plot_type].append(handler)

@pytest.fixture
def interactive_visualizer(explanation_visualizer):
    """Create interactive visualizer for testing."""
    config = InteractiveControlConfig()
    return InteractiveVisualizer(explanation_visualizer, config)

@pytest.mark.asyncio
async def test_interactive_feature_importance(
    interactive_visualizer,
    model_explainer,
    tmp_path
):
    """Test interactive feature importance visualization."""
    # Get base plot
    importance = model_explainer.analyze_feature_importance()
    base_fig = interactive_visualizer.base_visualizer.create_feature_importance_plot(
        importance
    )
    
    # Add interactive controls
    fig = interactive_visualizer.enhance_feature_importance_plot(
        base_fig,
        importance
    )
    
    # Verify interactive elements
    assert "updatemenus" in fig.layout
    assert fig.layout.dragmode == "zoom"
    assert fig.layout.modebar.add == ["zoomIn2d", "zoomOut2d", "resetScale2d"]

@pytest.mark.asyncio
async def test_event_handlers(interactive_visualizer):
    """Test event handler registration and triggering."""
    events = []
    
    def test_handler(event_data):
        events.append(event_data)
    
    # Register handlers
    interactive_visualizer.register_event_handler(
        "click",
        "feature_importance",
        test_handler
    )
    
    # Verify registration
    assert len(interactive_visualizer.click_handlers["feature_importance"]) == 1
    
    # Test invalid event type
    with pytest.raises(ValueError):
        interactive_visualizer.register_event_handler(
            "invalid",
            "feature_importance",
            test_handler
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
