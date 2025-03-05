"""Cost-benefit analysis for interventions."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy import stats, optimize
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .intervention_analysis import (
    InterventionAnalyzer, InterventionConfig, Intervention,
    InterventionEffect, InterventionResult
)

@dataclass
class PlotControls:
    """Configuration for plot controls."""
    enable_range_selector: bool = True
    enable_compare_mode: bool = True
    enable_highlight: bool = True
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_reset: bool = True
    range_selector_options: List[Dict[str, Any]] = field(default_factory=lambda: [
        {'count': 1, 'label': '1h', 'step': 'hour', 'stepmode': 'backward'},
        {'count': 24, 'label': '1d', 'step': 'hour', 'stepmode': 'backward'},
        {'count': 7, 'label': '1w', 'step': 'day', 'stepmode': 'backward'},
        {'step': 'all'}
    ])
    compare_metrics: List[str] = field(default_factory=lambda: [
        'net_present_value',
        'benefit_cost_ratio',
        'roi'
    ])

@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    theme: str = "plotly"
    interactive: bool = True
    animate_transitions: bool = True
    show_tooltips: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["html", "png", "svg"])
    width: int = 1200
    height: int = 800
    margin: Dict[str, int] = field(default_factory=lambda: {"l": 50, "r": 50, "t": 50, "b": 50})
    color_scheme: Dict[str, str] = field(default_factory=lambda: {
        "cost": "red",
        "benefit": "green",
        "neutral": "blue"
    })
    animation_duration: int = 500
    animation_easing: str = "cubic-in-out"
    tooltip_template: str = "<b>%{x}</b><br>Value: %{y:.2f}<br><extra></extra>"
    controls: PlotControls = field(default_factory=PlotControls)

[Previous CostConfig, CostComponent, BenefitComponent, and CostBenefitResult dataclasses remain the same]

class CostBenefitAnalyzer:
    """Analyze costs and benefits of interventions."""

    def __init__(
        self,
        intervention_analyzer: InterventionAnalyzer,
        config: Optional[CostConfig] = None
    ):
        """Initialize the analyzer."""
        self.intervention_analyzer = intervention_analyzer
        self.config = config or CostConfig()
        self.results: Dict[str, CostBenefitResult] = {}
        self.cost_models: Dict[str, LinearRegression] = {}
        self.benefit_models: Dict[str, LinearRegression] = {}
        self.last_update = datetime.min
        self.analyzer_task: Optional[asyncio.Task] = None
        
        # Event handlers
        self.click_handlers: Dict[str, List[Callable]] = {}
        self.hover_handlers: Dict[str, List[Callable]] = {}
        self.selection_handlers: Dict[str, List[Callable]] = {}

    def register_event_handler(
        self,
        event_type: str,
        plot_type: str,
        handler: Callable
    ) -> None:
        """Register an event handler for plot interactions."""
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

    def _add_interactive_controls(self, fig: go.Figure, plot_type: str) -> go.Figure:
        """Add interactive controls to figure."""
        controls = self.config.visualization.controls
        
        if controls.enable_range_selector:
            fig.update_xaxes(rangeslider=dict(visible=True))
            if plot_type == "financial_metrics":
                fig.update_xaxes(rangeselector=dict(buttons=controls.range_selector_options))
        
        if controls.enable_compare_mode and plot_type == "financial_metrics":
            buttons = []
            for metric in controls.compare_metrics:
                buttons.append(dict(
                    args=[{"visible": [metric in trace.name for trace in fig.data]}],
                    label=metric,
                    method="update"
                ))
            
            fig.update_layout(
                updatemenus=[dict(
                    type="dropdown",
                    direction="down",
                    buttons=buttons,
                    showactive=True,
                    x=0,
                    y=1.1,
                    xanchor="left",
                    yanchor="top"
                )]
            )
        
        if controls.enable_highlight:
            fig.update_traces(
                hoverinfo="x+y+name",
                line={"width": 3},
                marker={"size": 8},
                mode="lines+markers",
                showlegend=True
            )
        
        modebar_buttons = []
        if controls.enable_zoom:
            modebar_buttons.extend(['zoomIn2d', 'zoomOut2d'])
        if controls.enable_pan:
            modebar_buttons.append('pan2d')
        if controls.enable_reset:
            modebar_buttons.append('resetScale2d')
        
        if modebar_buttons:
            fig.update_layout(modebar=dict(add=modebar_buttons))
        
        return fig

[Previous plot creation methods remain the same, but add _add_interactive_controls call at the end]

def create_costbenefit_analyzer(
    intervention_analyzer: InterventionAnalyzer,
    config: Optional[CostConfig] = None
) -> CostBenefitAnalyzer:
    """Create cost-benefit analyzer."""
    return CostBenefitAnalyzer(intervention_analyzer, config)
