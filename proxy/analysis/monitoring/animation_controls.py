"""Animation controls for easing functions."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from .interactive_easing import InteractiveEasing, InteractiveConfig

logger = logging.getLogger(__name__)

@dataclass
class ControlConfig:
    """Configuration for animation controls."""
    loop: bool = True
    reverse: bool = False
    start_delay: float = 0.0
    end_delay: float = 0.0
    step_size: float = 0.01
    play_controls: bool = True
    timeline: bool = True
    repeat_count: Optional[int] = None
    output_path: Optional[Path] = None

class AnimationControls:
    """Control interface for easing animations."""
    
    def __init__(
        self,
        interactive: InteractiveEasing,
        config: ControlConfig
    ):
        self.interactive = interactive
        self.config = config
        self.setup_controls()
    
    def setup_controls(self):
        """Add animation controls to interactive interface."""
        control_layout = html.Div([
            dbc.Card([
                dbc.CardHeader("Animation Controls"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button(
                                    html.I(className="fas fa-play"),
                                    id="play-button",
                                    color="success",
                                    className="me-1"
                                ),
                                dbc.Button(
                                    html.I(className="fas fa-pause"),
                                    id="pause-button",
                                    color="warning",
                                    className="me-1"
                                ),
                                dbc.Button(
                                    html.I(className="fas fa-stop"),
                                    id="stop-button",
                                    color="danger",
                                    className="me-1"
                                ),
                                dbc.Button(
                                    html.I(className="fas fa-redo"),
                                    id="loop-button",
                                    color="info",
                                    className="me-1",
                                    active=self.config.loop
                                )
                            ]) if self.config.play_controls else None
                        ]),
                        dbc.Col([
                            dbc.Switch(
                                id="reverse-switch",
                                label="Reverse",
                                value=self.config.reverse
                            )
                        ])
                    ]),
                    
                    html.Br(),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Start Delay (s)"),
                            dbc.Input(
                                id="start-delay",
                                type="number",
                                value=self.config.start_delay,
                                min=0,
                                max=5,
                                step=0.1
                            )
                        ]),
                        dbc.Col([
                            dbc.Label("End Delay (s)"),
                            dbc.Input(
                                id="end-delay",
                                type="number",
                                value=self.config.end_delay,
                                min=0,
                                max=5,
                                step=0.1
                            )
                        ]),
                        dbc.Col([
                            dbc.Label("Repeat Count"),
                            dbc.Input(
                                id="repeat-count",
                                type="number",
                                value=self.config.repeat_count or -1,
                                min=-1
                            )
                        ])
                    ]),
                    
                    html.Br(),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Animation Progress"),
                            dcc.Slider(
                                id="progress-slider",
                                min=0,
                                max=1,
                                step=self.config.step_size,
                                value=0,
                                marks={
                                    0: "Start",
                                    0.25: "25%",
                                    0.5: "50%",
                                    0.75: "75%",
                                    1: "End"
                                },
                                updatemode="drag"
                            ) if self.config.timeline else None
                        ])
                    ])
                ])
            ]),
            
            html.Br(),
            
            dbc.Card([
                dbc.CardHeader("Animation State"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="animation-info")
                        ])
                    ])
                ])
            ]),
            
            dcc.Store(id="animation-state")
        ])
        
        # Add controls to interactive layout
        self.interactive.app.layout.children.insert(-1, control_layout)
        
        # Setup additional callbacks
        if self.config.play_controls:
            self.setup_control_callbacks()
    
    def setup_control_callbacks(self):
        """Setup callbacks for animation controls."""
        @self.interactive.app.callback(
            [
                Output("animation-state", "data"),
                Output("animation-box", "style")
            ],
            [
                Input("play-button", "n_clicks"),
                Input("pause-button", "n_clicks"),
                Input("stop-button", "n_clicks"),
                Input("loop-button", "n_clicks"),
                Input("reverse-switch", "value"),
                Input("progress-slider", "value"),
                Input("start-delay", "value"),
                Input("end-delay", "value"),
                Input("repeat-count", "value")
            ],
            [
                State("animation-state", "data"),
                State("easing-selector", "value"),
                State("animation-box", "style")
            ]
        )
        def update_animation(
            play_clicks: Optional[int],
            pause_clicks: Optional[int],
            stop_clicks: Optional[int],
            loop_clicks: Optional[int],
            reverse: bool,
            progress: float,
            start_delay: float,
            end_delay: float,
            repeat_count: int,
            current_state: Optional[Dict[str, Any]],
            selected_easing: List[str],
            current_style: Dict[str, Any]
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """Update animation state and style."""
            if not current_state:
                current_state = {
                    "playing": False,
                    "progress": 0,
                    "direction": 1,
                    "repeat_count": 0
                }
            
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if trigger_id == "play-button":
                current_state["playing"] = True
            elif trigger_id == "pause-button":
                current_state["playing"] = False
            elif trigger_id == "stop-button":
                current_state = {
                    "playing": False,
                    "progress": 0,
                    "direction": 1,
                    "repeat_count": 0
                }
            elif trigger_id == "reverse-switch":
                current_state["direction"] = -1 if reverse else 1
            elif trigger_id == "progress-slider":
                current_state["progress"] = progress
            
            # Update progress if playing
            if current_state["playing"]:
                if not selected_easing:
                    return current_state, current_style
                
                easing_func = self.interactive.visualizer.easing.get_easing_function(
                    selected_easing[0]
                )
                
                # Apply easing
                if reverse:
                    progress = 1 - easing_func(1 - current_state["progress"])
                else:
                    progress = easing_func(current_state["progress"])
                
                # Update position
                new_style = {
                    **current_style,
                    "left": f"{progress * 300}px",
                    "transition": (
                        f"left {1/self.interactive.config.refresh_interval}s "
                        "linear"
                    )
                }
                
                # Update progress
                new_progress = current_state["progress"] + (
                    self.config.step_size * current_state["direction"]
                )
                
                # Handle loop/repeat
                if new_progress >= 1 or new_progress <= 0:
                    if self.config.loop:
                        new_progress = 0 if new_progress >= 1 else 1
                        current_state["repeat_count"] += 1
                        
                        if (
                            repeat_count > 0 and
                            current_state["repeat_count"] >= repeat_count
                        ):
                            current_state["playing"] = False
                            new_progress = 0
                    else:
                        current_state["playing"] = False
                        new_progress = min(1, max(0, new_progress))
                
                current_state["progress"] = new_progress
                return current_state, new_style
            
            return current_state, current_style
        
        @self.interactive.app.callback(
            Output("animation-info", "children"),
            [Input("animation-state", "data")]
        )
        def update_animation_info(
            state: Optional[Dict[str, Any]]
        ) -> html.Div:
            """Update animation state information."""
            if not state:
                return html.P("Animation not started")
            
            return html.Div([
                html.P([
                    html.Strong("Status: "),
                    "Playing" if state["playing"] else "Stopped"
                ]),
                html.P([
                    html.Strong("Progress: "),
                    f"{state['progress']:.2%}"
                ]),
                html.P([
                    html.Strong("Direction: "),
                    "Reverse" if state["direction"] < 0 else "Forward"
                ]),
                html.P([
                    html.Strong("Repeat Count: "),
                    str(state["repeat_count"])
                ])
            ])
    
    def save_state(self):
        """Save animation control state."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            state = {
                "config": {
                    "loop": self.config.loop,
                    "reverse": self.config.reverse,
                    "start_delay": self.config.start_delay,
                    "end_delay": self.config.end_delay,
                    "step_size": self.config.step_size,
                    "repeat_count": self.config.repeat_count
                },
                "timestamp": datetime.now().isoformat()
            }
            
            state_file = output_path / "animation_state.json"
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved animation state to {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save animation state: {e}")
    
    def load_state(self):
        """Load animation control state."""
        if not self.config.output_path:
            return
        
        try:
            state_file = self.config.output_path / "animation_state.json"
            if not state_file.exists():
                return
            
            with open(state_file) as f:
                state = json.load(f)
            
            config = state.get("config", {})
            self.config.loop = config.get("loop", self.config.loop)
            self.config.reverse = config.get("reverse", self.config.reverse)
            self.config.start_delay = config.get("start_delay", self.config.start_delay)
            self.config.end_delay = config.get("end_delay", self.config.end_delay)
            self.config.step_size = config.get("step_size", self.config.step_size)
            self.config.repeat_count = config.get("repeat_count", self.config.repeat_count)
            
            logger.info(f"Loaded animation state from {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to load animation state: {e}")

def create_animation_controls(
    interactive: InteractiveEasing,
    output_path: Optional[Path] = None
) -> AnimationControls:
    """Create animation controls."""
    config = ControlConfig(output_path=output_path)
    return AnimationControls(interactive, config)

if __name__ == "__main__":
    # Example usage
    from .interactive_easing import create_interactive_easing
    from .easing_visualization import create_easing_visualizer
    from .easing_transitions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    visualizer = create_easing_visualizer(easing)
    interactive = create_interactive_easing(visualizer)
    controls = create_animation_controls(
        interactive,
        output_path=Path("animation_controls")
    )
    
    # Run interactive dashboard with controls
    interactive.run_server()
