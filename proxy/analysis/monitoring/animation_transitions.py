"""Transition effects for tuning animations."""

from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime
import plotly.graph_objects as go

from .tuning_animation import TuningAnimator, AnimationConfig

logger = logging.getLogger(__name__)

@dataclass
class TransitionConfig:
    """Configuration for animation transitions."""
    smoothing: bool = True
    fade_duration: int = 300
    scale_duration: int = 400
    color_duration: int = 500
    stagger_delay: int = 50
    ease_type: str = "cubic"
    interpolation: str = "linear"
    morph_shapes: bool = True
    output_path: Optional[Path] = None

class AnimationTransitions:
    """Transition effects for animations."""
    
    def __init__(
        self,
        animator: TuningAnimator,
        config: TransitionConfig
    ):
        self.animator = animator
        self.config = config
        self.transitions: Dict[str, Dict[str, Any]] = {}
    
    def add_fade_transition(
        self,
        fig: go.Figure,
        start_opacity: float = 0.0,
        end_opacity: float = 1.0
    ):
        """Add fade transition effect."""
        for trace in fig.data:
            if not hasattr(trace, "visible") or trace.visible:
                trace.opacity = start_opacity
        
        transition = {
            "duration": self.config.fade_duration,
            "easing": self.config.ease_type,
            "opacity": [
                [0.0, 1.0],
                [start_opacity, end_opacity]
            ]
        }
        
        return self._create_transition_frame(
            fig,
            transition,
            "fade"
        )
    
    def add_scale_transition(
        self,
        fig: go.Figure,
        start_scale: float = 0.5,
        end_scale: float = 1.0
    ):
        """Add scale transition effect."""
        for trace in fig.data:
            if hasattr(trace, "marker"):
                trace.marker.size *= start_scale
        
        transition = {
            "duration": self.config.scale_duration,
            "easing": self.config.ease_type,
            "transform": [
                [0.0, 1.0],
                [f"scale({start_scale})", f"scale({end_scale})"]
            ]
        }
        
        return self._create_transition_frame(
            fig,
            transition,
            "scale"
        )
    
    def add_color_transition(
        self,
        fig: go.Figure,
        start_color: str = "rgba(0,0,0,0)",
        end_color: Optional[str] = None
    ):
        """Add color transition effect."""
        original_colors = {}
        
        for i, trace in enumerate(fig.data):
            if hasattr(trace, "marker") and hasattr(trace.marker, "color"):
                original_colors[i] = trace.marker.color
                trace.marker.color = start_color
        
        transition = {
            "duration": self.config.color_duration,
            "easing": self.config.ease_type,
        }
        
        if end_color:
            transition["color"] = [
                [0.0, 1.0],
                [start_color, end_color]
            ]
        else:
            transition["targets"] = original_colors
        
        return self._create_transition_frame(
            fig,
            transition,
            "color"
        )
    
    def add_stagger_transition(
        self,
        fig: go.Figure,
        property_name: str,
        start_value: Union[float, str],
        end_value: Union[float, str]
    ):
        """Add staggered transition effect."""
        transitions = []
        
        for i, trace in enumerate(fig.data):
            delay = i * self.config.stagger_delay
            
            transition = {
                "duration": self.config.fade_duration,
                "easing": self.config.ease_type,
                "delay": delay,
                property_name: [
                    [0.0, 1.0],
                    [start_value, end_value]
                ]
            }
            
            transitions.append(
                self._create_transition_frame(
                    fig,
                    transition,
                    f"stagger_{i}"
                )
            )
        
        return transitions
    
    def add_morph_transition(
        self,
        source_fig: go.Figure,
        target_fig: go.Figure
    ):
        """Add shape morphing transition."""
        if not self.config.morph_shapes:
            return None
        
        def interpolate_points(p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
            """Interpolate between point sets."""
            if p1.shape != p2.shape:
                # Resample points if shapes don't match
                target_len = max(len(p1), len(p2))
                p1 = np.interp(
                    np.linspace(0, 1, target_len),
                    np.linspace(0, 1, len(p1)),
                    p1
                )
                p2 = np.interp(
                    np.linspace(0, 1, target_len),
                    np.linspace(0, 1, len(p2)),
                    p2
                )
            
            return p1 * (1 - t) + p2 * t
        
        frames = []
        steps = 20  # Number of interpolation steps
        
        for i in range(steps + 1):
            t = i / steps
            frame_data = []
            
            for source, target in zip(source_fig.data, target_fig.data):
                # Interpolate positions
                if hasattr(source, "x") and hasattr(target, "x"):
                    x = interpolate_points(
                        np.array(source.x),
                        np.array(target.x),
                        t
                    )
                    y = interpolate_points(
                        np.array(source.y),
                        np.array(target.y),
                        t
                    )
                else:
                    continue
                
                # Create interpolated trace
                trace = go.Scatter(
                    x=x,
                    y=y,
                    mode=source.mode,
                    name=source.name
                )
                
                # Interpolate marker properties
                if hasattr(source, "marker") and hasattr(target, "marker"):
                    marker_props = {}
                    
                    if hasattr(source.marker, "size") and hasattr(target.marker, "size"):
                        size = np.interp(
                            t,
                            [0, 1],
                            [source.marker.size, target.marker.size]
                        )
                        marker_props["size"] = size
                    
                    if hasattr(source.marker, "color") and hasattr(target.marker, "color"):
                        if isinstance(source.marker.color, str):
                            # Use color at endpoints
                            color = (
                                source.marker.color if t < 0.5
                                else target.marker.color
                            )
                        else:
                            # Interpolate color values
                            color = interpolate_points(
                                np.array(source.marker.color),
                                np.array(target.marker.color),
                                t
                            )
                        marker_props["color"] = color
                    
                    trace.marker = marker_props
                
                frame_data.append(trace)
            
            frames.append(
                go.Frame(
                    data=frame_data,
                    name=f"morph_{i}"
                )
            )
        
        return frames
    
    def _create_transition_frame(
        self,
        fig: go.Figure,
        transition: Dict[str, Any],
        name: str
    ) -> go.Frame:
        """Create transition animation frame."""
        frame_data = []
        
        for trace in fig.data:
            # Create new trace with transition properties
            new_trace = trace.update()
            
            # Apply transition properties
            if "opacity" in transition:
                new_trace.opacity = transition["opacity"][1][1]
            
            if "transform" in transition:
                new_trace.transforms = [{
                    "type": "transform",
                    "transform": transition["transform"][1][1]
                }]
            
            if "color" in transition:
                if hasattr(new_trace, "marker"):
                    new_trace.marker.color = transition["color"][1][1]
            
            if "targets" in transition:
                trace_idx = list(fig.data).index(trace)
                if trace_idx in transition["targets"]:
                    if hasattr(new_trace, "marker"):
                        new_trace.marker.color = transition["targets"][trace_idx]
            
            frame_data.append(new_trace)
        
        frame = go.Frame(
            data=frame_data,
            name=name
        )
        
        # Store transition for reference
        self.transitions[name] = transition
        
        return frame
    
    def apply_transitions(
        self,
        fig: go.Figure,
        transition_sequence: List[str]
    ):
        """Apply sequence of transitions to figure."""
        current_frames = list(fig.frames or [])
        total_duration = 0
        
        for name in transition_sequence:
            if name not in self.transitions:
                continue
            
            transition = self.transitions[name]
            duration = transition.get("duration", 0)
            delay = transition.get("delay", 0)
            
            # Update transition timing
            transition["start"] = total_duration + delay
            total_duration += duration + delay
            
            # Find and update corresponding frame
            for frame in current_frames:
                if frame.name == name:
                    frame.transition = transition
                    break
        
        # Update figure
        fig.frames = current_frames
        
        # Add animation controls if needed
        if not hasattr(fig.layout, "updatemenus"):
            self._add_animation_controls(fig)
    
    def _add_animation_controls(
        self,
        fig: go.Figure
    ):
        """Add animation control buttons."""
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 0},
                                "mode": "immediate",
                                "transition": {"duration": 0}
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
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        )

def create_animation_transitions(
    animator: TuningAnimator,
    output_path: Optional[Path] = None
) -> AnimationTransitions:
    """Create animation transitions."""
    config = TransitionConfig(output_path=output_path)
    return AnimationTransitions(animator, config)

if __name__ == "__main__":
    # Example usage
    from .tuning_animation import create_tuning_animator
    from .tuning_visualization import create_tuning_visualizer
    from .hyperparameter_tuning import create_hyperparameter_tuner
    from .profile_learning import create_profile_learning
    from .profile_analysis import create_profile_analyzer
    from .prediction_profiling import create_prediction_profiler
    from .prediction_performance import create_prediction_performance
    from .realtime_prediction import create_realtime_prediction
    from .prediction_controls import create_interactive_controls
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
    controls = create_interactive_controls(visualizer)
    realtime = create_realtime_prediction(controls)
    performance = create_prediction_performance(realtime)
    profiler = create_prediction_profiler(performance)
    analyzer = create_profile_analyzer(profiler)
    learning = create_profile_learning(analyzer)
    tuner = create_hyperparameter_tuner(learning)
    viz = create_tuning_visualizer(tuner)
    animator = create_tuning_animator(viz)
    transitions = create_animation_transitions(
        animator,
        output_path=Path("animation_transitions")
    )
    
    # Get training data
    data = learning._prepare_training_data("execution_time")
    if data is not None:
        X, y = data
        
        # Perform tuning
        tuner.tune_random_forest(X, y)
        
        # Create animation with transitions
        opt_anim = animator.create_optimization_animation("rf")
        
        # Add transitions
        transitions.add_fade_transition(opt_anim)
        transitions.add_scale_transition(opt_anim)
        transitions.add_color_transition(opt_anim)
        
        # Apply transition sequence
        transitions.apply_transitions(
            opt_anim,
            ["fade", "scale", "color"]
        )
        
        # Save animation
        opt_anim.write_html("rf_optimization_animated.html")
