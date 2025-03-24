"""Transition effects for learning animations."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path
import json
from datetime import datetime
from scipy.interpolate import interp1d
import plotly.express as px

from .learning_animation import LearningAnimator, AnimationConfig

logger = logging.getLogger(__name__)

@dataclass
class TransitionConfig:
    """Configuration for transition effects."""
    easing: str = "cubic-in-out"
    duration: int = 500
    interpolate_points: int = 30
    fade_duration: int = 200
    color_transition: bool = True
    size_transition: bool = True
    blend_frames: bool = True
    output_path: Optional[Path] = None

class TransitionEffects:
    """Add transition effects to learning animations."""
    
    def __init__(
        self,
        animator: LearningAnimator,
        config: TransitionConfig
    ):
        self.animator = animator
        self.config = config
    
    def add_transitions(self, fig: go.Figure) -> go.Figure:
        """Add transitions to figure."""
        if not fig.frames:
            return fig
        
        # Add intermediate frames
        new_frames = []
        for i in range(len(fig.frames) - 1):
            current = fig.frames[i]
            next_frame = fig.frames[i + 1]
            
            # Generate intermediate frames
            intermediates = self._interpolate_frames(current, next_frame)
            new_frames.extend(intermediates)
        
        # Update frames with transitions
        fig.frames = new_frames
        
        # Update layout with transition settings
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {
                            "frame": {
                                "duration": self.config.duration,
                                "redraw": True
                            },
                            "fromcurrent": True,
                            "transition": {
                                "duration": self.config.duration,
                                "easing": self.config.easing
                            }
                        }],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {
                            "frame": {"duration": 0},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "type": "buttons"
            }]
        )
        
        return fig
    
    def _interpolate_frames(
        self,
        frame1: go.Frame,
        frame2: go.Frame
    ) -> List[go.Frame]:
        """Interpolate between two frames."""
        frames = []
        steps = self.config.interpolate_points
        
        for i in range(steps):
            alpha = i / (steps - 1)
            
            # Create interpolated data
            new_data = []
            for trace1, trace2 in zip(frame1.data, frame2.data):
                if trace1.type == trace2.type:
                    if trace1.type in ["scatter", "scatter3d"]:
                        new_trace = self._interpolate_scatter(
                            trace1, trace2, alpha
                        )
                    elif trace1.type == "heatmap":
                        new_trace = self._interpolate_heatmap(
                            trace1, trace2, alpha
                        )
                    elif trace1.type == "bar":
                        new_trace = self._interpolate_bar(
                            trace1, trace2, alpha
                        )
                    else:
                        new_trace = trace1  # Fallback to original
                    
                    new_data.append(new_trace)
            
            # Create intermediate frame
            frame = go.Frame(
                data=new_data,
                name=f"{frame1.name}_{i}"
            )
            frames.append(frame)
        
        return frames
    
    def _interpolate_scatter(
        self,
        trace1: go.Scatter,
        trace2: go.Scatter,
        alpha: float
    ) -> go.Scatter:
        """Interpolate scatter traces."""
        # Interpolate positions
        x = self._lerp(trace1.x, trace2.x, alpha)
        y = self._lerp(trace1.y, trace2.y, alpha)
        
        # Interpolate colors if enabled
        marker = dict(trace1.marker or {})
        if self.config.color_transition and hasattr(trace1, "marker"):
            if isinstance(trace1.marker.color, (list, np.ndarray)):
                marker["color"] = self._lerp(
                    trace1.marker.color,
                    trace2.marker.color,
                    alpha
                )
        
        # Interpolate sizes if enabled
        if self.config.size_transition and hasattr(trace1, "marker"):
            if isinstance(trace1.marker.size, (list, np.ndarray)):
                marker["size"] = self._lerp(
                    trace1.marker.size,
                    trace2.marker.size,
                    alpha
                )
        
        return go.Scatter(
            x=x,
            y=y,
            mode=trace1.mode,
            name=trace1.name,
            marker=marker
        )
    
    def _interpolate_heatmap(
        self,
        trace1: go.Heatmap,
        trace2: go.Heatmap,
        alpha: float
    ) -> go.Heatmap:
        """Interpolate heatmap traces."""
        # Interpolate z values
        z = self._lerp(trace1.z, trace2.z, alpha)
        
        return go.Heatmap(
            z=z,
            colorscale=trace1.colorscale,
            showscale=trace1.showscale
        )
    
    def _interpolate_bar(
        self,
        trace1: go.Bar,
        trace2: go.Bar,
        alpha: float
    ) -> go.Bar:
        """Interpolate bar traces."""
        # Interpolate heights
        y = self._lerp(trace1.y, trace2.y, alpha)
        
        return go.Bar(
            x=trace1.x,
            y=y,
            name=trace1.name
        )
    
    def _lerp(
        self,
        start: np.ndarray,
        end: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        """Linear interpolation between arrays."""
        if isinstance(start, (list, tuple)):
            start = np.array(start)
        if isinstance(end, (list, tuple)):
            end = np.array(end)
        
        return start + alpha * (end - start)
    
    def add_fade_transitions(self, fig: go.Figure) -> go.Figure:
        """Add fade transitions between frames."""
        if not fig.frames:
            return fig
        
        for frame in fig.frames:
            for trace in frame.data:
                if hasattr(trace, "opacity"):
                    continue
                
                # Add opacity for fade effect
                if isinstance(trace, (go.Scatter, go.Scatter3d)):
                    trace.opacity = 0.7
                elif isinstance(trace, go.Heatmap):
                    trace.opacity = 0.8
                elif isinstance(trace, go.Bar):
                    trace.opacity = 0.9
        
        # Add fade animation settings
        fig.update_layout(
            transition={
                "duration": self.config.fade_duration,
                "easing": "cubic-in-out"
            }
        )
        
        return fig
    
    def add_color_transitions(self, fig: go.Figure) -> go.Figure:
        """Add color transition effects."""
        if not fig.frames:
            return fig
        
        # Create color scale interpolator
        colorscale = px.colors.sequential.Viridis
        
        for i, frame in enumerate(fig.frames):
            progress = i / (len(fig.frames) - 1)
            
            for trace in frame.data:
                if isinstance(trace, (go.Scatter, go.Scatter3d)):
                    if hasattr(trace, "marker"):
                        trace.marker.color = self._get_color(
                            progress, colorscale
                        )
                elif isinstance(trace, go.Heatmap):
                    trace.colorscale = [
                        [0, colorscale[0]],
                        [1, self._get_color(progress, colorscale)]
                    ]
        
        return fig
    
    def _get_color(
        self,
        progress: float,
        colorscale: List[str]
    ) -> str:
        """Get interpolated color from colorscale."""
        idx = int(progress * (len(colorscale) - 1))
        return colorscale[idx]
    
    def create_transition_preset(
        self,
        name: str,
        easing: str,
        duration: int,
        effects: List[str]
    ) -> Dict[str, Any]:
        """Create transition preset configuration."""
        preset = {
            "name": name,
            "easing": easing,
            "duration": duration,
            "effects": effects,
            "config": {
                "interpolate_points": self.config.interpolate_points,
                "fade_duration": self.config.fade_duration,
                "color_transition": "color" in effects,
                "size_transition": "size" in effects,
                "blend_frames": "blend" in effects
            }
        }
        
        if self.config.output_path:
            self._save_preset(name, preset)
        
        return preset
    
    def _save_preset(self, name: str, preset: Dict[str, Any]):
        """Save transition preset."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            preset_file = output_path / f"{name}_preset.json"
            with open(preset_file, "w") as f:
                json.dump(preset, f, indent=2)
            
            logger.info(f"Saved transition preset to {preset_file}")
            
        except Exception as e:
            logger.error(f"Failed to save preset: {e}")

def create_transition_effects(
    animator: LearningAnimator,
    output_path: Optional[Path] = None
) -> TransitionEffects:
    """Create transition effects."""
    config = TransitionConfig(output_path=output_path)
    return TransitionEffects(animator, config)

if __name__ == "__main__":
    # Example usage
    from .learning_animation import create_learning_animator
    from .learning_visualization import create_learning_visualizer
    from .reinforcement_optimization import create_rl_optimizer
    from .adaptive_optimization import create_adaptive_optimizer
    from .filter_optimization import create_filter_optimizer
    from .validation_filters import create_filter_manager
    from .preset_validation import create_preset_validator
    from .visualization_presets import create_preset_manager
    
    # Create components
    preset_manager = create_preset_manager()
    validator = create_preset_validator(preset_manager)
    filter_manager = create_filter_manager(validator)
    optimizer = create_filter_optimizer(filter_manager)
    adaptive = create_adaptive_optimizer(optimizer)
    rl_optimizer = create_rl_optimizer(adaptive)
    visualizer = create_learning_visualizer(rl_optimizer)
    animator = create_learning_animator(visualizer)
    
    # Create transition effects
    effects = create_transition_effects(
        animator,
        output_path=Path("transition_effects")
    )
    
    # Create and save animations with effects
    fig = animator.create_learning_progress_animation()
    fig = effects.add_transitions(fig)
    fig = effects.add_fade_transitions(fig)
    fig = effects.add_color_transitions(fig)
    
    # Save with effects
    fig.write_html("learning_progress_with_effects.html")
    
    # Create transition preset
    effects.create_transition_preset(
        "smooth_fade",
        "cubic-in-out",
        500,
        ["color", "size", "blend"]
    )
