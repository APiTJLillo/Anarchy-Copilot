"""Animation features for tuning visualization."""

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
import plotly.express as px

from .tuning_visualization import TuningVisualizer, VisualizationConfig

logger = logging.getLogger(__name__)

@dataclass
class AnimationConfig:
    """Configuration for tuning animations."""
    frame_duration: int = 100
    transition_duration: int = 500
    redraw: bool = True
    easing: str = "cubic-in-out"
    dynamic_traces: bool = True
    max_frames: int = 200
    auto_play: bool = True
    output_path: Optional[Path] = None

class TuningAnimator:
    """Animate tuning visualizations."""
    
    def __init__(
        self,
        visualizer: TuningVisualizer,
        config: AnimationConfig
    ):
        self.visualizer = visualizer
        self.config = config
        self.frames: Dict[str, List[go.Frame]] = {}
    
    def create_optimization_animation(
        self,
        model_name: str
    ) -> go.Figure:
        """Create animated optimization history."""
        if model_name not in self.visualizer.tuner.study_results:
            return go.Figure()
        
        study = self.visualizer.tuner.study_results[model_name]
        trials_df = study.trials_dataframe()
        
        # Create figure
        fig = go.Figure()
        
        frames = []
        for i in range(len(trials_df)):
            frame_data = trials_df.iloc[:i+1]
            
            frame = go.Frame(
                data=[
                    # Trial points
                    go.Scatter(
                        x=frame_data.index,
                        y=frame_data["value"],
                        mode="markers",
                        name="Trials",
                        marker=dict(
                            size=8,
                            color=frame_data["value"],
                            colorscale="Viridis",
                            showscale=True
                        )
                    ),
                    # Best value line
                    go.Scatter(
                        x=frame_data.index,
                        y=np.minimum.accumulate(frame_data["value"]),
                        mode="lines",
                        name="Best Value",
                        line=dict(color="red", width=2)
                    )
                ],
                name=f"frame{i}"
            )
            frames.append(frame)
        
        # Add frames to figure
        fig.frames = frames
        
        # Add initial data
        fig.add_trace(
            go.Scatter(
                x=[trials_df.index[0]],
                y=[trials_df["value"].iloc[0]],
                mode="markers",
                name="Trials",
                marker=dict(size=8, color="blue")
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[trials_df.index[0]],
                y=[trials_df["value"].iloc[0]],
                mode="lines",
                name="Best Value",
                line=dict(color="red", width=2)
            )
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
                                    "duration": self.config.frame_duration,
                                    "redraw": self.config.redraw
                                },
                                "fromcurrent": True,
                                "transition": {
                                    "duration": self.config.transition_duration,
                                    "easing": self.config.easing
                                }
                            }
                        ],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": self.config.redraw
                                },
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
        
        # Add slider
        fig.update_layout(
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Trial:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {
                    "duration": self.config.transition_duration,
                    "easing": self.config.easing
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [f"frame{i}"],
                            {
                                "frame": {
                                    "duration": self.config.frame_duration,
                                    "redraw": self.config.redraw
                                },
                                "mode": "immediate",
                                "transition": {
                                    "duration": self.config.transition_duration
                                }
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
            title=f"{model_name} Optimization Progress",
            xaxis_title="Trial",
            yaxis_title="Value",
            showlegend=True,
            template="plotly_dark" if self.visualizer.config.dark_mode else "plotly_white"
        )
        
        self.frames[model_name] = frames
        return fig
    
    def create_parameter_animation(
        self,
        model_name: str
    ) -> go.Figure:
        """Create animated parameter space exploration."""
        if model_name not in self.visualizer.tuner.study_results:
            return go.Figure()
        
        study = self.visualizer.tuner.study_results[model_name]
        trials_df = study.trials_dataframe()
        
        # Get parameter columns
        param_cols = [c for c in trials_df.columns if c.startswith("params_")]
        if len(param_cols) < 2:
            return go.Figure()
        
        # Create figure
        fig = go.Figure()
        
        frames = []
        for i in range(len(trials_df)):
            frame_data = trials_df.iloc[:i+1]
            
            frame = go.Frame(
                data=[
                    # Parameter scatter plot
                    go.Scatter(
                        x=frame_data[param_cols[0]],
                        y=frame_data[param_cols[1]],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=frame_data["value"],
                            colorscale="Viridis",
                            showscale=True
                        ),
                        text=[
                            f"Trial {j}<br>" +
                            f"Value: {row['value']:.4f}<br>" +
                            "<br>".join(
                                f"{p.replace('params_', '')}: {row[p]}"
                                for p in param_cols
                            )
                            for j, row in frame_data.iterrows()
                        ],
                        hoverinfo="text"
                    )
                ],
                name=f"frame{i}"
            )
            frames.append(frame)
        
        # Add frames to figure
        fig.frames = frames
        
        # Add initial data
        fig.add_trace(
            go.Scatter(
                x=[trials_df[param_cols[0]].iloc[0]],
                y=[trials_df[param_cols[1]].iloc[0]],
                mode="markers",
                marker=dict(
                    size=10,
                    color=trials_df["value"].iloc[0],
                    colorscale="Viridis",
                    showscale=True
                )
            )
        )
        
        # Add animation controls and slider (similar to optimization animation)
        self._add_animation_controls(fig, len(frames))
        
        # Update layout
        fig.update_layout(
            title=f"{model_name} Parameter Space Exploration",
            xaxis_title=param_cols[0].replace("params_", ""),
            yaxis_title=param_cols[1].replace("params_", ""),
            template="plotly_dark" if self.visualizer.config.dark_mode else "plotly_white"
        )
        
        return fig
    
    def create_convergence_animation(
        self,
        model_name: str
    ) -> go.Figure:
        """Create animated convergence analysis."""
        if model_name not in self.visualizer.tuner.study_results:
            return go.Figure()
        
        study = self.visualizer.tuner.study_results[model_name]
        trials_df = study.trials_dataframe()
        
        fig = go.Figure()
        
        frames = []
        window_size = 5
        
        for i in range(window_size, len(trials_df)):
            frame_data = trials_df.iloc[:i+1]
            window = frame_data.iloc[-window_size:]
            
            frame = go.Frame(
                data=[
                    # Rolling mean
                    go.Scatter(
                        x=frame_data.index,
                        y=frame_data["value"].rolling(window_size).mean(),
                        mode="lines",
                        name="Rolling Mean",
                        line=dict(color="blue", width=2)
                    ),
                    # Rolling std
                    go.Scatter(
                        x=frame_data.index,
                        y=frame_data["value"].rolling(window_size).std(),
                        mode="lines",
                        name="Rolling Std",
                        line=dict(color="red", width=2)
                    ),
                    # Current window points
                    go.Scatter(
                        x=window.index,
                        y=window["value"],
                        mode="markers",
                        name="Current Window",
                        marker=dict(size=10, color="green")
                    )
                ],
                name=f"frame{i}"
            )
            frames.append(frame)
        
        # Add frames to figure
        fig.frames = frames
        
        # Add initial data
        fig.add_trace(go.Scatter(name="Rolling Mean"))  # Placeholder
        fig.add_trace(go.Scatter(name="Rolling Std"))   # Placeholder
        fig.add_trace(go.Scatter(name="Current Window"))  # Placeholder
        
        # Add animation controls and slider
        self._add_animation_controls(fig, len(frames))
        
        # Update layout
        fig.update_layout(
            title=f"{model_name} Convergence Analysis",
            xaxis_title="Trial",
            yaxis_title="Value",
            template="plotly_dark" if self.visualizer.config.dark_mode else "plotly_white"
        )
        
        return fig
    
    def _add_animation_controls(
        self,
        fig: go.Figure,
        n_frames: int
    ):
        """Add animation controls to figure."""
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": self.config.frame_duration,
                                    "redraw": self.config.redraw
                                },
                                "fromcurrent": True,
                                "transition": {
                                    "duration": self.config.transition_duration,
                                    "easing": self.config.easing
                                }
                            }
                        ],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": self.config.redraw
                                },
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
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {
                    "duration": self.config.transition_duration,
                    "easing": self.config.easing
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [f"frame{i}"],
                            {
                                "frame": {
                                    "duration": self.config.frame_duration,
                                    "redraw": self.config.redraw
                                },
                                "mode": "immediate",
                                "transition": {
                                    "duration": self.config.transition_duration
                                }
                            }
                        ],
                        "label": str(i),
                        "method": "animate"
                    }
                    for i in range(n_frames)
                ]
            }]
        )
    
    def save_animations(
        self,
        model_name: Optional[str] = None
    ):
        """Save animation results."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            if model_name:
                # Save model-specific animations
                opt_anim = self.create_optimization_animation(model_name)
                opt_anim.write_html(
                    str(output_path / f"{model_name}_optimization.html")
                )
                
                param_anim = self.create_parameter_animation(model_name)
                param_anim.write_html(
                    str(output_path / f"{model_name}_parameters.html")
                )
                
                conv_anim = self.create_convergence_animation(model_name)
                conv_anim.write_html(
                    str(output_path / f"{model_name}_convergence.html")
                )
            else:
                # Save animations for all models
                for model in self.visualizer.tuner.study_results.keys():
                    self.save_animations(model)
            
            logger.info(f"Saved animations to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save animations: {e}")

def create_tuning_animator(
    visualizer: TuningVisualizer,
    output_path: Optional[Path] = None
) -> TuningAnimator:
    """Create tuning animator."""
    config = AnimationConfig(output_path=output_path)
    return TuningAnimator(visualizer, config)

if __name__ == "__main__":
    # Example usage
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
    animator = create_tuning_animator(
        viz,
        output_path=Path("tuning_animations")
    )
    
    # Get training data
    data = learning._prepare_training_data("execution_time")
    if data is not None:
        X, y = data
        
        # Perform tuning
        tuner.tune_random_forest(X, y)
        tuner.tune_xgboost(X, y)
        
        # Generate animations
        animator.save_animations("rf")  # Random Forest animations
        animator.save_animations("xgb")  # XGBoost animations
