"""Animation tools for learning visualization."""

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
import plotly.express as px
from IPython.display import HTML

from .learning_visualization import LearningVisualizer, VisualizationConfig

logger = logging.getLogger(__name__)

@dataclass
class AnimationConfig:
    """Configuration for learning animations."""
    frame_duration: int = 100
    transition_duration: int = 50
    fps: int = 30
    max_frames: int = 100
    smooth_transitions: bool = True
    loop: bool = True
    show_slider: bool = True
    show_buttons: bool = True
    output_path: Optional[Path] = None

class LearningAnimator:
    """Animate learning visualizations."""
    
    def __init__(
        self,
        visualizer: LearningVisualizer,
        config: AnimationConfig
    ):
        self.visualizer = visualizer
        self.config = config
        self.frames: Dict[str, List[go.Frame]] = {}
    
    def create_reward_evolution(self) -> go.Figure:
        """Create animated reward evolution."""
        rewards = self.visualizer.history["rewards"]
        steps = np.arange(len(rewards))
        
        # Create frames
        frames = []
        window_size = min(50, len(rewards))
        
        for i in range(len(rewards) - window_size + 1):
            window_rewards = rewards[i:i+window_size]
            window_steps = steps[i:i+window_size]
            
            # Calculate statistics
            mean = np.mean(window_rewards)
            std = np.std(window_rewards)
            
            frame = go.Frame(
                data=[
                    # Reward line
                    go.Scatter(
                        x=window_steps,
                        y=window_rewards,
                        mode="lines",
                        name="Rewards",
                        line=dict(color="blue")
                    ),
                    # Confidence interval
                    go.Scatter(
                        x=np.concatenate([window_steps, window_steps[::-1]]),
                        y=np.concatenate([
                            mean + std * np.ones_like(window_steps),
                            (mean - std * np.ones_like(window_steps))[::-1]
                        ]),
                        fill="toself",
                        fillcolor="rgba(0,100,80,0.2)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="Confidence"
                    ),
                    # Moving average
                    go.Scatter(
                        x=window_steps,
                        y=pd.Series(window_rewards).rolling(10).mean(),
                        mode="lines",
                        name="Moving Average",
                        line=dict(color="red", width=2)
                    )
                ],
                name=str(i)
            )
            frames.append(frame)
        
        # Create base figure
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title="Reward Evolution",
            xaxis_title="Step",
            yaxis_title="Reward",
            template="plotly_dark" if self.visualizer.config.dark_mode else "plotly_white",
            showlegend=True,
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {
                            "frame": {
                                "duration": self.config.frame_duration,
                                "redraw": True
                            },
                            "fromcurrent": True,
                            "transition": {
                                "duration": self.config.transition_duration
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
            }],
            sliders=[{
                "currentvalue": {"prefix": "Frame: "},
                "steps": [
                    {
                        "args": [[str(i)], {
                            "frame": {"duration": 0},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }],
                        "label": str(i),
                        "method": "animate"
                    }
                    for i in range(len(frames))
                ]
            }] if self.config.show_slider else None
        )
        
        return fig
    
    def create_policy_evolution(self) -> go.Figure:
        """Create animated policy evolution."""
        states = np.array([
            self.visualizer._encode_state(s)
            for s in self.visualizer.history["states"]
        ])
        q_values = self.visualizer.rl_optimizer.q_network.forward(states)
        policies = np.exp(q_values) / np.sum(np.exp(q_values), axis=1)[:, None]
        
        # Create frames
        frames = []
        window_size = min(20, len(policies))
        
        for i in range(len(policies) - window_size + 1):
            window_policies = policies[i:i+window_size]
            
            frame = go.Frame(
                data=[
                    go.Heatmap(
                        z=window_policies.T,
                        colorscale="Viridis",
                        showscale=True,
                        name="Policy"
                    )
                ],
                name=str(i)
            )
            frames.append(frame)
        
        # Create base figure
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Add animation controls
        self._add_animation_controls(fig, len(frames))
        
        fig.update_layout(
            title="Policy Evolution",
            xaxis_title="State",
            yaxis_title="Action",
            template="plotly_dark" if self.visualizer.config.dark_mode else "plotly_white"
        )
        
        return fig
    
    def create_state_space_animation(self) -> go.Figure:
        """Create animated state space exploration."""
        states = np.array([
            self.visualizer._encode_state(s)
            for s in self.visualizer.history["states"]
        ])
        
        if states.shape[1] < 3:
            return go.Figure()  # Not enough dimensions
        
        # Create frames
        frames = []
        window_size = min(50, len(states))
        
        for i in range(len(states) - window_size + 1):
            window_states = states[i:i+window_size]
            window_rewards = self.visualizer.history["rewards"][i:i+window_size]
            
            frame = go.Frame(
                data=[
                    go.Scatter3d(
                        x=window_states[:, 0],
                        y=window_states[:, 1],
                        z=window_states[:, 2],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=window_rewards,
                            colorscale="Viridis",
                            showscale=True
                        ),
                        name="States"
                    )
                ],
                name=str(i)
            )
            frames.append(frame)
        
        # Create base figure
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Add animation controls
        self._add_animation_controls(fig, len(frames))
        
        fig.update_layout(
            title="State Space Exploration",
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3"
            ),
            template="plotly_dark" if self.visualizer.config.dark_mode else "plotly_white"
        )
        
        return fig
    
    def create_learning_progress_animation(self) -> go.Figure:
        """Create comprehensive learning progress animation."""
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scatter"}, {"type": "scatter3d"}],
                [{"type": "heatmap"}, {"type": "bar"}]
            ],
            subplot_titles=[
                "Reward Evolution",
                "State Exploration",
                "Policy Heatmap",
                "Strategy Performance"
            ]
        )
        
        # Generate frames
        frames = []
        window_size = min(30, len(self.visualizer.history["rewards"]))
        
        for i in range(len(self.visualizer.history["rewards"]) - window_size + 1):
            frame_data = []
            
            # Reward evolution
            window_rewards = self.visualizer.history["rewards"][i:i+window_size]
            frame_data.append(
                go.Scatter(
                    x=np.arange(window_size),
                    y=window_rewards,
                    mode="lines",
                    name="Rewards"
                )
            )
            
            # State exploration (if enough dimensions)
            states = np.array([
                self.visualizer._encode_state(s)
                for s in self.visualizer.history["states"][i:i+window_size]
            ])
            if states.shape[1] >= 3:
                frame_data.append(
                    go.Scatter3d(
                        x=states[:, 0],
                        y=states[:, 1],
                        z=states[:, 2],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=window_rewards,
                            colorscale="Viridis"
                        ),
                        name="States"
                    )
                )
            
            # Policy heatmap
            q_values = self.visualizer.rl_optimizer.q_network.forward(states)
            policy = np.exp(q_values) / np.sum(np.exp(q_values), axis=1)[:, None]
            frame_data.append(
                go.Heatmap(
                    z=policy.T,
                    colorscale="RdBu",
                    name="Policy"
                )
            )
            
            # Strategy performance
            strategies = self.visualizer.history["strategies"][i:i+window_size]
            strategy_rewards = defaultdict(list)
            for s, r in zip(strategies, window_rewards):
                strategy_rewards[s].append(r)
            
            avg_rewards = {
                s: np.mean(r) for s, r in strategy_rewards.items()
            }
            frame_data.append(
                go.Bar(
                    x=list(avg_rewards.keys()),
                    y=list(avg_rewards.values()),
                    name="Strategy Rewards"
                )
            )
            
            frames.append(go.Frame(
                data=frame_data,
                name=str(i)
            ))
        
        # Set initial data
        for trace in frames[0].data:
            fig.add_trace(trace)
        
        # Add frames
        fig.frames = frames
        
        # Add animation controls
        self._add_animation_controls(fig, len(frames))
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            template="plotly_dark" if self.visualizer.config.dark_mode else "plotly_white"
        )
        
        return fig
    
    def _add_animation_controls(self, fig: go.Figure, n_frames: int):
        """Add animation controls to figure."""
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {
                            "frame": {
                                "duration": self.config.frame_duration,
                                "redraw": True
                            },
                            "fromcurrent": True,
                            "transition": {
                                "duration": self.config.transition_duration,
                                "easing": "quadratic-in-out"
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
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": self.config.transition_duration},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[str(i)], {
                            "frame": {"duration": 0},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }],
                        "label": str(i),
                        "method": "animate"
                    }
                    for i in range(n_frames)
                ]
            }] if self.config.show_slider else None
        )
    
    def save_animations(self):
        """Save all animations."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save reward evolution
            reward_fig = self.create_reward_evolution()
            reward_fig.write_html(
                str(output_path / "reward_evolution.html"),
                include_plotlyjs="cdn"
            )
            
            # Save policy evolution
            policy_fig = self.create_policy_evolution()
            policy_fig.write_html(
                str(output_path / "policy_evolution.html"),
                include_plotlyjs="cdn"
            )
            
            # Save state space animation
            state_fig = self.create_state_space_animation()
            state_fig.write_html(
                str(output_path / "state_space.html"),
                include_plotlyjs="cdn"
            )
            
            # Save comprehensive animation
            progress_fig = self.create_learning_progress_animation()
            progress_fig.write_html(
                str(output_path / "learning_progress.html"),
                include_plotlyjs="cdn"
            )
            
            logger.info(f"Saved animations to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save animations: {e}")

def create_learning_animator(
    visualizer: LearningVisualizer,
    output_path: Optional[Path] = None
) -> LearningAnimator:
    """Create learning animator."""
    config = AnimationConfig(output_path=output_path)
    return LearningAnimator(visualizer, config)

if __name__ == "__main__":
    # Example usage
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
    
    # Create animator
    animator = create_learning_animator(
        visualizer,
        output_path=Path("learning_animations")
    )
    
    # Generate and save animations
    animator.save_animations()
