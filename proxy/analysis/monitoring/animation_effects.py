"""Animation transition effects and enhancements."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import pandas as pd
import logging
from pathlib import Path
from copy import deepcopy
import json
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

@dataclass
class TransitionConfig:
    """Configuration for animation transitions."""
    duration: int = 500
    easing: str = "cubic-in-out"
    fade_duration: int = 200
    node_transition: str = "linear"
    color_interpolation: str = "rgb"
    frame_overlap: float = 0.2
    smoothing_window: int = 3

class AnimationEffects:
    """Enhanced animation effects and transitions."""
    
    def __init__(self, config: TransitionConfig):
        self.config = config
    
    def apply_matrix_transitions(
        self,
        frames: List[go.Frame],
        correlation_data: List[np.ndarray]
    ) -> List[go.Frame]:
        """Apply smooth transitions to correlation matrix."""
        enhanced_frames = []
        
        for i in range(len(frames)):
            # Get current frame data
            current = correlation_data[i]
            
            if i < len(frames) - 1:
                # Interpolate between current and next frame
                next_data = correlation_data[i + 1]
                interpolated = self._interpolate_matrices(
                    current,
                    next_data,
                    num_steps=int(self.config.duration * self.config.frame_overlap)
                )
                
                # Create intermediate frames
                for step, matrix in enumerate(interpolated):
                    frame = deepcopy(frames[i])
                    frame.data[0].z = matrix
                    frame.name = f"{frames[i].name}_step_{step}"
                    enhanced_frames.append(frame)
            else:
                # Add final frame
                enhanced_frames.append(frames[i])
        
        return enhanced_frames
    
    def apply_network_transitions(
        self,
        frames: List[go.Frame],
        positions: List[Dict[str, Tuple[float, float]]],
        edge_weights: List[Dict[Tuple[str, str], float]]
    ) -> List[go.Frame]:
        """Apply smooth transitions to network animation."""
        enhanced_frames = []
        
        for i in range(len(frames)):
            # Get current positions and weights
            current_pos = positions[i]
            current_weights = edge_weights[i]
            
            if i < len(frames) - 1:
                # Get next state
                next_pos = positions[i + 1]
                next_weights = edge_weights[i + 1]
                
                # Create intermediate frames
                steps = int(self.config.duration * self.config.frame_overlap)
                for step in range(steps):
                    alpha = step / steps
                    
                    # Interpolate positions
                    interp_pos = self._interpolate_positions(
                        current_pos,
                        next_pos,
                        alpha
                    )
                    
                    # Interpolate edge weights
                    interp_weights = self._interpolate_weights(
                        current_weights,
                        next_weights,
                        alpha
                    )
                    
                    # Create frame with interpolated data
                    frame = self._create_network_frame(
                        interp_pos,
                        interp_weights,
                        f"{frames[i].name}_step_{step}"
                    )
                    enhanced_frames.append(frame)
            else:
                # Add final frame
                enhanced_frames.append(frames[i])
        
        return enhanced_frames
    
    def apply_cluster_transitions(
        self,
        frames: List[go.Frame],
        cluster_data: List[Dict[str, Any]]
    ) -> List[go.Frame]:
        """Apply smooth transitions to cluster animation."""
        enhanced_frames = []
        
        for i in range(len(frames)):
            # Get current cluster data
            current = cluster_data[i]
            
            if i < len(frames) - 1:
                # Get next cluster state
                next_data = cluster_data[i + 1]
                
                # Create intermediate frames
                steps = int(self.config.duration * self.config.frame_overlap)
                for step in range(steps):
                    alpha = step / steps
                    
                    # Interpolate cluster positions and sizes
                    interp_data = self._interpolate_clusters(
                        current,
                        next_data,
                        alpha
                    )
                    
                    # Create frame with interpolated data
                    frame = self._create_cluster_frame(
                        interp_data,
                        f"{frames[i].name}_step_{step}"
                    )
                    enhanced_frames.append(frame)
            else:
                # Add final frame
                enhanced_frames.append(frames[i])
        
        return enhanced_frames
    
    def _interpolate_matrices(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_steps: int
    ) -> List[np.ndarray]:
        """Interpolate between correlation matrices."""
        matrices = []
        for step in range(num_steps):
            alpha = step / num_steps
            interpolated = start * (1 - alpha) + end * alpha
            
            # Apply smoothing if configured
            if self.config.smoothing_window > 1:
                interpolated = self._smooth_matrix(
                    interpolated,
                    self.config.smoothing_window
                )
            
            matrices.append(interpolated)
        return matrices
    
    def _interpolate_positions(
        self,
        start: Dict[str, Tuple[float, float]],
        end: Dict[str, Tuple[float, float]],
        alpha: float
    ) -> Dict[str, Tuple[float, float]]:
        """Interpolate node positions."""
        positions = {}
        nodes = set(start.keys()) | set(end.keys())
        
        for node in nodes:
            if node in start and node in end:
                # Interpolate existing node
                x1, y1 = start[node]
                x2, y2 = end[node]
                positions[node] = (
                    x1 * (1 - alpha) + x2 * alpha,
                    y1 * (1 - alpha) + y2 * alpha
                )
            elif node in start:
                # Fade out node
                positions[node] = start[node]
            else:
                # Fade in node
                positions[node] = end[node]
        
        return positions
    
    def _interpolate_weights(
        self,
        start: Dict[Tuple[str, str], float],
        end: Dict[Tuple[str, str], float],
        alpha: float
    ) -> Dict[Tuple[str, str], float]:
        """Interpolate edge weights."""
        weights = {}
        edges = set(start.keys()) | set(end.keys())
        
        for edge in edges:
            if edge in start and edge in end:
                # Interpolate existing edge
                w1 = start[edge]
                w2 = end[edge]
                weights[edge] = w1 * (1 - alpha) + w2 * alpha
            elif edge in start:
                # Fade out edge
                weights[edge] = start[edge] * (1 - alpha)
            else:
                # Fade in edge
                weights[edge] = end[edge] * alpha
        
        return weights
    
    def _interpolate_clusters(
        self,
        start: Dict[str, Any],
        end: Dict[str, Any],
        alpha: float
    ) -> Dict[str, Any]:
        """Interpolate cluster properties."""
        interpolated = {
            "positions": self._interpolate_positions(
                start["positions"],
                end["positions"],
                alpha
            ),
            "sizes": {},
            "colors": {}
        }
        
        # Interpolate sizes and colors
        nodes = set(start["sizes"].keys()) | set(end["sizes"].keys())
        for node in nodes:
            if node in start["sizes"] and node in end["sizes"]:
                # Interpolate existing node
                s1 = start["sizes"][node]
                s2 = end["sizes"][node]
                interpolated["sizes"][node] = s1 * (1 - alpha) + s2 * alpha
                
                c1 = start["colors"][node]
                c2 = end["colors"][node]
                interpolated["colors"][node] = self._interpolate_color(c1, c2, alpha)
            elif node in start["sizes"]:
                # Fade out node
                interpolated["sizes"][node] = start["sizes"][node] * (1 - alpha)
                interpolated["colors"][node] = start["colors"][node]
            else:
                # Fade in node
                interpolated["sizes"][node] = end["sizes"][node] * alpha
                interpolated["colors"][node] = end["colors"][node]
        
        return interpolated
    
    def _smooth_matrix(
        self,
        matrix: np.ndarray,
        window_size: int
    ) -> np.ndarray:
        """Apply smoothing to matrix values."""
        kernel = np.ones((window_size, window_size)) / (window_size * window_size)
        return np.real(
            np.fft.ifft2(
                np.fft.fft2(matrix) * np.fft.fft2(kernel, matrix.shape)
            )
        )
    
    def _interpolate_color(
        self,
        color1: str,
        color2: str,
        alpha: float
    ) -> str:
        """Interpolate between two colors."""
        # Convert hex to RGB if needed
        if color1.startswith("#"):
            color1 = self._hex_to_rgb(color1)
        if color2.startswith("#"):
            color2 = self._hex_to_rgb(color2)
        
        # Parse RGB values
        r1, g1, b1 = map(int, color1.strip("rgb()").split(","))
        r2, g2, b2 = map(int, color2.strip("rgb()").split(","))
        
        # Interpolate
        r = int(r1 * (1 - alpha) + r2 * alpha)
        g = int(g1 * (1 - alpha) + g2 * alpha)
        b = int(b1 * (1 - alpha) + b2 * alpha)
        
        return f"rgb({r},{g},{b})"
    
    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert hex color to RGB string."""
        h = hex_color.lstrip("#")
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    
    def _create_network_frame(
        self,
        positions: Dict[str, Tuple[float, float]],
        weights: Dict[Tuple[str, str], float],
        name: str
    ) -> go.Frame:
        """Create network frame with given positions and weights."""
        edge_x = []
        edge_y = []
        edge_colors = []
        
        for edge, weight in weights.items():
            n1, n2 = edge
            if n1 in positions and n2 in positions:
                x1, y1 = positions[n1]
                x2, y2 = positions[n2]
                edge_x.extend([x1, x2, None])
                edge_y.extend([y1, y2, None])
                edge_colors.extend([weight, weight, None])
        
        node_x = []
        node_y = []
        for node, (x, y) in positions.items():
            node_x.append(x)
            node_y.append(y)
        
        return go.Frame(
            data=[
                go.Scatter(  # Edges
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(
                        width=1,
                        color=edge_colors,
                        colorscale="Viridis"
                    ),
                    hoverinfo="none"
                ),
                go.Scatter(  # Nodes
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    text=list(positions.keys()),
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color="lightblue",
                        line=dict(width=2)
                    )
                )
            ],
            name=name
        )
    
    def _create_cluster_frame(
        self,
        cluster_data: Dict[str, Any],
        name: str
    ) -> go.Frame:
        """Create cluster frame with interpolated data."""
        positions = cluster_data["positions"]
        sizes = cluster_data["sizes"]
        colors = cluster_data["colors"]
        
        x_positions = []
        y_positions = []
        marker_sizes = []
        marker_colors = []
        texts = []
        
        for node, (x, y) in positions.items():
            x_positions.append(x)
            y_positions.append(y)
            marker_sizes.append(sizes[node])
            marker_colors.append(colors[node])
            texts.append(node)
        
        return go.Frame(
            data=[go.Scatter(
                x=x_positions,
                y=y_positions,
                mode="markers+text",
                text=texts,
                textposition="top center",
                marker=dict(
                    size=marker_sizes,
                    color=marker_colors,
                    colorscale="Viridis",
                    showscale=True
                ),
                hoverinfo="text"
            )],
            name=name
        )

def enhance_animations(
    frames: Dict[str, List[go.Frame]],
    data: Dict[str, Any],
    config: Optional[TransitionConfig] = None
) -> Dict[str, List[go.Frame]]:
    """Enhance animations with transitions."""
    effects = AnimationEffects(config or TransitionConfig())
    enhanced = {}
    
    # Enhance matrix animation
    if "matrix" in frames and "correlation_data" in data:
        enhanced["matrix"] = effects.apply_matrix_transitions(
            frames["matrix"],
            data["correlation_data"]
        )
    
    # Enhance network animation
    if "network" in frames and "network_data" in data:
        enhanced["network"] = effects.apply_network_transitions(
            frames["network"],
            data["network_data"]["positions"],
            data["network_data"]["weights"]
        )
    
    # Enhance cluster animation
    if "clusters" in frames and "cluster_data" in data:
        enhanced["clusters"] = effects.apply_cluster_transitions(
            frames["clusters"],
            data["cluster_data"]
        )
    
    return enhanced

if __name__ == "__main__":
    # Example usage
    from .correlation_animation import create_correlation_animations
    
    # Create sample data and animations
    metrics = {
        "cpu": pd.Series(...),  # Add sample data
        "memory": pd.Series(...),
        "io": pd.Series(...)
    }
    
    animations = create_correlation_animations(metrics)
    
    # Add transitions
    enhanced = enhance_animations(
        animations,
        {
            "correlation_data": [...],  # Add correlation matrices
            "network_data": {
                "positions": [...],
                "weights": [...]
            },
            "cluster_data": [...]
        }
    )
    
    # Save enhanced animations
    output_dir = Path("correlation_animations")
    output_dir.mkdir(exist_ok=True)
    
    for name, frames in enhanced.items():
        fig = go.Figure(frames=frames)
        fig.write_html(str(output_dir / f"{name}_enhanced.html"))
