"""Custom easing functions for animation transitions."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from scipy.interpolate import BSpline

logger = logging.getLogger(__name__)

@dataclass
class EasingConfig:
    """Configuration for easing functions."""
    resolution: int = 100
    smoothness: float = 0.5
    bounce_elasticity: float = 0.3
    spring_tension: float = 2.0
    custom_curves: Optional[Dict[str, List[Tuple[float, float]]]] = None
    output_path: Optional[Path] = None

class EasingFunctions:
    """Custom easing functions for animations."""
    
    def __init__(self, config: EasingConfig):
        self.config = config
        self.custom_curves = config.custom_curves or {}
        self.easing_cache: Dict[str, np.ndarray] = {}
    
    def linear(self, t: float) -> float:
        """Linear easing (no easing)."""
        return t
    
    def ease_in_quad(self, t: float) -> float:
        """Quadratic ease in."""
        return t * t
    
    def ease_out_quad(self, t: float) -> float:
        """Quadratic ease out."""
        return 1 - (1 - t) * (1 - t)
    
    def ease_in_out_quad(self, t: float) -> float:
        """Quadratic ease in and out."""
        return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2
    
    def ease_in_cubic(self, t: float) -> float:
        """Cubic ease in."""
        return t * t * t
    
    def ease_out_cubic(self, t: float) -> float:
        """Cubic ease out."""
        return 1 - (1 - t) ** 3
    
    def ease_in_out_cubic(self, t: float) -> float:
        """Cubic ease in and out."""
        return (4 * t * t * t if t < 0.5 
                else 1 - (-2 * t + 2) ** 3 / 2)
    
    def ease_elastic(self, t: float) -> float:
        """Elastic easing."""
        if t == 0 or t == 1:
            return t
        
        p = 0.3  # Period
        s = p / 4  # Phase shift
        
        return (
            -pow(2, 10 * t - 10) * 
            np.sin((t * 10 - 10.75) * ((2 * np.pi) / 3))
        )
    
    def ease_bounce(self, t: float) -> float:
        """Bounce easing."""
        n1 = 7.5625
        d1 = 2.75
        
        if t < 1 / d1:
            return n1 * t * t
        elif t < 2 / d1:
            t -= 1.5 / d1
            return n1 * t * t + 0.75
        elif t < 2.5 / d1:
            t -= 2.25 / d1
            return n1 * t * t + 0.9375
        else:
            t -= 2.625 / d1
            return n1 * t * t + 0.984375
    
    def ease_spring(self, t: float) -> float:
        """Spring easing."""
        tension = self.config.spring_tension
        d = 1 - t
        return (
            1 - 
            (np.cos(20 * t * np.pi * tension) * d)
        )
    
    def create_custom_curve(
        self,
        name: str,
        control_points: List[Tuple[float, float]]
    ):
        """Create custom easing curve from control points."""
        if len(control_points) < 4:
            raise ValueError("Need at least 4 control points for cubic spline")
        
        # Create B-spline curve
        x = np.array([p[0] for p in control_points])
        y = np.array([p[1] for p in control_points])
        t = np.linspace(0, 1, len(control_points))
        
        # Fit cubic B-spline
        spl = BSpline(t, np.vstack((x, y)).T, k=3)
        
        # Sample curve
        t_fine = np.linspace(0, 1, self.config.resolution)
        curve = spl(t_fine)
        
        # Normalize to ensure curve starts at (0,0) and ends at (1,1)
        curve = (curve - curve[0]) / (curve[-1] - curve[0])
        
        # Store curve
        self.custom_curves[name] = list(map(tuple, curve))
        
        if self.config.output_path:
            self._save_curve(name, curve)
    
    def get_easing_function(
        self,
        name: str
    ) -> Callable[[float], float]:
        """Get named easing function."""
        easing_functions = {
            "linear": self.linear,
            "ease-in-quad": self.ease_in_quad,
            "ease-out-quad": self.ease_out_quad,
            "ease-in-out-quad": self.ease_in_out_quad,
            "ease-in-cubic": self.ease_in_cubic,
            "ease-out-cubic": self.ease_out_cubic,
            "ease-in-out-cubic": self.ease_in_out_cubic,
            "ease-elastic": self.ease_elastic,
            "ease-bounce": self.ease_bounce,
            "ease-spring": self.ease_spring
        }
        
        if name in easing_functions:
            return easing_functions[name]
        elif name in self.custom_curves:
            return self._create_interpolation_function(name)
        else:
            raise ValueError(f"Unknown easing function: {name}")
    
    def interpolate_value(
        self,
        start: float,
        end: float,
        progress: float,
        easing: str = "linear"
    ) -> float:
        """Interpolate value using easing function."""
        easing_func = self.get_easing_function(easing)
        t = easing_func(progress)
        return start + (end - start) * t
    
    def interpolate_array(
        self,
        start: np.ndarray,
        end: np.ndarray,
        progress: float,
        easing: str = "linear"
    ) -> np.ndarray:
        """Interpolate array using easing function."""
        easing_func = self.get_easing_function(easing)
        t = easing_func(progress)
        return start + (end - start) * t
    
    def _create_interpolation_function(
        self,
        curve_name: str
    ) -> Callable[[float], float]:
        """Create interpolation function from custom curve."""
        curve = np.array(self.custom_curves[curve_name])
        x = curve[:, 0]
        y = curve[:, 1]
        
        def interpolate(t: float) -> float:
            if t <= 0:
                return 0
            if t >= 1:
                return 1
            idx = np.searchsorted(x, t) - 1
            t_local = (t - x[idx]) / (x[idx + 1] - x[idx])
            return y[idx] + t_local * (y[idx + 1] - y[idx])
        
        return interpolate
    
    def combine_easings(
        self,
        easing1: str,
        easing2: str,
        weight: float = 0.5
    ) -> Callable[[float], float]:
        """Combine two easing functions."""
        f1 = self.get_easing_function(easing1)
        f2 = self.get_easing_function(easing2)
        
        def combined(t: float) -> float:
            return (1 - weight) * f1(t) + weight * f2(t)
        
        return combined
    
    def create_keyframe_easing(
        self,
        keyframes: List[Tuple[float, str]]
    ) -> Callable[[float], float]:
        """Create easing function from keyframes."""
        if not keyframes:
            return self.linear
        
        keyframes = sorted(keyframes, key=lambda k: k[0])
        
        def keyframe_ease(t: float) -> float:
            if t <= keyframes[0][0]:
                return self.get_easing_function(keyframes[0][1])(t)
            if t >= keyframes[-1][0]:
                return self.get_easing_function(keyframes[-1][1])(t)
            
            # Find surrounding keyframes
            idx = np.searchsorted([k[0] for k in keyframes], t) - 1
            t1, ease1 = keyframes[idx]
            t2, ease2 = keyframes[idx + 1]
            
            # Interpolate between easing functions
            local_t = (t - t1) / (t2 - t1)
            weight = self.ease_in_out_quad(local_t)  # Smooth transition
            return self.combine_easings(ease1, ease2, weight)(t)
        
        return keyframe_ease
    
    def _save_curve(self, name: str, curve: np.ndarray):
        """Save custom curve to file."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            curve_file = output_path / f"{name}_curve.json"
            curve_data = {
                "name": name,
                "points": curve.tolist(),
                "resolution": self.config.resolution
            }
            
            with open(curve_file, "w") as f:
                json.dump(curve_data, f, indent=2)
            
            logger.info(f"Saved curve to {curve_file}")
            
        except Exception as e:
            logger.error(f"Failed to save curve: {e}")

def create_easing_functions(
    output_path: Optional[Path] = None
) -> EasingFunctions:
    """Create easing functions."""
    config = EasingConfig(output_path=output_path)
    return EasingFunctions(config)

if __name__ == "__main__":
    # Example usage
    easing = create_easing_functions(output_path=Path("easing_curves"))
    
    # Create custom curve
    control_points = [
        (0.0, 0.0),
        (0.2, 0.1),
        (0.4, 0.8),
        (0.8, 0.9),
        (1.0, 1.0)
    ]
    easing.create_custom_curve("custom1", control_points)
    
    # Create keyframe easing
    keyframes = [
        (0.0, "ease-in-quad"),
        (0.3, "ease-elastic"),
        (0.7, "ease-bounce"),
        (1.0, "ease-out-cubic")
    ]
    keyframe_ease = easing.create_keyframe_easing(keyframes)
    
    # Test interpolation
    start = np.array([0, 0, 0])
    end = np.array([1, 2, 3])
    
    for t in np.linspace(0, 1, 10):
        value = easing.interpolate_array(
            start, end, t, "ease-in-out-cubic"
        )
        print(f"t={t:.1f}: {value}")
