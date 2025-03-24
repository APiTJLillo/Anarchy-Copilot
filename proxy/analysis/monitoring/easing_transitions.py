"""Easing functions for animation transitions."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class EasingConfig:
    """Configuration for easing functions."""
    steps: int = 100
    precision: int = 3
    overshoot: float = 1.70158
    amplitude: float = 1.0
    period: float = 0.3
    bounce_factor: float = 2.75
    elastic_period: float = 0.3
    elastic_amplitude: float = 1.0
    output_path: Optional[Path] = None

class EasingFunctions:
    """Collection of easing functions for transitions."""
    
    def __init__(
        self,
        config: EasingConfig
    ):
        self.config = config
    
    # Linear
    def linear(self, t: float) -> float:
        """Linear easing."""
        return t
    
    # Quadratic
    def ease_in_quad(self, t: float) -> float:
        """Quadratic ease in."""
        return t * t
    
    def ease_out_quad(self, t: float) -> float:
        """Quadratic ease out."""
        return -t * (t - 2)
    
    def ease_in_out_quad(self, t: float) -> float:
        """Quadratic ease in and out."""
        t *= 2
        if t < 1:
            return 0.5 * t * t
        t -= 1
        return -0.5 * (t * (t - 2) - 1)
    
    # Cubic
    def ease_in_cubic(self, t: float) -> float:
        """Cubic ease in."""
        return t * t * t
    
    def ease_out_cubic(self, t: float) -> float:
        """Cubic ease out."""
        t -= 1
        return t * t * t + 1
    
    def ease_in_out_cubic(self, t: float) -> float:
        """Cubic ease in and out."""
        t *= 2
        if t < 1:
            return 0.5 * t * t * t
        t -= 2
        return 0.5 * (t * t * t + 2)
    
    # Quartic
    def ease_in_quart(self, t: float) -> float:
        """Quartic ease in."""
        return t * t * t * t
    
    def ease_out_quart(self, t: float) -> float:
        """Quartic ease out."""
        t -= 1
        return -(t * t * t * t - 1)
    
    def ease_in_out_quart(self, t: float) -> float:
        """Quartic ease in and out."""
        t *= 2
        if t < 1:
            return 0.5 * t * t * t * t
        t -= 2
        return -0.5 * (t * t * t * t - 2)
    
    # Quintic
    def ease_in_quint(self, t: float) -> float:
        """Quintic ease in."""
        return t * t * t * t * t
    
    def ease_out_quint(self, t: float) -> float:
        """Quintic ease out."""
        t -= 1
        return t * t * t * t * t + 1
    
    def ease_in_out_quint(self, t: float) -> float:
        """Quintic ease in and out."""
        t *= 2
        if t < 1:
            return 0.5 * t * t * t * t * t
        t -= 2
        return 0.5 * (t * t * t * t * t + 2)
    
    # Sinusoidal
    def ease_in_sine(self, t: float) -> float:
        """Sinusoidal ease in."""
        return -np.cos(t * np.pi / 2) + 1
    
    def ease_out_sine(self, t: float) -> float:
        """Sinusoidal ease out."""
        return np.sin(t * np.pi / 2)
    
    def ease_in_out_sine(self, t: float) -> float:
        """Sinusoidal ease in and out."""
        return -0.5 * (np.cos(np.pi * t) - 1)
    
    # Exponential
    def ease_in_expo(self, t: float) -> float:
        """Exponential ease in."""
        return 0 if t == 0 else pow(2, 10 * (t - 1))
    
    def ease_out_expo(self, t: float) -> float:
        """Exponential ease out."""
        return 1 if t == 1 else -pow(2, -10 * t) + 1
    
    def ease_in_out_expo(self, t: float) -> float:
        """Exponential ease in and out."""
        if t == 0:
            return 0
        if t == 1:
            return 1
        t *= 2
        if t < 1:
            return 0.5 * pow(2, 10 * (t - 1))
        t -= 1
        return 0.5 * (-pow(2, -10 * t) + 2)
    
    # Circular
    def ease_in_circ(self, t: float) -> float:
        """Circular ease in."""
        return -(np.sqrt(1 - t * t) - 1)
    
    def ease_out_circ(self, t: float) -> float:
        """Circular ease out."""
        t -= 1
        return np.sqrt(1 - t * t)
    
    def ease_in_out_circ(self, t: float) -> float:
        """Circular ease in and out."""
        t *= 2
        if t < 1:
            return -0.5 * (np.sqrt(1 - t * t) - 1)
        t -= 2
        return 0.5 * (np.sqrt(1 - t * t) + 1)
    
    # Back
    def ease_in_back(self, t: float) -> float:
        """Back ease in."""
        return t * t * ((self.config.overshoot + 1) * t - self.config.overshoot)
    
    def ease_out_back(self, t: float) -> float:
        """Back ease out."""
        t -= 1
        return t * t * ((self.config.overshoot + 1) * t + self.config.overshoot) + 1
    
    def ease_in_out_back(self, t: float) -> float:
        """Back ease in and out."""
        s = self.config.overshoot * 1.525
        t *= 2
        if t < 1:
            return 0.5 * (t * t * ((s + 1) * t - s))
        t -= 2
        return 0.5 * (t * t * ((s + 1) * t + s) + 2)
    
    # Elastic
    def ease_in_elastic(self, t: float) -> float:
        """Elastic ease in."""
        if t == 0 or t == 1:
            return t
        
        t -= 1
        return -(
            self.config.elastic_amplitude *
            pow(2, 10 * t) *
            np.sin(
                (t - self.config.elastic_period / 4) *
                (2 * np.pi) / self.config.elastic_period
            )
        )
    
    def ease_out_elastic(self, t: float) -> float:
        """Elastic ease out."""
        if t == 0 or t == 1:
            return t
        
        return (
            self.config.elastic_amplitude *
            pow(2, -10 * t) *
            np.sin(
                (t - self.config.elastic_period / 4) *
                (2 * np.pi) / self.config.elastic_period
            ) + 1
        )
    
    def ease_in_out_elastic(self, t: float) -> float:
        """Elastic ease in and out."""
        if t == 0 or t == 1:
            return t
        
        t *= 2
        if t < 1:
            return -0.5 * (
                self.config.elastic_amplitude *
                pow(2, 10 * (t - 1)) *
                np.sin(
                    (t - 1 - self.config.elastic_period / 4) *
                    (2 * np.pi) / self.config.elastic_period
                )
            )
        
        return (
            self.config.elastic_amplitude *
            pow(2, -10 * (t - 1)) *
            np.sin(
                (t - 1 - self.config.elastic_period / 4) *
                (2 * np.pi) / self.config.elastic_period
            ) * 0.5 + 1
        )
    
    # Bounce
    def ease_out_bounce(self, t: float) -> float:
        """Bounce ease out."""
        if t < 1 / self.config.bounce_factor:
            return self.config.bounce_factor * t * t
        
        if t < 2 / self.config.bounce_factor:
            t -= 1.5 / self.config.bounce_factor
            return self.config.bounce_factor * t * t + 0.75
        
        if t < 2.5 / self.config.bounce_factor:
            t -= 2.25 / self.config.bounce_factor
            return self.config.bounce_factor * t * t + 0.9375
        
        t -= 2.625 / self.config.bounce_factor
        return self.config.bounce_factor * t * t + 0.984375
    
    def ease_in_bounce(self, t: float) -> float:
        """Bounce ease in."""
        return 1 - self.ease_out_bounce(1 - t)
    
    def ease_in_out_bounce(self, t: float) -> float:
        """Bounce ease in and out."""
        if t < 0.5:
            return self.ease_in_bounce(t * 2) * 0.5
        return self.ease_out_bounce(t * 2 - 1) * 0.5 + 0.5
    
    def get_easing_function(
        self,
        name: str
    ) -> Callable[[float], float]:
        """Get easing function by name."""
        functions = {
            "linear": self.linear,
            "ease-in-quad": self.ease_in_quad,
            "ease-out-quad": self.ease_out_quad,
            "ease-in-out-quad": self.ease_in_out_quad,
            "ease-in-cubic": self.ease_in_cubic,
            "ease-out-cubic": self.ease_out_cubic,
            "ease-in-out-cubic": self.ease_in_out_cubic,
            "ease-in-quart": self.ease_in_quart,
            "ease-out-quart": self.ease_out_quart,
            "ease-in-out-quart": self.ease_in_out_quart,
            "ease-in-quint": self.ease_in_quint,
            "ease-out-quint": self.ease_out_quint,
            "ease-in-out-quint": self.ease_in_out_quint,
            "ease-in-sine": self.ease_in_sine,
            "ease-out-sine": self.ease_out_sine,
            "ease-in-out-sine": self.ease_in_out_sine,
            "ease-in-expo": self.ease_in_expo,
            "ease-out-expo": self.ease_out_expo,
            "ease-in-out-expo": self.ease_in_out_expo,
            "ease-in-circ": self.ease_in_circ,
            "ease-out-circ": self.ease_out_circ,
            "ease-in-out-circ": self.ease_in_out_circ,
            "ease-in-back": self.ease_in_back,
            "ease-out-back": self.ease_out_back,
            "ease-in-out-back": self.ease_in_out_back,
            "ease-in-elastic": self.ease_in_elastic,
            "ease-out-elastic": self.ease_out_elastic,
            "ease-in-out-elastic": self.ease_in_out_elastic,
            "ease-in-bounce": self.ease_in_bounce,
            "ease-out-bounce": self.ease_out_bounce,
            "ease-in-out-bounce": self.ease_in_out_bounce
        }
        
        return functions.get(name, self.linear)
    
    def generate_easing_curve(
        self,
        name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate easing curve points."""
        easing_func = self.get_easing_function(name)
        t = np.linspace(0, 1, self.config.steps)
        y = np.array([easing_func(x) for x in t])
        return t, y
    
    def interpolate_value(
        self,
        name: str,
        start: float,
        end: float,
        progress: float
    ) -> float:
        """Interpolate value using easing function."""
        easing_func = self.get_easing_function(name)
        t = np.clip(progress, 0, 1)
        return start + (end - start) * easing_func(t)
    
    def save_easing_curves(self):
        """Save easing curves data."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            curves = {}
            for name in self.get_easing_function("linear").__code__.co_names:
                if name.startswith("ease"):
                    t, y = self.generate_easing_curve(name)
                    curves[name] = {
                        "t": t.tolist(),
                        "y": y.tolist()
                    }
            
            curves_file = output_path / "easing_curves.json"
            with open(curves_file, "w") as f:
                json.dump(curves, f, indent=2)
            
            logger.info(f"Saved easing curves to {curves_file}")
            
        except Exception as e:
            logger.error(f"Failed to save easing curves: {e}")

def create_easing_functions(
    output_path: Optional[Path] = None
) -> EasingFunctions:
    """Create easing functions."""
    config = EasingConfig(output_path=output_path)
    return EasingFunctions(config)

if __name__ == "__main__":
    # Example usage
    easing = create_easing_functions(
        output_path=Path("easing_curves")
    )
    
    # Generate and save all easing curves
    easing.save_easing_curves()
    
    # Example interpolation
    start_value = 0
    end_value = 100
    steps = 10
    
    for t in np.linspace(0, 1, steps):
        value = easing.interpolate_value(
            "ease-in-out-elastic",
            start_value,
            end_value,
            t
        )
        print(f"Progress {t:.1f}: {value:.2f}")
