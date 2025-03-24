"""Analysis metrics for easing functions."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from scipy import integrate
from scipy.signal import find_peaks

from .easing_functions import EasingFunctions, EasingConfig

logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    """Configuration for easing metrics."""
    samples: int = 1000
    smoothness_window: int = 10
    peak_threshold: float = 0.1
    velocity_tolerance: float = 0.01
    output_path: Optional[Path] = None

class EasingMetrics:
    """Calculate and analyze easing metrics."""
    
    def __init__(
        self,
        easing: EasingFunctions,
        config: MetricsConfig
    ):
        self.easing = easing
        self.config = config
    
    def calculate_metrics(
        self,
        name: str
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics for easing function."""
        t = np.linspace(0, 1, self.config.samples)
        easing_func = self.easing.get_easing_function(name)
        y = np.array([easing_func(x) for x in t])
        
        # Calculate derivatives
        velocity = np.gradient(y, t)
        acceleration = np.gradient(velocity, t)
        jerk = np.gradient(acceleration, t)
        
        metrics = {
            # Temporal metrics
            "duration": 1.0,  # Normalized time
            "peak_time": t[np.argmax(y)],
            "settling_time": self._calculate_settling_time(t, y),
            
            # Motion metrics
            "max_velocity": np.max(np.abs(velocity)),
            "avg_velocity": np.mean(np.abs(velocity)),
            "max_acceleration": np.max(np.abs(acceleration)),
            "avg_acceleration": np.mean(np.abs(acceleration)),
            "max_jerk": np.max(np.abs(jerk)),
            "avg_jerk": np.mean(np.abs(jerk)),
            
            # Quality metrics
            "smoothness": self._calculate_smoothness(jerk),
            "efficiency": self._calculate_efficiency(velocity),
            "overshoot": self._calculate_overshoot(y),
            "symmetry": self._calculate_symmetry(y),
            
            # Characteristic metrics
            "area_under_curve": integrate.simps(y, t),
            "peak_count": self._count_peaks(y),
            "zero_crossings": self._count_zero_crossings(velocity),
            "monotonicity": self._calculate_monotonicity(y)
        }
        
        # Add energy metrics
        energy_metrics = self._calculate_energy_metrics(y, velocity, acceleration)
        metrics.update(energy_metrics)
        
        return metrics
    
    def analyze_easing(
        self,
        name: str
    ) -> Dict[str, Any]:
        """Perform detailed analysis of easing function."""
        metrics = self.calculate_metrics(name)
        
        analysis = {
            "metrics": metrics,
            "characteristics": self._analyze_characteristics(metrics),
            "performance": self._analyze_performance(metrics),
            "quality": self._analyze_quality(metrics),
            "recommendations": self._generate_recommendations(metrics)
        }
        
        if self.config.output_path:
            self._save_analysis(name, analysis)
        
        return analysis
    
    def compare_easings(
        self,
        names: List[str]
    ) -> pd.DataFrame:
        """Compare multiple easing functions."""
        metrics_list = []
        
        for name in names:
            metrics = self.calculate_metrics(name)
            metrics["name"] = name
            metrics_list.append(metrics)
        
        df = pd.DataFrame(metrics_list)
        df.set_index("name", inplace=True)
        
        return df
    
    def find_similar_easings(
        self,
        name: str,
        threshold: float = 0.1
    ) -> List[Tuple[str, float]]:
        """Find similar easing functions."""
        base_metrics = self.calculate_metrics(name)
        similarities = []
        
        for other_name in self.easing.custom_curves:
            if other_name == name:
                continue
            
            other_metrics = self.calculate_metrics(other_name)
            similarity = self._calculate_similarity(
                base_metrics,
                other_metrics
            )
            
            if similarity >= 1 - threshold:
                similarities.append((other_name, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)
    
    def _calculate_settling_time(
        self,
        t: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Calculate settling time (time to reach 95% of final value)."""
        final_value = y[-1]
        threshold = 0.95 * final_value
        
        for i in range(len(t)):
            if y[i] >= threshold:
                return t[i]
        
        return 1.0
    
    def _calculate_smoothness(self, jerk: np.ndarray) -> float:
        """Calculate motion smoothness using normalized jerk."""
        return 1.0 / (1.0 + np.mean(np.abs(jerk)))
    
    def _calculate_efficiency(self, velocity: np.ndarray) -> float:
        """Calculate motion efficiency (ratio of direct to actual path)."""
        actual_distance = integrate.simps(np.abs(velocity))
        direct_distance = 1.0  # Normalized
        return direct_distance / actual_distance if actual_distance > 0 else 0.0
    
    def _calculate_overshoot(self, y: np.ndarray) -> float:
        """Calculate maximum overshoot percentage."""
        final_value = y[-1]
        max_value = np.max(y)
        
        if final_value == 0:
            return 0.0
        
        overshoot = (max_value - final_value) / final_value
        return max(0.0, overshoot)
    
    def _calculate_symmetry(self, y: np.ndarray) -> float:
        """Calculate curve symmetry around midpoint."""
        mid = len(y) // 2
        first_half = y[:mid]
        second_half = y[-1:-(mid+1):-1]  # Reverse second half
        
        diff = np.abs(first_half - second_half)
        return 1.0 - np.mean(diff)
    
    def _calculate_energy_metrics(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray
    ) -> Dict[str, float]:
        """Calculate energy-related metrics."""
        # Kinetic energy (proportional to velocity squared)
        kinetic_energy = np.mean(velocity ** 2)
        
        # Potential energy (proportional to position)
        potential_energy = np.mean(position)
        
        # Work done (force * distance)
        work = np.sum(np.abs(acceleration * velocity))
        
        # Power (rate of work)
        power = np.mean(np.abs(acceleration * velocity))
        
        return {
            "kinetic_energy": kinetic_energy,
            "potential_energy": potential_energy,
            "work": work,
            "power": power
        }
    
    def _count_peaks(self, y: np.ndarray) -> int:
        """Count number of local maxima."""
        peaks, _ = find_peaks(y, height=self.config.peak_threshold)
        return len(peaks)
    
    def _count_zero_crossings(self, signal: np.ndarray) -> int:
        """Count number of zero crossings."""
        return np.sum(np.diff(np.signbit(signal)))
    
    def _calculate_monotonicity(self, y: np.ndarray) -> float:
        """Calculate degree of monotonicity."""
        diff = np.diff(y)
        mono_violations = np.sum(diff < 0)
        return 1.0 - (mono_violations / len(diff))
    
    def _calculate_similarity(
        self,
        metrics1: Dict[str, float],
        metrics2: Dict[str, float]
    ) -> float:
        """Calculate similarity between two sets of metrics."""
        # Use subset of metrics for comparison
        compare_keys = [
            "smoothness",
            "efficiency",
            "symmetry",
            "monotonicity"
        ]
        
        diffs = []
        for key in compare_keys:
            if key in metrics1 and key in metrics2:
                diff = abs(metrics1[key] - metrics2[key])
                diffs.append(diff)
        
        if not diffs:
            return 0.0
        
        return 1.0 - np.mean(diffs)
    
    def _analyze_characteristics(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, str]:
        """Analyze easing characteristics."""
        characteristics = {}
        
        # Motion type
        if metrics["monotonicity"] > 0.95:
            characteristics["motion_type"] = "Monotonic"
        elif metrics["peak_count"] > 2:
            characteristics["motion_type"] = "Oscillatory"
        else:
            characteristics["motion_type"] = "Mixed"
        
        # Smoothness category
        if metrics["smoothness"] > 0.9:
            characteristics["smoothness"] = "Very Smooth"
        elif metrics["smoothness"] > 0.7:
            characteristics["smoothness"] = "Smooth"
        else:
            characteristics["smoothness"] = "Rough"
        
        # Efficiency category
        if metrics["efficiency"] > 0.9:
            characteristics["efficiency"] = "Highly Efficient"
        elif metrics["efficiency"] > 0.7:
            characteristics["efficiency"] = "Efficient"
        else:
            characteristics["efficiency"] = "Inefficient"
        
        return characteristics
    
    def _analyze_performance(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze performance characteristics."""
        return {
            "responsiveness": {
                "score": 1.0 - metrics["settling_time"],
                "bottlenecks": self._identify_bottlenecks(metrics)
            },
            "stability": {
                "score": 1.0 - metrics["overshoot"],
                "oscillations": metrics["peak_count"]
            },
            "efficiency": {
                "score": metrics["efficiency"],
                "energy_usage": metrics["work"]
            }
        }
    
    def _analyze_quality(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze motion quality."""
        return {
            "smoothness": {
                "score": metrics["smoothness"],
                "jerk": metrics["avg_jerk"]
            },
            "naturalness": {
                "score": self._calculate_naturalness(metrics),
                "energy_distribution": metrics["kinetic_energy"] / metrics["work"]
            },
            "precision": {
                "score": 1.0 - metrics["overshoot"],
                "control_points": metrics["zero_crossings"]
            }
        }
    
    def _generate_recommendations(
        self,
        metrics: Dict[str, float]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if metrics["smoothness"] < 0.7:
            recommendations.append(
                "Consider reducing jerk by smoothing acceleration transitions"
            )
        
        if metrics["efficiency"] < 0.7:
            recommendations.append(
                "Motion path could be more direct to improve efficiency"
            )
        
        if metrics["overshoot"] > 0.1:
            recommendations.append(
                "Reduce overshoot by adjusting acceleration profile"
            )
        
        if metrics["peak_count"] > 2:
            recommendations.append(
                "Consider reducing oscillations for more stable motion"
            )
        
        return recommendations
    
    def _identify_bottlenecks(
        self,
        metrics: Dict[str, float]
    ) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        if metrics["max_acceleration"] > 3.0:
            bottlenecks.append("High acceleration peaks")
        
        if metrics["avg_jerk"] > 1.0:
            bottlenecks.append("High average jerk")
        
        if metrics["settling_time"] > 0.8:
            bottlenecks.append("Slow settling time")
        
        return bottlenecks
    
    def _calculate_naturalness(
        self,
        metrics: Dict[str, float]
    ) -> float:
        """Calculate motion naturalness score."""
        # Combine multiple factors for naturalness
        smoothness_weight = 0.4
        efficiency_weight = 0.3
        energy_weight = 0.3
        
        naturalness = (
            smoothness_weight * metrics["smoothness"] +
            efficiency_weight * metrics["efficiency"] +
            energy_weight * (1.0 - metrics["work"] / 10.0)  # Normalize work
        )
        
        return max(0.0, min(1.0, naturalness))
    
    def _save_analysis(self, name: str, analysis: Dict[str, Any]):
        """Save analysis results."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            analysis_file = output_path / f"{name}_analysis.json"
            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Saved analysis to {analysis_file}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")

def create_easing_metrics(
    easing: EasingFunctions,
    output_path: Optional[Path] = None
) -> EasingMetrics:
    """Create easing metrics analyzer."""
    config = MetricsConfig(output_path=output_path)
    return EasingMetrics(easing, config)

if __name__ == "__main__":
    # Example usage
    from .easing_functions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    metrics = create_easing_metrics(
        easing,
        output_path=Path("easing_analysis")
    )
    
    # Analyze single easing
    analysis = metrics.analyze_easing("ease-in-out-cubic")
    print(json.dumps(analysis, indent=2))
    
    # Compare multiple easings
    comparison = metrics.compare_easings([
        "ease-in-quad",
        "ease-out-quad",
        "ease-elastic",
        "ease-bounce"
    ])
    print(comparison)
    
    # Find similar easings
    similar = metrics.find_similar_easings("ease-in-quad", threshold=0.2)
    print("\nSimilar easings:", similar)
