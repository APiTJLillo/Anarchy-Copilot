"""Filtering capabilities for learning analysis."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

from .interactive_learning import InteractiveLearning, InteractiveConfig
from .learning_visualization import LearningVisualizer
from .optimization_learning import OptimizationLearner

logger = logging.getLogger(__name__)

@dataclass
class FilterConfig:
    """Configuration for learning filters."""
    min_confidence: float = 0.1
    max_complexity: int = 10
    window_size: int = 100
    feature_threshold: float = 0.01
    group_similar: bool = True
    enable_caching: bool = True
    output_path: Optional[Path] = None

class LearningFilter:
    """Filter and preprocess learning data."""
    
    def __init__(
        self,
        interactive: InteractiveLearning,
        config: FilterConfig
    ):
        self.interactive = interactive
        self.config = config
        self.filters: Dict[str, Callable] = {}
        self.cache: Dict[str, Any] = {}
        
        self.register_default_filters()
    
    def register_default_filters(self):
        """Register default data filters."""
        self.filters.update({
            "time_range": self.filter_by_time,
            "confidence": self.filter_by_confidence,
            "feature_importance": self.filter_by_feature_importance,
            "success_rate": self.filter_by_success_rate,
            "complexity": self.filter_by_complexity,
            "sample_count": self.filter_by_sample_count,
            "correlation": self.filter_by_correlation,
            "stability": self.filter_by_stability
        })
    
    def register_filter(
        self,
        name: str,
        func: Callable
    ):
        """Register custom data filter."""
        self.filters[name] = func
    
    def apply_filters(
        self,
        data: Dict[str, Any],
        filters: List[str],
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Apply specified filters to data."""
        if not filters:
            return data
        
        params = params or {}
        filtered_data = data.copy()
        
        for filter_name in filters:
            if filter_name in self.filters:
                cache_key = self._get_cache_key(filter_name, params)
                
                if self.config.enable_caching and cache_key in self.cache:
                    filtered_data = self.cache[cache_key]
                else:
                    try:
                        filter_func = self.filters[filter_name]
                        filtered_data = filter_func(
                            filtered_data,
                            **params.get(filter_name, {})
                        )
                        
                        if self.config.enable_caching:
                            self.cache[cache_key] = filtered_data
                            
                    except Exception as e:
                        logger.error(f"Filter {filter_name} failed: {e}")
        
        return filtered_data
    
    def filter_by_time(
        self,
        data: Dict[str, Any],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        window: Optional[int] = None
    ) -> Dict[str, Any]:
        """Filter data by time range."""
        filtered = data.copy()
        
        if "timestamp" not in filtered:
            return filtered
        
        timestamps = pd.to_datetime(filtered["timestamp"])
        
        if window is not None:
            # Use sliding window
            start_time = timestamps.max() - pd.Timedelta(window, unit="D")
            mask = timestamps >= start_time
        else:
            # Use specific time range
            mask = pd.Series(True, index=timestamps.index)
            if start_time:
                mask &= timestamps >= start_time
            if end_time:
                mask &= timestamps <= end_time
        
        return {
            key: value[mask] if isinstance(value, (pd.Series, np.ndarray)) else value
            for key, value in filtered.items()
        }
    
    def filter_by_confidence(
        self,
        data: Dict[str, Any],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Filter data by prediction confidence."""
        filtered = data.copy()
        min_conf = threshold or self.config.min_confidence
        
        if "confidence" in filtered:
            mask = filtered["confidence"] >= min_conf
            return {
                key: value[mask] if isinstance(value, (pd.Series, np.ndarray)) else value
                for key, value in filtered.items()
            }
        
        return filtered
    
    def filter_by_feature_importance(
        self,
        data: Dict[str, Any],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Filter features by importance."""
        filtered = data.copy()
        min_importance = threshold or self.config.feature_threshold
        
        if "feature_importances" in filtered:
            mask = filtered["feature_importances"] >= min_importance
            return {
                key: value[mask] if isinstance(value, (pd.Series, np.ndarray)) else value
                for key, value in filtered.items()
            }
        
        return filtered
    
    def filter_by_success_rate(
        self,
        data: Dict[str, Any],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Filter data by success rate."""
        filtered = data.copy()
        
        if "success" in filtered:
            success_rate = pd.Series(filtered["success"]).rolling(
                window=self.config.window_size,
                min_periods=1
            ).mean()
            
            mask = success_rate >= threshold
            return {
                key: value[mask] if isinstance(value, (pd.Series, np.ndarray)) else value
                for key, value in filtered.items()
            }
        
        return filtered
    
    def filter_by_complexity(
        self,
        data: Dict[str, Any],
        max_complexity: Optional[int] = None
    ) -> Dict[str, Any]:
        """Filter data by complexity."""
        filtered = data.copy()
        complexity_limit = max_complexity or self.config.max_complexity
        
        if "complexity" in filtered:
            mask = filtered["complexity"] <= complexity_limit
            return {
                key: value[mask] if isinstance(value, (pd.Series, np.ndarray)) else value
                for key, value in filtered.items()
            }
        
        return filtered
    
    def filter_by_sample_count(
        self,
        data: Dict[str, Any],
        min_samples: int = 10
    ) -> Dict[str, Any]:
        """Filter data by sample count."""
        filtered = data.copy()
        
        if isinstance(filtered.get("samples"), (list, np.ndarray)):
            mask = np.array([
                isinstance(samples, (list, np.ndarray)) and len(samples) >= min_samples
                for samples in filtered["samples"]
            ])
            return {
                key: value[mask] if isinstance(value, (pd.Series, np.ndarray)) else value
                for key, value in filtered.items()
            }
        
        return filtered
    
    def filter_by_correlation(
        self,
        data: Dict[str, Any],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Filter features by correlation."""
        filtered = data.copy()
        
        if isinstance(filtered.get("features"), pd.DataFrame):
            corr_matrix = filtered["features"].corr().abs()
            
            # Remove highly correlated features
            features_to_drop = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] >= threshold:
                        features_to_drop.add(corr_matrix.columns[j])
            
            filtered["features"] = filtered["features"].drop(
                columns=list(features_to_drop)
            )
        
        return filtered
    
    def filter_by_stability(
        self,
        data: Dict[str, Any],
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Filter data by feature stability."""
        filtered = data.copy()
        
        if isinstance(filtered.get("features"), pd.DataFrame):
            # Calculate feature stability using coefficient of variation
            cv = filtered["features"].std() / filtered["features"].mean()
            stable_features = cv[cv <= threshold].index
            
            filtered["features"] = filtered["features"][stable_features]
        
        return filtered
    
    def group_filters(
        self,
        filters: List[str]
    ) -> Dict[str, List[str]]:
        """Group similar filters together."""
        if not self.config.group_similar:
            return {"all": filters}
        
        groups = defaultdict(list)
        
        for filter_name in filters:
            if "time" in filter_name:
                groups["temporal"].append(filter_name)
            elif any(x in filter_name for x in ["confidence", "success"]):
                groups["performance"].append(filter_name)
            elif any(x in filter_name for x in ["feature", "correlation"]):
                groups["features"].append(filter_name)
            elif any(x in filter_name for x in ["complexity", "sample"]):
                groups["complexity"].append(filter_name)
            else:
                groups["other"].append(filter_name)
        
        return dict(groups)
    
    def get_filter_stats(
        self,
        data: Dict[str, Any],
        filters: List[str]
    ) -> Dict[str, Any]:
        """Get statistics about applied filters."""
        original_size = self._get_data_size(data)
        filtered_data = self.apply_filters(data, filters)
        filtered_size = self._get_data_size(filtered_data)
        
        stats = {
            "original_size": original_size,
            "filtered_size": filtered_size,
            "reduction": 1 - (filtered_size / original_size) if original_size > 0 else 0,
            "filters_applied": len(filters),
            "filter_groups": self.group_filters(filters)
        }
        
        return stats
    
    def _get_data_size(
        self,
        data: Dict[str, Any]
    ) -> int:
        """Get size of data dictionary."""
        if isinstance(data, dict):
            return sum(
                self._get_data_size(value)
                for value in data.values()
            )
        elif isinstance(data, (list, np.ndarray, pd.Series, pd.DataFrame)):
            return len(data)
        else:
            return 1
    
    def _get_cache_key(
        self,
        filter_name: str,
        params: Dict[str, Any]
    ) -> str:
        """Generate cache key for filter and parameters."""
        param_str = json.dumps(params.get(filter_name, {}), sort_keys=True)
        return f"{filter_name}:{param_str}"
    
    def save_filtered_data(
        self,
        data: Dict[str, Any],
        filters: List[str],
        output_path: Optional[Path] = None
    ):
        """Save filtered data to file."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            # Apply filters
            filtered_data = self.apply_filters(data, filters)
            
            # Save data
            output_file = path / "filtered_data.json"
            with open(output_file, "w") as f:
                json.dump(filtered_data, f, indent=2)
            
            # Save filter stats
            stats_file = path / "filter_stats.json"
            stats = self.get_filter_stats(data, filters)
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Saved filtered data to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save filtered data: {e}")

def create_learning_filter(
    interactive: InteractiveLearning,
    output_path: Optional[Path] = None
) -> LearningFilter:
    """Create learning filter."""
    config = FilterConfig(output_path=output_path)
    return LearningFilter(interactive, config)

if __name__ == "__main__":
    # Example usage
    from .interactive_learning import create_interactive_learning
    from .learning_visualization import create_learning_visualizer
    from .optimization_learning import create_optimization_learner
    from .composition_optimization import create_composition_optimizer
    from .composition_analysis import create_composition_analysis
    from .pattern_composition import create_pattern_composer
    from .scheduling_patterns import create_scheduling_pattern
    from .event_scheduler import create_event_scheduler
    from .animation_events import create_event_manager
    from .animation_controls import create_animation_controls
    from .interactive_easing import create_interactive_easing
    from .easing_visualization import create_easing_visualizer
    from .easing_transitions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    visualizer = create_easing_visualizer(easing)
    interactive = create_interactive_easing(visualizer)
    controls = create_animation_controls(interactive)
    events = create_event_manager(controls)
    scheduler = create_event_scheduler(events)
    pattern = create_scheduling_pattern(scheduler)
    composer = create_pattern_composer(pattern)
    analyzer = create_composition_analysis(composer)
    optimizer = create_composition_optimizer(analyzer)
    learner = create_optimization_learner(optimizer)
    viz = create_learning_visualizer(learner)
    interactive_learning = create_interactive_learning(viz)
    filters = create_learning_filter(
        interactive_learning,
        output_path=Path("learning_filters")
    )
    
    # Example data
    data = {
        "timestamp": pd.date_range(start="2025-01-01", periods=1000, freq="H"),
        "confidence": np.random.uniform(0, 1, 1000),
        "success": np.random.choice([True, False], 1000),
        "complexity": np.random.randint(1, 20, 1000),
        "features": pd.DataFrame(
            np.random.randn(1000, 5),
            columns=["f1", "f2", "f3", "f4", "f5"]
        )
    }
    
    # Apply filters
    filtered_data = filters.apply_filters(
        data,
        ["confidence", "complexity"],
        {
            "confidence": {"threshold": 0.7},
            "complexity": {"max_complexity": 5}
        }
    )
    
    # Get filter stats
    stats = filters.get_filter_stats(data, ["confidence", "complexity"])
    print(json.dumps(stats, indent=2))
    
    # Save filtered data
    filters.save_filtered_data(
        data,
        ["confidence", "complexity"]
    )
