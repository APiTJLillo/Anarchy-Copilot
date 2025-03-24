"""Adaptive thresholds for anomaly detection."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AdaptiveConfig:
    """Configuration for adaptive thresholds."""
    min_history: int = 100
    max_history: int = 10000
    sensitivity_factor: float = 1.0
    learning_rate: float = 0.1
    decay_factor: float = 0.95
    update_interval: timedelta = timedelta(hours=1)
    min_threshold: float = 1.0
    max_threshold: float = 10.0
    seasonal_adjust: bool = True
    trend_adjust: bool = True

class AdaptiveThresholds:
    """Adaptive threshold management."""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.thresholds: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.history: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.seasonal_factors: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
        self.trend_factors: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.last_update: Dict[str, datetime] = {}
    
    def get_threshold(
        self,
        metric_key: str,
        metric_type: str,
        current_value: float,
        timestamp: datetime
    ) -> float:
        """Get adaptive threshold for metric."""
        # Update thresholds if needed
        if self._should_update(metric_key):
            self._update_thresholds(metric_key)
        
        # Get base threshold
        base_threshold = self.thresholds.get(
            metric_key, {}
        ).get(metric_type, self.config.min_threshold)
        
        # Apply adjustments
        adjusted = self._adjust_threshold(
            base_threshold,
            metric_key,
            metric_type,
            current_value,
            timestamp
        )
        
        return min(
            max(adjusted, self.config.min_threshold),
            self.config.max_threshold
        )
    
    def add_value(
        self,
        metric_key: str,
        metric_type: str,
        value: float,
        timestamp: datetime
    ):
        """Add new value to history."""
        self.history[metric_key][metric_type].append(value)
        
        # Trim history if needed
        if len(self.history[metric_key][metric_type]) > self.config.max_history:
            self.history[metric_key][metric_type] = \
                self.history[metric_key][metric_type][-self.config.max_history:]
    
    def _should_update(self, metric_key: str) -> bool:
        """Check if thresholds should be updated."""
        if metric_key not in self.last_update:
            return True
        
        return (
            datetime.now() - self.last_update[metric_key] >
            self.config.update_interval
        )
    
    def _update_thresholds(self, metric_key: str):
        """Update thresholds for metric."""
        for metric_type, values in self.history[metric_key].items():
            if len(values) < self.config.min_history:
                continue
            
            # Calculate base threshold using robust statistics
            median = np.median(values)
            mad = stats.median_abs_deviation(values)
            
            threshold = (
                median +
                mad * self.config.sensitivity_factor *
                (1 + np.log10(len(values)))
            )
            
            # Apply learning rate
            if metric_type in self.thresholds[metric_key]:
                old_threshold = self.thresholds[metric_key][metric_type]
                threshold = (
                    old_threshold * (1 - self.config.learning_rate) +
                    threshold * self.config.learning_rate
                )
            
            self.thresholds[metric_key][metric_type] = threshold
        
        # Update seasonal factors
        if self.config.seasonal_adjust:
            self._update_seasonal_factors(metric_key)
        
        # Update trend factors
        if self.config.trend_adjust:
            self._update_trend_factors(metric_key)
        
        self.last_update[metric_key] = datetime.now()
    
    def _update_seasonal_factors(self, metric_key: str):
        """Update seasonal adjustment factors."""
        for metric_type, values in self.history[metric_key].items():
            if len(values) < 24 * 7:  # Need at least a week of data
                continue
            
            # Convert to hourly buckets
            hourly = np.array(values[-24*7:]).reshape(-1, 24).mean(axis=0)
            daily_mean = hourly.mean()
            
            # Calculate seasonal factors
            seasonal_factors = hourly / daily_mean
            self.seasonal_factors[metric_key][metric_type] = seasonal_factors
    
    def _update_trend_factors(self, metric_key: str):
        """Update trend adjustment factors."""
        for metric_type, values in self.history[metric_key].items():
            if len(values) < self.config.min_history:
                continue
            
            # Calculate trend using recent data
            recent = values[-min(len(values), 24*7):]
            x = np.arange(len(recent))
            slope, _ = np.polyfit(x, recent, 1)
            
            # Convert to relative trend factor
            mean_value = np.mean(recent)
            if mean_value != 0:
                trend_factor = slope / mean_value
            else:
                trend_factor = 0
            
            self.trend_factors[metric_key][metric_type] = trend_factor
    
    def _adjust_threshold(
        self,
        base: float,
        metric_key: str,
        metric_type: str,
        current_value: float,
        timestamp: datetime
    ) -> float:
        """Apply adjustments to base threshold."""
        adjusted = base
        
        # Apply seasonal adjustment
        if self.config.seasonal_adjust:
            seasonal_factors = self.seasonal_factors.get(metric_key, {}).get(metric_type)
            if seasonal_factors is not None:
                hour = timestamp.hour
                adjusted *= seasonal_factors[hour]
        
        # Apply trend adjustment
        if self.config.trend_adjust:
            trend_factor = self.trend_factors.get(metric_key, {}).get(metric_type)
            if trend_factor is not None:
                # Project trend forward
                hours_forward = (
                    datetime.now() - timestamp
                ).total_seconds() / 3600
                adjusted *= (1 + trend_factor * hours_forward)
        
        # Apply decay based on history size
        history_size = len(self.history[metric_key].get(metric_type, []))
        if history_size > 0:
            decay = self.config.decay_factor ** (
                history_size / self.config.min_history
            )
            adjusted *= decay
        
        return adjusted
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of adaptive thresholds."""
        return {
            "thresholds": dict(self.thresholds),
            "history_sizes": {
                key: {
                    metric: len(values)
                    for metric, values in metrics.items()
                }
                for key, metrics in self.history.items()
            },
            "seasonal_factors": {
                key: {
                    metric: factors.tolist()
                    for metric, factors in metrics.items()
                }
                for key, metrics in self.seasonal_factors.items()
            },
            "trend_factors": dict(self.trend_factors),
            "last_update": {
                key: ts.isoformat()
                for key, ts in self.last_update.items()
            }
        }
    
    def save_state(self, path: Path):
        """Save state to file."""
        state = self.get_state()
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: Path):
        """Load state from file."""
        with open(path) as f:
            state = json.load(f)
        
        self.thresholds = defaultdict(dict, state["thresholds"])
        
        self.seasonal_factors = defaultdict(dict)
        for key, metrics in state["seasonal_factors"].items():
            for metric, factors in metrics.items():
                self.seasonal_factors[key][metric] = np.array(factors)
        
        self.trend_factors = defaultdict(
            dict,
            state["trend_factors"]
        )
        
        self.last_update = {
            key: datetime.fromisoformat(ts)
            for key, ts in state["last_update"].items()
        }

def create_adaptive_thresholds(
    sensitivity: float = 1.0,
    learning_rate: float = 0.1,
    seasonal_adjust: bool = True
) -> AdaptiveThresholds:
    """Create adaptive thresholds with configuration."""
    config = AdaptiveConfig(
        sensitivity_factor=sensitivity,
        learning_rate=learning_rate,
        seasonal_adjust=seasonal_adjust
    )
    return AdaptiveThresholds(config)

if __name__ == "__main__":
    # Example usage
    thresholds = create_adaptive_thresholds()
    
    # Add some values
    for hour in range(24):
        value = 50 + 10 * np.sin(hour * np.pi / 12)  # Daily pattern
        thresholds.add_value(
            "cpu",
            "percent",
            value,
            datetime.now() - timedelta(hours=24-hour)
        )
    
    # Get adaptive threshold
    threshold = thresholds.get_threshold(
        "cpu",
        "percent",
        60.0,
        datetime.now()
    )
    print(f"Adaptive threshold: {threshold:.2f}")
