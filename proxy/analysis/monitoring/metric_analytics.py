"""Analytics for metric behavior and threshold patterns."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats, signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from collections import defaultdict
import json
from pathlib import Path

from .adaptive_thresholds import AdaptiveThresholds

logger = logging.getLogger(__name__)

@dataclass
class AnalyticsConfig:
    """Configuration for metric analytics."""
    min_samples: int = 100
    seasonality_test_size: int = 24 * 7  # One week
    cluster_max_k: int = 5
    correlation_threshold: float = 0.7
    change_detect_window: int = 24
    forecast_horizon: int = 24
    trend_window: int = 48

class MetricAnalytics:
    """Analytics for metric behavior."""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.analytics_cache: Dict[str, Dict[str, Any]] = {}
        self.last_update: Dict[str, datetime] = {}
    
    def analyze_metric(
        self,
        metric_key: str,
        metric_type: str,
        values: List[float],
        timestamps: List[datetime]
    ) -> Dict[str, Any]:
        """Analyze metric behavior."""
        if len(values) < self.config.min_samples:
            return {"error": "Insufficient samples"}
        
        # Convert to numpy array
        data = np.array(values)
        
        # Basic statistics
        stats = self._compute_statistics(data)
        
        # Pattern analysis
        patterns = self._analyze_patterns(data, timestamps)
        
        # Change point detection
        changes = self._detect_changes(data)
        
        # Forecasting
        forecast = self._forecast_values(data)
        
        # Cache results
        self.analytics_cache[f"{metric_key}:{metric_type}"] = {
            "stats": stats,
            "patterns": patterns,
            "changes": changes,
            "forecast": forecast,
            "timestamp": datetime.now()
        }
        
        return {
            "stats": stats,
            "patterns": patterns,
            "changes": changes,
            "forecast": forecast
        }
    
    def _compute_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute detailed statistics."""
        return {
            "basic": {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "median": float(np.median(data))
            },
            "distribution": {
                "skewness": float(stats.skew(data)),
                "kurtosis": float(stats.kurtosis(data)),
                "normality_test": self._test_normality(data),
                "percentiles": {
                    str(p): float(np.percentile(data, p))
                    for p in [1, 5, 25, 50, 75, 95, 99]
                }
            },
            "stability": {
                "coefficient_of_variation": float(np.std(data) / np.mean(data)),
                "range_ratio": float((np.max(data) - np.min(data)) / np.mean(data)),
                "iqr": float(np.percentile(data, 75) - np.percentile(data, 25))
            }
        }
    
    def _test_normality(self, data: np.ndarray) -> Dict[str, Any]:
        """Test for normal distribution."""
        statistic, p_value = stats.normaltest(data)
        shapiro_stat, shapiro_p = stats.shapiro(data)
        
        return {
            "normal_test": {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_normal": p_value > 0.05
            },
            "shapiro": {
                "statistic": float(shapiro_stat),
                "p_value": float(shapiro_p),
                "is_normal": shapiro_p > 0.05
            }
        }
    
    def _analyze_patterns(
        self,
        data: np.ndarray,
        timestamps: List[datetime]
    ) -> Dict[str, Any]:
        """Analyze patterns in the data."""
        # Test for seasonality
        seasonal = self._test_seasonality(data)
        
        # Detect cycles
        cycles = self._detect_cycles(data)
        
        # Cluster analysis
        clusters = self._cluster_analysis(data)
        
        # Temporal patterns
        temporal = self._analyze_temporal_patterns(data, timestamps)
        
        return {
            "seasonality": seasonal,
            "cycles": cycles,
            "clusters": clusters,
            "temporal": temporal
        }
    
    def _test_seasonality(self, data: np.ndarray) -> Dict[str, Any]:
        """Test for seasonal patterns."""
        if len(data) < self.config.seasonality_test_size:
            return {"error": "Insufficient data"}
        
        # Test different periods
        periods = [24, 12, 8, 6]  # hours
        results = {}
        
        for period in periods:
            if len(data) < period * 2:
                continue
            
            # Reshape data into period-length segments
            segments = len(data) // period
            matrix = data[:segments * period].reshape(segments, period)
            
            # Calculate correlation between segments
            correlations = np.corrcoef(matrix)
            mean_correlation = np.mean(correlations[np.triu_indices(segments, 1)])
            
            results[str(period)] = {
                "correlation": float(mean_correlation),
                "is_seasonal": mean_correlation > self.config.correlation_threshold
            }
        
        return results
    
    def _detect_cycles(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect cyclic patterns."""
        # Compute FFT
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data))
        
        # Find dominant frequencies
        magnitude = np.abs(fft)
        top_indices = np.argsort(magnitude)[-5:][::-1]
        
        cycles = []
        for idx in top_indices:
            if freqs[idx] > 0:  # Ignore negative frequencies
                period = 1 / freqs[idx]
                cycles.append({
                    "period": float(period),
                    "magnitude": float(magnitude[idx]),
                    "normalized_magnitude": float(magnitude[idx] / len(data))
                })
        
        return {
            "dominant_cycles": cycles,
            "has_cycles": bool(cycles)
        }
    
    def _cluster_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform cluster analysis."""
        # Prepare features
        scaler = StandardScaler()
        X = scaler.fit_transform(data.reshape(-1, 1))
        
        # Find optimal number of clusters
        best_k = 2
        best_score = float('-inf')
        scores = []
        
        for k in range(2, min(self.config.cluster_max_k + 1, len(data))):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            score = kmeans.score(X)
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        # Fit final model
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Analyze clusters
        clusters = []
        for i in range(best_k):
            cluster_data = data[labels == i]
            clusters.append({
                "size": int(len(cluster_data)),
                "mean": float(np.mean(cluster_data)),
                "std": float(np.std(cluster_data)),
                "min": float(np.min(cluster_data)),
                "max": float(np.max(cluster_data))
            })
        
        return {
            "optimal_clusters": best_k,
            "cluster_scores": [float(s) for s in scores],
            "clusters": clusters
        }
    
    def _analyze_temporal_patterns(
        self,
        data: np.ndarray,
        timestamps: List[datetime]
    ) -> Dict[str, Any]:
        """Analyze temporal patterns."""
        # Convert timestamps to hours
        hours = [ts.hour for ts in timestamps]
        
        # Hourly statistics
        hourly_stats = defaultdict(list)
        for hour, value in zip(hours, data):
            hourly_stats[hour].append(value)
        
        hourly_analysis = {
            str(hour): {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values)
            }
            for hour, values in hourly_stats.items()
        }
        
        # Day of week analysis
        days = [ts.weekday() for ts in timestamps]
        daily_stats = defaultdict(list)
        for day, value in zip(days, data):
            daily_stats[day].append(value)
        
        daily_analysis = {
            str(day): {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values)
            }
            for day, values in daily_stats.items()
        }
        
        return {
            "hourly": hourly_analysis,
            "daily": daily_analysis
        }
    
    def _detect_changes(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect changes in the data."""
        window = self.config.change_detect_window
        if len(data) < window * 2:
            return {"error": "Insufficient data"}
        
        # Moving average
        ma = np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Change points based on threshold
        threshold = np.std(data) * 2
        changes = np.where(np.abs(np.diff(ma)) > threshold)[0]
        
        # Analyze changes
        change_points = []
        for idx in changes:
            change_points.append({
                "index": int(idx + window//2),
                "magnitude": float(abs(ma[idx+1] - ma[idx])),
                "direction": "increase" if ma[idx+1] > ma[idx] else "decrease"
            })
        
        return {
            "changes": change_points,
            "num_changes": len(change_points)
        }
    
    def _forecast_values(self, data: np.ndarray) -> Dict[str, Any]:
        """Forecast future values."""
        if len(data) < self.config.trend_window:
            return {"error": "Insufficient data"}
        
        # Simple trend-based forecast
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(x[-self.config.trend_window:], data[-self.config.trend_window:])
        
        # Generate forecast
        future_x = np.arange(
            len(data),
            len(data) + self.config.forecast_horizon
        )
        forecast = slope * future_x + intercept
        
        # Calculate confidence intervals
        confidence = 0.95
        ci = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 2)
        prediction_interval = np.zeros((2, len(forecast)))
        prediction_interval[0] = forecast - ci  # Lower bound
        prediction_interval[1] = forecast + ci  # Upper bound
        
        return {
            "forecast": forecast.tolist(),
            "confidence_interval": prediction_interval.tolist(),
            "trend": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value)
            }
        }

def analyze_metric_behavior(
    thresholds: AdaptiveThresholds,
    metric_key: str,
    metric_type: str
) -> Dict[str, Any]:
    """Analyze metric behavior and patterns."""
    analytics = MetricAnalytics(AnalyticsConfig())
    
    values = thresholds.history[metric_key][metric_type]
    if not values:
        return {"error": "No data available"}
    
    # Create timestamps
    end_time = datetime.now()
    timestamps = [
        end_time - timedelta(minutes=i)
        for i in range(len(values)-1, -1, -1)
    ]
    
    return analytics.analyze_metric(
        metric_key,
        metric_type,
        values,
        timestamps
    )

if __name__ == "__main__":
    # Example usage
    from .adaptive_thresholds import create_adaptive_thresholds
    
    thresholds = create_adaptive_thresholds()
    
    # Add sample data
    for hour in range(24 * 7):
        value = 50 + 10 * np.sin(hour * np.pi / 12)  # Daily pattern
        value += 5 * np.sin(hour * np.pi / (24 * 7))  # Weekly pattern
        value += np.random.normal(0, 2)  # Random noise
        
        thresholds.add_value(
            "cpu",
            "percent",
            value,
            datetime.now() - timedelta(hours=24*7-hour)
        )
    
    # Analyze metric
    analysis = analyze_metric_behavior(thresholds, "cpu", "percent")
    print(json.dumps(analysis, indent=2))
