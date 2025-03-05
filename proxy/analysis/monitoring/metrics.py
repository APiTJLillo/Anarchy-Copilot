"""Core metric types and operations."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

logger = logging.getLogger(__name__)

@dataclass
class MetricValue:
    """Single metric value with metadata."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    source: str = "proxy"
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags or {},
            "source": self.source,
            "unit": self.unit
        }

@dataclass
class MetricAggregation:
    """Aggregated metric values."""
    name: str
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    std: float
    percentiles: Dict[str, float]
    start_time: datetime
    end_time: datetime
    tags: Dict[str, str] = None
    source: str = "proxy"
    
    @classmethod
    def from_values(
        cls,
        name: str,
        values: List[float],
        start_time: datetime,
        end_time: datetime,
        percentiles: List[float] = None,
        tags: Dict[str, str] = None,
        source: str = "proxy"
    ) -> 'MetricAggregation':
        """Create aggregation from list of values."""
        if not values:
            raise ValueError("Cannot aggregate empty values")
        
        percentiles = percentiles or [50, 75, 90, 95, 99]
        p_values = np.percentile(values, percentiles)
        p_dict = {f"p{p}": v for p, v in zip(percentiles, p_values)}
        
        return cls(
            name=name,
            count=len(values),
            sum=float(sum(values)),
            min=float(min(values)),
            max=float(max(values)),
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            std=float(np.std(values)),
            percentiles=p_dict,
            start_time=start_time,
            end_time=end_time,
            tags=tags,
            source=source
        )

@dataclass
class TimeseriesMetric:
    """Time series of metric values."""
    name: str
    values: List[float]
    timestamps: List[datetime]
    tags: Dict[str, str] = None
    source: str = "proxy"
    unit: str = ""
    
    def aggregate(
        self,
        window: timedelta
    ) -> List[MetricAggregation]:
        """Aggregate values into windows."""
        if not self.values:
            return []
        
        # Group values by window
        windows: Dict[datetime, List[float]] = {}
        for ts, val in zip(self.timestamps, self.values):
            window_start = ts.replace(
                microsecond=0,
                second=0,
                minute=ts.minute - (ts.minute % window.total_seconds() // 60)
            )
            if window_start not in windows:
                windows[window_start] = []
            windows[window_start].append(val)
        
        # Create aggregations for each window
        aggregations = []
        for start_time, values in sorted(windows.items()):
            end_time = start_time + window
            agg = MetricAggregation.from_values(
                name=self.name,
                values=values,
                start_time=start_time,
                end_time=end_time,
                tags=self.tags,
                source=self.source
            )
            aggregations.append(agg)
        
        return aggregations
    
    def resample(
        self,
        interval: timedelta,
        method: str = "linear"
    ) -> 'TimeseriesMetric':
        """Resample time series to new interval."""
        if not self.values:
            return self
        
        # Create uniform time grid
        start = min(self.timestamps)
        end = max(self.timestamps)
        new_times = []
        current = start
        while current <= end:
            new_times.append(current)
            current += interval
        
        # Interpolate values
        if method == "linear":
            new_values = np.interp(
                [t.timestamp() for t in new_times],
                [t.timestamp() for t in self.timestamps],
                self.values
            )
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        return TimeseriesMetric(
            name=self.name,
            values=list(new_values),
            timestamps=new_times,
            tags=self.tags,
            source=self.source,
            unit=self.unit
        )

@dataclass
class MetricStatistics:
    """Statistical analysis of metric behavior."""
    name: str
    sample_size: int
    distribution: Dict[str, Any]  # Parameters of fitted distribution
    stationarity: Dict[str, float]  # Stationarity test results
    seasonality: Dict[str, Any]  # Detected seasonal patterns
    trend: Dict[str, Any]  # Trend analysis results
    outliers: List[Dict[str, Any]]  # Detected outliers
    change_points: List[Dict[str, Any]]  # Detected change points
    forecast: Dict[str, Any]  # Forecast parameters and confidence intervals
    
    @classmethod
    def analyze(
        cls,
        metric: TimeseriesMetric,
        analysis_config: Dict[str, Any] = None
    ) -> 'MetricStatistics':
        """Perform statistical analysis of metric."""
        config = analysis_config or {}
        
        # Analyze distribution
        values = np.array(metric.values)
        dist = cls._fit_distribution(values)
        
        # Check stationarity
        station = cls._test_stationarity(values)
        
        # Detect seasonality
        season = cls._detect_seasonality(
            values,
            [t.timestamp() for t in metric.timestamps]
        )
        
        # Analyze trend
        trend = cls._analyze_trend(values)
        
        # Detect outliers
        outliers = cls._detect_outliers(values)
        
        # Find change points
        changes = cls._detect_changes(values)
        
        # Generate forecast
        forecast = cls._generate_forecast(values)
        
        return cls(
            name=metric.name,
            sample_size=len(values),
            distribution=dist,
            stationarity=station,
            seasonality=season,
            trend=trend,
            outliers=outliers,
            change_points=changes,
            forecast=forecast
        )
    
    @staticmethod
    def _fit_distribution(values: np.ndarray) -> Dict[str, Any]:
        """Fit statistical distribution to values."""
        try:
            # Try common distributions
            distributions = [
                "norm", "gamma", "exponential", "lognorm"
            ]
            best_fit = None
            best_kstest = float("inf")
            
            for dist_name in distributions:
                # Fit distribution
                dist = getattr(stats, dist_name)
                params = dist.fit(values)
                
                # Test goodness of fit
                ks_stat, p_value = stats.kstest(values, dist_name, params)
                if ks_stat < best_kstest:
                    best_kstest = ks_stat
                    best_fit = {
                        "distribution": dist_name,
                        "parameters": list(params),
                        "ks_statistic": float(ks_stat),
                        "p_value": float(p_value)
                    }
            
            return best_fit or {}
            
        except Exception as e:
            logger.warning(f"Error fitting distribution: {e}")
            return {}
    
    @staticmethod
    def _test_stationarity(values: np.ndarray) -> Dict[str, float]:
        """Test for stationarity using Augmented Dickey-Fuller test."""
        try:
            result = adfuller(values)
            return {
                "adf_statistic": float(result[0]),
                "p_value": float(result[1]),
                "critical_values": {
                    str(key): float(val)
                    for key, val in result[4].items()
                }
            }
        except Exception as e:
            logger.warning(f"Error testing stationarity: {e}")
            return {}
    
    @staticmethod
    def _detect_seasonality(
        values: np.ndarray,
        timestamps: List[float]
    ) -> Dict[str, Any]:
        """Detect seasonal patterns in data."""
        try:
            # Decompose time series
            decomposition = seasonal_decompose(
                values,
                period=24,  # Assume hourly data
                extrapolate_trend=True
            )
            
            return {
                "seasonal": decomposition.seasonal.tolist(),
                "trend": decomposition.trend.tolist(),
                "resid": decomposition.resid.tolist(),
                "strength": float(np.std(decomposition.seasonal) / np.std(values))
            }
        except Exception as e:
            logger.warning(f"Error detecting seasonality: {e}")
            return {}
    
    @staticmethod
    def _analyze_trend(values: np.ndarray) -> Dict[str, Any]:
        """Analyze trend components."""
        try:
            # Simple linear trend
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            return {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "std_error": float(std_err)
            }
        except Exception as e:
            logger.warning(f"Error analyzing trend: {e}")
            return {}
    
    @staticmethod
    def _detect_outliers(values: np.ndarray) -> List[Dict[str, Any]]:
        """Detect outliers using multiple methods."""
        try:
            # Z-score method
            z_scores = np.abs(stats.zscore(values))
            z_outliers = np.where(z_scores > 3)[0]
            
            # IQR method
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            iqr_outliers = np.where(
                (values < q1 - 1.5 * iqr) |
                (values > q3 + 1.5 * iqr)
            )[0]
            
            # Combine results
            outliers = []
            for idx in set(z_outliers) | set(iqr_outliers):
                outliers.append({
                    "index": int(idx),
                    "value": float(values[idx]),
                    "z_score": float(z_scores[idx]),
                    "method": "combined"
                })
            
            return outliers
            
        except Exception as e:
            logger.warning(f"Error detecting outliers: {e}")
            return []
    
    @staticmethod
    def _detect_changes(values: np.ndarray) -> List[Dict[str, Any]]:
        """Detect change points in time series."""
        try:
            # Moving average comparison
            window = min(30, len(values) // 10)
            ma = np.convolve(values, np.ones(window)/window, mode='valid')
            diffs = np.abs(np.diff(ma))
            threshold = np.mean(diffs) + 2 * np.std(diffs)
            
            changes = []
            for idx in np.where(diffs > threshold)[0]:
                changes.append({
                    "index": int(idx + window//2),
                    "magnitude": float(diffs[idx]),
                    "direction": "increase" if diffs[idx] > 0 else "decrease"
                })
            
            return changes
            
        except Exception as e:
            logger.warning(f"Error detecting changes: {e}")
            return []
    
    @staticmethod
    def _generate_forecast(values: np.ndarray) -> Dict[str, Any]:
        """Generate simple forecast."""
        try:
            # Simple exponential smoothing
            alpha = 0.3
            forecast = []
            last_value = values[-1]
            
            for _ in range(10):  # Forecast next 10 points
                last_value = alpha * values[-1] + (1 - alpha) * last_value
                forecast.append(float(last_value))
            
            return {
                "method": "exponential_smoothing",
                "parameters": {"alpha": alpha},
                "forecast": forecast,
                "confidence_intervals": {
                    "lower": [f - 2*np.std(values) for f in forecast],
                    "upper": [f + 2*np.std(values) for f in forecast]
                }
            }
            
        except Exception as e:
            logger.warning(f"Error generating forecast: {e}")
            return {}
