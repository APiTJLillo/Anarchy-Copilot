"""Analysis tools for profiling data."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

from .profile_visualization import ProfileVisualizer
from .prediction_profiling import PredictionProfiler

logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration for profile analysis."""
    significance_level: float = 0.05
    min_cluster_size: int = 5
    seasonality_period: int = 20
    correlation_threshold: float = 0.7
    outlier_threshold: float = 2.0
    trend_window: int = 10
    output_path: Optional[Path] = None

class ProfileAnalyzer:
    """Analyze profiling data."""
    
    def __init__(
        self,
        profiler: PredictionProfiler,
        config: AnalysisConfig
    ):
        self.profiler = profiler
        self.config = config
        self.analysis_cache: Dict[str, Any] = {}
    
    def analyze_performance_patterns(
        self,
        metric: str
    ) -> Dict[str, Any]:
        """Analyze performance patterns in profiling data."""
        stats = self.profiler.function_stats
        values = [s.get(metric, 0) for s in stats.values()]
        
        if not values:
            return {}
        
        patterns = {
            "basic_stats": self._calculate_basic_stats(values),
            "distribution": self._analyze_distribution(values),
            "trends": self._analyze_trends(values),
            "clusters": self._analyze_clusters(values),
            "anomalies": self._detect_anomalies(values),
            "correlations": self._analyze_correlations(stats, metric)
        }
        
        # Cache analysis
        self.analysis_cache[f"{metric}_patterns"] = patterns
        return patterns
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks."""
        stats = self.profiler.function_stats
        
        bottlenecks = {
            "time": self._analyze_time_bottlenecks(stats),
            "memory": self._analyze_memory_bottlenecks(stats),
            "calls": self._analyze_call_bottlenecks(stats),
            "dependencies": self._analyze_dependencies(stats),
            "hotspots": self._identify_hotspots(stats)
        }
        
        self.analysis_cache["bottlenecks"] = bottlenecks
        return bottlenecks
    
    def analyze_resource_usage(self) -> Dict[str, Any]:
        """Analyze resource usage patterns."""
        if not self.profiler.memory_snapshots:
            return {}
        
        snapshots = self.profiler.memory_snapshots
        
        usage = {
            "memory_trends": self._analyze_memory_trends(snapshots),
            "allocation_patterns": self._analyze_allocations(snapshots),
            "memory_efficiency": self._analyze_memory_efficiency(snapshots),
            "peak_analysis": self._analyze_peak_usage(snapshots),
            "resource_pressure": self._analyze_resource_pressure(snapshots)
        }
        
        self.analysis_cache["resource_usage"] = usage
        return usage
    
    def analyze_optimization_potential(self) -> Dict[str, Any]:
        """Analyze potential optimizations."""
        stats = self.profiler.function_stats
        
        optimization = {
            "opportunities": self._identify_optimization_opportunities(stats),
            "impact_analysis": self._analyze_optimization_impact(stats),
            "trade_offs": self._analyze_optimization_tradeoffs(stats),
            "recommendations": self._generate_optimization_recommendations(stats),
            "priorities": self._prioritize_optimizations(stats)
        }
        
        self.analysis_cache["optimization"] = optimization
        return optimization
    
    def _calculate_basic_stats(
        self,
        values: List[float]
    ) -> Dict[str, float]:
        """Calculate basic statistical metrics."""
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "skewness": float(stats.skew(values)),
            "kurtosis": float(stats.kurtosis(values))
        }
    
    def _analyze_distribution(
        self,
        values: List[float]
    ) -> Dict[str, Any]:
        """Analyze value distribution."""
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(values)
        ks_stat, ks_p = stats.kstest(values, "norm")
        
        # Fit distributions
        distributions = ["norm", "gamma", "lognorm", "expon"]
        fits = {}
        
        for dist_name in distributions:
            dist = getattr(stats, dist_name)
            params = dist.fit(values)
            ks_stat, p_value = stats.kstest(values, dist_name, params)
            fits[dist_name] = {
                "parameters": [float(p) for p in params],
                "p_value": float(p_value)
            }
        
        return {
            "normality": {
                "shapiro": {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p)
                },
                "ks_test": {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_p)
                }
            },
            "fits": fits,
            "best_fit": max(fits.items(), key=lambda x: x[1]["p_value"])[0]
        }
    
    def _analyze_trends(
        self,
        values: List[float]
    ) -> Dict[str, Any]:
        """Analyze temporal trends."""
        if len(values) < self.config.trend_window:
            return {}
        
        # Create time series
        ts = pd.Series(values)
        
        # Decompose trend and seasonality
        decomposition = seasonal_decompose(
            ts,
            period=min(self.config.seasonality_period, len(values) // 2),
            extrapolate_trend="freq"
        )
        
        # Mann-Kendall trend test
        mk_result = stats.kendalltau(range(len(values)), values)
        
        return {
            "trend": {
                "slope": float(np.polyfit(range(len(values)), values, 1)[0]),
                "mann_kendall": {
                    "statistic": float(mk_result.statistic),
                    "p_value": float(mk_result.pvalue)
                }
            },
            "seasonality": {
                "strength": float(np.std(decomposition.seasonal)),
                "period": self.config.seasonality_period
            },
            "decomposition": {
                "trend": decomposition.trend.tolist(),
                "seasonal": decomposition.seasonal.tolist(),
                "residual": decomposition.resid.tolist()
            }
        }
    
    def _analyze_clusters(
        self,
        values: List[float]
    ) -> Dict[str, Any]:
        """Analyze value clusters."""
        if len(values) < self.config.min_cluster_size:
            return {}
        
        # Scale data
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(np.array(values).reshape(-1, 1))
        
        # Determine optimal number of clusters
        max_clusters = min(5, len(values) // self.config.min_cluster_size)
        if max_clusters < 2:
            return {}
        
        inertias = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_values)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        optimal_k = 2 + np.argmin(np.gradient(np.gradient(inertias)))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        labels = kmeans.fit_predict(scaled_values)
        
        return {
            "n_clusters": int(optimal_k),
            "cluster_centers": [
                float(c[0]) for c in scaler.inverse_transform(kmeans.cluster_centers_)
            ],
            "cluster_sizes": [
                int(sum(labels == i)) for i in range(optimal_k)
            ],
            "cluster_labels": labels.tolist()
        }
    
    def _detect_anomalies(
        self,
        values: List[float]
    ) -> Dict[str, Any]:
        """Detect anomalies in values."""
        if not values:
            return {}
        
        # Z-score method
        z_scores = stats.zscore(values)
        z_outliers = np.abs(z_scores) > self.config.outlier_threshold
        
        # IQR method
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        iqr_outliers = (
            (values < q1 - 1.5 * iqr) |
            (values > q3 + 1.5 * iqr)
        )
        
        # Combine methods
        outliers = z_outliers | iqr_outliers
        
        return {
            "count": int(sum(outliers)),
            "indices": np.where(outliers)[0].tolist(),
            "values": [float(v) for v in np.array(values)[outliers]],
            "z_scores": z_scores[outliers].tolist(),
            "thresholds": {
                "z_score": float(self.config.outlier_threshold),
                "iqr_lower": float(q1 - 1.5 * iqr),
                "iqr_upper": float(q3 + 1.5 * iqr)
            }
        }
    
    def _analyze_correlations(
        self,
        stats: Dict[str, Dict[str, Any]],
        metric: str
    ) -> Dict[str, Any]:
        """Analyze correlations between metrics."""
        if not stats:
            return {}
        
        # Extract available metrics
        metrics = set()
        for func_stats in stats.values():
            metrics.update(func_stats.keys())
        
        metrics = list(metrics - {"name", "timestamp"})
        if not metrics:
            return {}
        
        # Create correlation matrix
        data = []
        for func_stats in stats.values():
            row = [func_stats.get(m, 0) for m in metrics]
            data.append(row)
        
        corr_matrix = np.corrcoef(np.array(data).T)
        
        # Find significant correlations
        significant = []
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics[i+1:], i+1):
                corr = corr_matrix[i, j]
                if abs(corr) >= self.config.correlation_threshold:
                    significant.append({
                        "metrics": (metric1, metric2),
                        "correlation": float(corr),
                        "strength": "strong" if abs(corr) > 0.8 else "moderate"
                    })
        
        return {
            "matrix": corr_matrix.tolist(),
            "metrics": metrics,
            "significant": significant,
            "threshold": self.config.correlation_threshold
        }
    
    def save_analysis(self):
        """Save analysis results."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            for key, analysis in self.analysis_cache.items():
                file_path = output_path / f"{key}_analysis.json"
                with open(file_path, "w") as f:
                    json.dump(analysis, f, indent=2)
            
            logger.info(f"Saved analysis to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")

def create_profile_analyzer(
    profiler: PredictionProfiler,
    output_path: Optional[Path] = None
) -> ProfileAnalyzer:
    """Create profile analyzer."""
    config = AnalysisConfig(output_path=output_path)
    return ProfileAnalyzer(profiler, config)

if __name__ == "__main__":
    # Example usage
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
    analyzer = create_profile_analyzer(
        profiler,
        output_path=Path("profile_analysis")
    )
    
    # Analyze patterns
    patterns = analyzer.analyze_performance_patterns("execution_time")
    print("Performance patterns:", json.dumps(patterns, indent=2))
    
    # Analyze bottlenecks
    bottlenecks = analyzer.analyze_bottlenecks()
    print("\nBottlenecks:", json.dumps(bottlenecks, indent=2))
    
    # Save analysis
    analyzer.save_analysis()
