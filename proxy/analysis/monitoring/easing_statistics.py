"""Statistical analysis tools for easing functions."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
import statsmodels.api as sm

from .easing_metrics import EasingMetrics, MetricsConfig
from .easing_functions import EasingFunctions

logger = logging.getLogger(__name__)

@dataclass
class StatisticsConfig:
    """Configuration for statistical analysis."""
    confidence_level: float = 0.95
    min_samples: int = 100
    cluster_threshold: float = 0.75
    outlier_threshold: float = 2.0
    pca_components: int = 3
    bootstrap_iterations: int = 1000
    output_path: Optional[Path] = None

class EasingStatistics:
    """Statistical analysis of easing functions."""
    
    def __init__(
        self,
        metrics: EasingMetrics,
        config: StatisticsConfig
    ):
        self.metrics = metrics
        self.config = config
        self.analysis_cache: Dict[str, Any] = {}
    
    def analyze_distribution(
        self,
        name: str
    ) -> Dict[str, Any]:
        """Analyze statistical distribution of easing function."""
        t = np.linspace(0, 1, self.config.min_samples)
        easing_func = self.metrics.easing.get_easing_function(name)
        values = np.array([easing_func(x) for x in t])
        
        analysis = {
            "basic_stats": {
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "skewness": stats.skew(values),
                "kurtosis": stats.kurtosis(values)
            },
            "normality": {
                "shapiro": self._test_normality(values),
                "anderson": self._test_anderson(values)
            },
            "distribution_fit": self._fit_distribution(values),
            "confidence_intervals": self._calculate_confidence_intervals(values),
            "outliers": self._detect_outliers(values)
        }
        
        self.analysis_cache[name] = analysis
        return analysis
    
    def compare_distributions(
        self,
        name1: str,
        name2: str
    ) -> Dict[str, Any]:
        """Compare statistical distributions of two easing functions."""
        values1 = self._get_easing_values(name1)
        values2 = self._get_easing_values(name2)
        
        comparison = {
            "ks_test": self._kolmogorov_smirnov_test(values1, values2),
            "t_test": self._t_test(values1, values2),
            "mann_whitney": self._mann_whitney_test(values1, values2),
            "effect_size": self._calculate_effect_size(values1, values2),
            "correlation": self._calculate_correlation(values1, values2)
        }
        
        return comparison
    
    def cluster_analysis(
        self,
        names: List[str]
    ) -> Dict[str, Any]:
        """Perform cluster analysis on multiple easing functions."""
        # Collect metrics for all functions
        metrics_data = []
        for name in names:
            metrics = self.metrics.calculate_metrics(name)
            metrics_data.append([
                metrics["smoothness"],
                metrics["efficiency"],
                metrics["symmetry"],
                metrics["monotonicity"]
            ])
        
        metrics_array = np.array(metrics_data)
        
        # Perform hierarchical clustering
        linkage = hierarchy.linkage(metrics_array, method="ward")
        clusters = hierarchy.fcluster(
            linkage,
            self.config.cluster_threshold,
            criterion="distance"
        )
        
        # Perform PCA for visualization
        pca = PCA(n_components=self.config.pca_components)
        pca_result = pca.fit_transform(metrics_array)
        
        return {
            "clusters": {
                name: cluster for name, cluster in zip(names, clusters)
            },
            "linkage": linkage.tolist(),
            "pca": {
                "components": pca_result.tolist(),
                "explained_variance": pca.explained_variance_ratio_.tolist()
            },
            "silhouette": self._calculate_silhouette(metrics_array, clusters)
        }
    
    def bootstrap_analysis(
        self,
        name: str
    ) -> Dict[str, Any]:
        """Perform bootstrap analysis of easing metrics."""
        base_metrics = self.metrics.calculate_metrics(name)
        bootstrap_results = defaultdict(list)
        
        for _ in range(self.config.bootstrap_iterations):
            # Generate bootstrap sample
            t = np.linspace(0, 1, self.config.min_samples)
            easing_func = self.metrics.easing.get_easing_function(name)
            values = np.array([easing_func(x) for x in t])
            bootstrap_sample = np.random.choice(
                values,
                size=len(values),
                replace=True
            )
            
            # Calculate metrics for bootstrap sample
            metrics = {
                "smoothness": self._calculate_smoothness(bootstrap_sample),
                "efficiency": self._calculate_efficiency(bootstrap_sample),
                "symmetry": self._calculate_symmetry(bootstrap_sample)
            }
            
            for key, value in metrics.items():
                bootstrap_results[key].append(value)
        
        return {
            "base_metrics": base_metrics,
            "bootstrap_statistics": {
                key: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "confidence_interval": self._bootstrap_confidence_interval(
                        values
                    )
                }
                for key, values in bootstrap_results.items()
            }
        }
    
    def time_series_analysis(
        self,
        name: str
    ) -> Dict[str, Any]:
        """Perform time series analysis of easing function."""
        t = np.linspace(0, 1, self.config.min_samples)
        easing_func = self.metrics.easing.get_easing_function(name)
        values = np.array([easing_func(x) for x in t])
        
        return {
            "autocorrelation": self._calculate_autocorrelation(values),
            "spectral": self._spectral_analysis(values),
            "trend": self._trend_analysis(values),
            "stationarity": self._test_stationarity(values)
        }
    
    def _test_normality(
        self,
        values: np.ndarray
    ) -> Dict[str, Any]:
        """Test for normality using Shapiro-Wilk test."""
        statistic, p_value = stats.shapiro(values)
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": p_value > (1 - self.config.confidence_level)
        }
    
    def _test_anderson(
        self,
        values: np.ndarray
    ) -> Dict[str, Any]:
        """Perform Anderson-Darling test."""
        result = stats.anderson(values)
        return {
            "statistic": float(result.statistic),
            "critical_values": result.critical_values.tolist(),
            "significance_level": result.significance_level.tolist()
        }
    
    def _fit_distribution(
        self,
        values: np.ndarray
    ) -> Dict[str, Any]:
        """Fit probability distributions to data."""
        distributions = ["norm", "beta", "gamma"]
        fits = {}
        
        for dist_name in distributions:
            dist = getattr(stats, dist_name)
            params = dist.fit(values)
            
            # Calculate goodness of fit
            ks_statistic, p_value = stats.kstest(
                values,
                dist_name,
                params
            )
            
            fits[dist_name] = {
                "parameters": [float(p) for p in params],
                "ks_statistic": float(ks_statistic),
                "p_value": float(p_value)
            }
        
        return fits
    
    def _calculate_confidence_intervals(
        self,
        values: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate confidence intervals."""
        mean = np.mean(values)
        std_error = stats.sem(values)
        ci = stats.t.interval(
            self.config.confidence_level,
            len(values) - 1,
            loc=mean,
            scale=std_error
        )
        
        return {
            "mean": float(mean),
            "std_error": float(std_error),
            "lower": float(ci[0]),
            "upper": float(ci[1])
        }
    
    def _detect_outliers(
        self,
        values: np.ndarray
    ) -> Dict[str, Any]:
        """Detect statistical outliers."""
        z_scores = stats.zscore(values)
        mad = stats.median_abs_deviation(values)
        
        outliers = np.abs(z_scores) > self.config.outlier_threshold
        
        return {
            "count": int(np.sum(outliers)),
            "indices": np.where(outliers)[0].tolist(),
            "z_scores": z_scores[outliers].tolist(),
            "mad": float(mad)
        }
    
    def _kolmogorov_smirnov_test(
        self,
        values1: np.ndarray,
        values2: np.ndarray
    ) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test."""
        statistic, p_value = stats.ks_2samp(values1, values2)
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < (1 - self.config.confidence_level)
        }
    
    def _t_test(
        self,
        values1: np.ndarray,
        values2: np.ndarray
    ) -> Dict[str, Any]:
        """Perform independent t-test."""
        statistic, p_value = stats.ttest_ind(values1, values2)
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < (1 - self.config.confidence_level)
        }
    
    def _mann_whitney_test(
        self,
        values1: np.ndarray,
        values2: np.ndarray
    ) -> Dict[str, Any]:
        """Perform Mann-Whitney U test."""
        statistic, p_value = stats.mannwhitneyu(values1, values2)
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < (1 - self.config.confidence_level)
        }
    
    def _calculate_effect_size(
        self,
        values1: np.ndarray,
        values2: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate effect size metrics."""
        # Cohen's d
        pooled_std = np.sqrt(
            (np.var(values1) + np.var(values2)) / 2
        )
        cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std
        
        # Cliff's delta
        cliff_delta = stats.pointbiserialr(
            np.concatenate([
                np.zeros(len(values1)),
                np.ones(len(values2))
            ]),
            np.concatenate([values1, values2])
        )[0]
        
        return {
            "cohens_d": float(cohens_d),
            "cliffs_delta": float(cliff_delta)
        }
    
    def _calculate_correlation(
        self,
        values1: np.ndarray,
        values2: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate correlation metrics."""
        pearson = stats.pearsonr(values1, values2)
        spearman = stats.spearmanr(values1, values2)
        
        return {
            "pearson": {
                "correlation": float(pearson[0]),
                "p_value": float(pearson[1])
            },
            "spearman": {
                "correlation": float(spearman[0]),
                "p_value": float(spearman[1])
            }
        }
    
    def _calculate_silhouette(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Calculate silhouette score for clustering."""
        from sklearn.metrics import silhouette_score
        return float(silhouette_score(data, labels))
    
    def _calculate_autocorrelation(
        self,
        values: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate autocorrelation."""
        acf = sm.tsa.acf(values, nlags=10)
        pacf = sm.tsa.pacf(values, nlags=10)
        
        return {
            "acf": acf.tolist(),
            "pacf": pacf.tolist(),
            "significant_lags": np.where(
                np.abs(acf) > 2/np.sqrt(len(values))
            )[0].tolist()
        }
    
    def _spectral_analysis(
        self,
        values: np.ndarray
    ) -> Dict[str, Any]:
        """Perform spectral analysis."""
        freqs, psd = stats.welch(values)
        peaks, properties = stats.find_peaks(psd, height=np.mean(psd))
        
        return {
            "frequencies": freqs.tolist(),
            "power_spectral_density": psd.tolist(),
            "peak_frequencies": freqs[peaks].tolist(),
            "peak_powers": psd[peaks].tolist()
        }
    
    def _trend_analysis(
        self,
        values: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze trends in the data."""
        # Linear trend
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x, values
        )
        
        # Mann-Kendall trend test
        mk_result = sm.stats.diagnostic.acorr_ljungbox(values, lags=10)
        
        return {
            "linear_trend": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value)
            },
            "mann_kendall": {
                "statistic": float(mk_result[0][0]),
                "p_value": float(mk_result[1][0])
            }
        }
    
    def _test_stationarity(
        self,
        values: np.ndarray
    ) -> Dict[str, Any]:
        """Test for stationarity."""
        # Augmented Dickey-Fuller test
        adf_result = sm.tsa.stattools.adfuller(values)
        
        # KPSS test
        kpss_result = sm.tsa.stattools.kpss(values)
        
        return {
            "adf_test": {
                "statistic": float(adf_result[0]),
                "p_value": float(adf_result[1]),
                "critical_values": {
                    str(key): float(value)
                    for key, value in adf_result[4].items()
                }
            },
            "kpss_test": {
                "statistic": float(kpss_result[0]),
                "p_value": float(kpss_result[1])
            }
        }
    
    def save_analysis(
        self,
        name: str,
        analysis: Dict[str, Any]
    ):
        """Save statistical analysis results."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            analysis_file = output_path / f"{name}_stats.json"
            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Saved statistical analysis to {analysis_file}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")

def create_easing_statistics(
    metrics: EasingMetrics,
    output_path: Optional[Path] = None
) -> EasingStatistics:
    """Create easing statistics analyzer."""
    config = StatisticsConfig(output_path=output_path)
    return EasingStatistics(metrics, config)

if __name__ == "__main__":
    # Example usage
    from .easing_metrics import create_easing_metrics
    from .easing_functions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    metrics = create_easing_metrics(easing)
    stats = create_easing_statistics(
        metrics,
        output_path=Path("easing_stats")
    )
    
    # Analyze distribution
    analysis = stats.analyze_distribution("ease-in-out-cubic")
    print(json.dumps(analysis, indent=2))
    
    # Compare distributions
    comparison = stats.compare_distributions(
        "ease-in-quad",
        "ease-out-quad"
    )
    print("\nComparison:", json.dumps(comparison, indent=2))
    
    # Perform cluster analysis
    clustering = stats.cluster_analysis([
        "ease-in-quad",
        "ease-out-quad",
        "ease-in-out-quad",
        "ease-elastic",
        "ease-bounce"
    ])
    print("\nClustering:", json.dumps(clustering, indent=2))
    
    # Save analysis
    stats.save_analysis("ease-in-out-cubic", analysis)
