"""Statistical tests for prediction evaluation."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import logging
from pathlib import Path
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

@dataclass
class StatisticalConfig:
    """Configuration for statistical tests."""
    confidence_level: float = 0.95
    min_samples: int = 30
    window_size: int = 100
    bonferroni_correction: bool = True
    permutation_tests: int = 1000
    bootstrap_samples: int = 1000
    test_metrics: List[str] = None
    output_path: Optional[Path] = None
    save_results: bool = True
    
    def __post_init__(self):
        if self.test_metrics is None:
            self.test_metrics = [
                "mse", "rmse", "r2", "explained_variance",
                "mean_error", "max_error", "error_std"
            ]

class StatisticalTester:
    """Perform statistical tests on predictions."""
    
    def __init__(self, config: StatisticalConfig):
        self.config = config
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.significance_cache: Dict[str, Dict[str, float]] = {}
        self.distribution_stats: Dict[str, Dict[str, Any]] = {}
    
    def analyze_predictions(
        self,
        actual: List[float],
        predictions: Dict[str, List[float]],
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform statistical analysis on predictions."""
        results = {}
        
        try:
            if len(actual) < self.config.min_samples:
                return {"error": "Insufficient samples"}
            
            # Basic distribution tests
            results["distribution"] = self._test_distributions(
                actual,
                predictions
            )
            
            # Model comparison tests
            results["comparisons"] = self._compare_models(
                actual,
                predictions,
                model_names
            )
            
            # Performance significance
            results["significance"] = self._test_significance(
                actual,
                predictions
            )
            
            # Autocorrelation analysis
            results["autocorrelation"] = self._test_autocorrelation(
                actual,
                predictions
            )
            
            # Store results
            self.test_results = results
            
            if self.config.save_results:
                self._save_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {"error": str(e)}
    
    def _test_distributions(
        self,
        actual: List[float],
        predictions: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Test prediction distributions."""
        results = {
            "actual": self._analyze_distribution(actual, "actual")
        }
        
        for model, preds in predictions.items():
            results[model] = self._analyze_distribution(preds, model)
            
            # Compare with actual distribution
            ks_stat, ks_pval = stats.ks_2samp(actual, preds)
            results[f"{model}_vs_actual"] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "different_distribution": bool(ks_pval < (1 - self.config.confidence_level))
            }
        
        return results
    
    def _analyze_distribution(
        self,
        values: List[float],
        name: str
    ) -> Dict[str, Any]:
        """Analyze single distribution."""
        # Basic statistics
        basic_stats = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "skewness": float(stats.skew(values)),
            "kurtosis": float(stats.kurtosis(values))
        }
        
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(values)
        dagostino_stat, dagostino_p = stats.normaltest(values)
        
        normality = {
            "shapiro": {
                "statistic": float(shapiro_stat),
                "pvalue": float(shapiro_p),
                "is_normal": bool(shapiro_p > (1 - self.config.confidence_level))
            },
            "dagostino": {
                "statistic": float(dagostino_stat),
                "pvalue": float(dagostino_p),
                "is_normal": bool(dagostino_p > (1 - self.config.confidence_level))
            }
        }
        
        # Store distribution stats
        self.distribution_stats[name] = {
            "stats": basic_stats,
            "normality": normality
        }
        
        return {
            "statistics": basic_stats,
            "normality": normality,
            "quantiles": {
                str(q): float(np.percentile(values, q))
                for q in [1, 5, 25, 50, 75, 95, 99]
            }
        }
    
    def _compare_models(
        self,
        actual: List[float],
        predictions: Dict[str, List[float]],
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare model predictions."""
        if not model_names:
            model_names = list(predictions.keys())
        
        results = {}
        
        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison = self._compare_pair(
                    actual,
                    predictions[model1],
                    predictions[model2],
                    model1,
                    model2
                )
                results[f"{model1}_vs_{model2}"] = comparison
        
        # ANOVA if normal distributions
        if all(
            self.distribution_stats[model]["normality"]["shapiro"]["is_normal"]
            for model in model_names
        ):
            f_stat, f_pval = stats.f_oneway(
                *[predictions[model] for model in model_names]
            )
            results["anova"] = {
                "f_statistic": float(f_stat),
                "pvalue": float(f_pval),
                "significant": bool(f_pval < (1 - self.config.confidence_level))
            }
        
        # Kruskal-Wallis test (non-parametric ANOVA)
        h_stat, h_pval = stats.kruskal(
            *[predictions[model] for model in model_names]
        )
        results["kruskal"] = {
            "h_statistic": float(h_stat),
            "pvalue": float(h_pval),
            "significant": bool(h_pval < (1 - self.config.confidence_level))
        }
        
        return results
    
    def _compare_pair(
        self,
        actual: List[float],
        pred1: List[float],
        pred2: List[float],
        name1: str,
        name2: str
    ) -> Dict[str, Any]:
        """Compare pair of models."""
        # Calculate errors
        errors1 = np.array(actual) - np.array(pred1)
        errors2 = np.array(actual) - np.array(pred2)
        
        # T-test on errors
        t_stat, t_pval = stats.ttest_ind(errors1, errors2)
        
        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_pval = stats.wilcoxon(errors1, errors2)
        
        # Effect size (Cohen's d)
        d = (np.mean(errors1) - np.mean(errors2)) / np.sqrt(
            (np.var(errors1) + np.var(errors2)) / 2
        )
        
        # Permutation test
        perm_p = self._permutation_test(errors1, errors2)
        
        return {
            "t_test": {
                "statistic": float(t_stat),
                "pvalue": float(t_pval),
                "significant": bool(t_pval < (1 - self.config.confidence_level))
            },
            "wilcoxon": {
                "statistic": float(w_stat),
                "pvalue": float(w_pval),
                "significant": bool(w_pval < (1 - self.config.confidence_level))
            },
            "effect_size": {
                "cohens_d": float(d),
                "interpretation": self._interpret_effect_size(d)
            },
            "permutation": {
                "pvalue": float(perm_p),
                "significant": bool(perm_p < (1 - self.config.confidence_level))
            }
        }
    
    def _test_significance(
        self,
        actual: List[float],
        predictions: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Test prediction significance."""
        results = {}
        
        for model, preds in predictions.items():
            # Calculate residuals
            residuals = np.array(actual) - np.array(preds)
            
            # Regression analysis
            X = sm.add_constant(preds)
            model_fit = sm.OLS(actual, X).fit()
            
            # Store results
            results[model] = {
                "regression": {
                    "r_squared": float(model_fit.rsquared),
                    "adj_r_squared": float(model_fit.rsquared_adj),
                    "f_pvalue": float(model_fit.f_pvalue),
                    "significant": bool(model_fit.f_pvalue < (1 - self.config.confidence_level))
                },
                "residuals": {
                    "normality": self._test_residuals(residuals),
                    "autocorrelation": self._test_residual_autocorrelation(residuals)
                }
            }
        
        # Apply multiple testing correction if needed
        if self.config.bonferroni_correction:
            pvalues = [
                results[model]["regression"]["f_pvalue"]
                for model in predictions
            ]
            significant = multipletests(
                pvalues,
                alpha=(1 - self.config.confidence_level),
                method="bonferroni"
            )[0]
            
            for model, sig in zip(predictions, significant):
                results[model]["regression"]["significant_corrected"] = bool(sig)
        
        return results
    
    def _test_residuals(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test residual normality."""
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        
        # Breusch-Pagan test for heteroscedasticity
        bp_stat, bp_p = self._breusch_pagan_test(residuals)
        
        return {
            "shapiro": {
                "statistic": float(shapiro_stat),
                "pvalue": float(shapiro_p),
                "normal": bool(shapiro_p > (1 - self.config.confidence_level))
            },
            "heteroscedasticity": {
                "statistic": float(bp_stat),
                "pvalue": float(bp_p),
                "homoscedastic": bool(bp_p > (1 - self.config.confidence_level))
            }
        }
    
    def _test_residual_autocorrelation(
        self,
        residuals: np.ndarray
    ) -> Dict[str, Any]:
        """Test residual autocorrelation."""
        # Durbin-Watson test
        dw = sm.stats.stattools.durbin_watson(residuals)
        
        # Ljung-Box test
        lb_stat, lb_p = sm.stats.diagnostic.acorr_ljungbox(
            residuals,
            lags=[10],
            return_df=False
        )
        
        return {
            "durbin_watson": {
                "statistic": float(dw),
                "no_autocorrelation": bool(1.5 < dw < 2.5)
            },
            "ljung_box": {
                "statistic": float(lb_stat[0]),
                "pvalue": float(lb_p[0]),
                "no_autocorrelation": bool(lb_p[0] > (1 - self.config.confidence_level))
            }
        }
    
    def _test_autocorrelation(
        self,
        actual: List[float],
        predictions: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Test for autocorrelation in predictions."""
        results = {}
        
        # Analyze actual values
        results["actual"] = self._analyze_autocorrelation(actual)
        
        # Analyze predictions
        for model, preds in predictions.items():
            results[model] = self._analyze_autocorrelation(preds)
        
        return results
    
    def _analyze_autocorrelation(
        self,
        values: List[float]
    ) -> Dict[str, Any]:
        """Analyze autocorrelation in time series."""
        # Calculate autocorrelation
        acf = sm.tsa.acf(values, nlags=10)
        pacf = sm.tsa.pacf(values, nlags=10)
        
        # Test stationarity
        adf = sm.tsa.stattools.adfuller(values)
        
        return {
            "autocorrelation": {
                "acf": acf.tolist(),
                "pacf": pacf.tolist(),
                "significant_lags": [
                    i for i, v in enumerate(acf)
                    if abs(v) > 2/np.sqrt(len(values))
                ]
            },
            "stationarity": {
                "adf_statistic": float(adf[0]),
                "pvalue": float(adf[1]),
                "stationary": bool(adf[1] < (1 - self.config.confidence_level))
            }
        }
    
    def _permutation_test(
        self,
        errors1: np.ndarray,
        errors2: np.ndarray,
        n_permutations: int = None
    ) -> float:
        """Perform permutation test."""
        if n_permutations is None:
            n_permutations = self.config.permutation_tests
        
        # Calculate observed difference
        obs_diff = np.mean(errors1) - np.mean(errors2)
        
        # Combine samples
        pooled = np.concatenate([errors1, errors2])
        n1 = len(errors1)
        
        # Permutation test
        count = 0
        for _ in range(n_permutations):
            # Shuffle and split
            np.random.shuffle(pooled)
            perm_diff = np.mean(pooled[:n1]) - np.mean(pooled[n1:])
            
            # Count more extreme differences
            if abs(perm_diff) >= abs(obs_diff):
                count += 1
        
        return count / n_permutations
    
    def _breusch_pagan_test(self, residuals: np.ndarray) -> Tuple[float, float]:
        """Perform Breusch-Pagan test."""
        # Fit OLS to get residuals
        X = np.ones((len(residuals), 1))
        model = sm.OLS(residuals, X).fit()
        
        # Calculate test statistic
        fitted = model.fittedvalues
        aux_ols = sm.OLS(residuals**2, sm.add_constant(fitted)).fit()
        bp_stat = aux_ols.nobs * aux_ols.rsquared
        p_value = 1 - stats.chi2.cdf(bp_stat, 1)
        
        return bp_stat, p_value
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if abs(d) < 0.2:
            return "negligible"
        elif abs(d) < 0.5:
            return "small"
        elif abs(d) < 0.8:
            return "medium"
        else:
            return "large"
    
    def plot_test_results(self) -> Dict[str, go.Figure]:
        """Create visualizations of statistical tests."""
        plots = {}
        
        # Distribution comparison plots
        for name, stats in self.distribution_stats.items():
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(
                    f"{name} - Distribution",
                    f"{name} - Q-Q Plot"
                )
            )
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    name="Distribution",
                    nbinsx=30
                ),
                row=1,
                col=1
            )
            
            # Add Q-Q plot
            theoretical_quantiles = stats.norm.ppf(
                np.linspace(0.01, 0.99, len(values))
            )
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=np.sort(values),
                    mode="markers",
                    name="Q-Q Plot"
                ),
                row=2,
                col=1
            )
            
            plots[f"{name}_distribution"] = fig
        
        # Autocorrelation plots
        for model, results in self.test_results["autocorrelation"].items():
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(
                    f"{model} - ACF",
                    f"{model} - PACF"
                )
            )
            
            # Add ACF
            fig.add_trace(
                go.Bar(
                    x=list(range(len(results["autocorrelation"]["acf"]))),
                    y=results["autocorrelation"]["acf"],
                    name="ACF"
                ),
                row=1,
                col=1
            )
            
            # Add PACF
            fig.add_trace(
                go.Bar(
                    x=list(range(len(results["autocorrelation"]["pacf"]))),
                    y=results["autocorrelation"]["pacf"],
                    name="PACF"
                ),
                row=2,
                col=1
            )
            
            plots[f"{model}_autocorrelation"] = fig
        
        return plots
    
    def _save_results(self, results: Dict[str, Any]):
        """Save test results."""
        if not self.config.output_path:
            return
        
        try:
            output_path = Path(self.config.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save results
            with open(output_path / "statistical_tests.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # Save visualizations
            plots = self.plot_test_results()
            for name, fig in plots.items():
                fig.write_html(str(output_path / f"{name}.html"))
            
            logger.info(f"Saved test results to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")

def create_tester(
    output_path: Optional[Path] = None
) -> StatisticalTester:
    """Create statistical tester."""
    config = StatisticalConfig(output_path=output_path)
    return StatisticalTester(config)

if __name__ == "__main__":
    # Example usage
    from .prediction_comparison import create_comparator
    from .performance_prediction import create_predictor
    from .prediction_interpretation import create_interpreter
    from .performance_metrics import monitor_performance
    
    # Setup components
    monitor = monitor_performance()
    predictor = create_predictor(monitor.config)
    interpreter = create_interpreter(predictor)
    comparator = create_comparator(predictor, interpreter)
    tester = create_tester(Path("statistical_tests"))
    
    # Generate test data
    np.random.seed(42)
    actual = np.random.normal(50, 10, 1000)
    predictions = {
        "model1": actual + np.random.normal(0, 5, 1000),
        "model2": actual + np.random.normal(2, 8, 1000)
    }
    
    # Run tests
    results = tester.analyze_predictions(actual, predictions)
    print(json.dumps(results, indent=2))
