#!/usr/bin/env python3
"""Distribution fitting analysis for performance metrics."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from scripts.analyze_bootstrap import BootstrapResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FittedDistribution:
    """Results of distribution fitting."""
    name: str
    params: Dict[str, float]
    aic: float
    bic: float
    ks_statistic: float
    p_value: float

@dataclass
class DistributionFitResult:
    """Results of distribution fitting analysis."""
    metric: str
    parameter_value: float
    best_fit: FittedDistribution
    all_fits: List[FittedDistribution]
    goodness_of_fit: Dict[str, float]
    qq_data: Tuple[np.ndarray, np.ndarray]
    pp_data: Tuple[np.ndarray, np.ndarray]

class DistributionAnalyzer:
    """Analyze probability distributions of metrics."""
    
    def __init__(
        self,
        history_dir: Path,
        config_file: Optional[Path] = None,
        workers: int = 4
    ):
        self.history_dir = history_dir
        self.config = self._load_config(config_file)
        self.workers = workers
        self._setup_distributions()

    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load distribution analysis configuration."""
        default_config = {
            "distributions": {
                "candidates": [
                    "norm", "lognorm", "gamma", "weibull_min",
                    "expon", "pareto", "uniform", "beta"
                ],
                "min_samples": 100,
                "significance_level": 0.05
            },
            "fitting": {
                "max_iterations": 1000,
                "optimization_method": "Nelder-Mead",
                "use_bootstrap": True,
                "bootstrap_samples": 100
            },
            "visualization": {
                "qq_plot": True,
                "pp_plot": True,
                "pdf_comparison": True,
                "histogram_bins": "auto"
            }
        }

        if config_file and config_file.exists():
            custom_config = json.loads(config_file.read_text())
            default_config.update(custom_config)

        return default_config

    def _setup_distributions(self) -> None:
        """Set up distribution objects for fitting."""
        self.distributions = {}
        for name in self.config["distributions"]["candidates"]:
            try:
                dist = getattr(stats, name)
                self.distributions[name] = dist
            except AttributeError:
                logger.warning(f"Distribution {name} not found in scipy.stats")

    def fit_distribution(
        self,
        data: np.ndarray,
        dist_name: str
    ) -> Optional[FittedDistribution]:
        """Fit a specific distribution to the data."""
        try:
            # Fit distribution
            params = self.distributions[dist_name].fit(data)
            
            # Get fitted distribution object
            dist = self.distributions[dist_name](*params)
            
            # Calculate goodness-of-fit metrics
            aic = 2 * len(params) - 2 * np.sum(dist.logpdf(data))
            bic = len(params) * np.log(len(data)) - 2 * np.sum(dist.logpdf(data))
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.kstest(data, dist_name, params)
            
            # Create parameter dictionary
            param_names = getattr(dist, 'shapes', '').split(',') + ['loc', 'scale']
            param_names = [p.strip() for p in param_names if p.strip()]
            param_dict = dict(zip(param_names, params))
            
            return FittedDistribution(
                name=dist_name,
                params=param_dict,
                aic=aic,
                bic=bic,
                ks_statistic=ks_stat,
                p_value=p_value
            )
            
        except Exception as e:
            logger.warning(f"Failed to fit {dist_name}: {e}")
            return None

    def create_qq_plot_data(
        self,
        data: np.ndarray,
        dist: FittedDistribution
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create Q-Q plot data."""
        dist_obj = self.distributions[dist.name](**dist.params)
        theoretical_quantiles = dist_obj.ppf(
            np.linspace(0.01, 0.99, len(data))
        )
        sorted_data = np.sort(data)
        return theoretical_quantiles, sorted_data

    def create_pp_plot_data(
        self,
        data: np.ndarray,
        dist: FittedDistribution
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create P-P plot data."""
        dist_obj = self.distributions[dist.name](**dist.params)
        empirical = np.arange(1, len(data) + 1) / len(data)
        theoretical = dist_obj.cdf(np.sort(data))
        return theoretical, empirical

    def analyze_distributions(
        self,
        bootstrap_result: BootstrapResult
    ) -> Dict[float, DistributionFitResult]:
        """Analyze distributions for each parameter value."""
        results = {}
        
        for param_value, distribution in bootstrap_result.distributions.items():
            data = distribution.samples
            
            if len(data) < self.config["distributions"]["min_samples"]:
                logger.warning(
                    f"Insufficient samples for parameter {param_value}"
                )
                continue
            
            # Fit all candidate distributions
            fits = []
            for dist_name in self.distributions:
                fit = self.fit_distribution(data, dist_name)
                if fit is not None:
                    fits.append(fit)
            
            if not fits:
                logger.warning(
                    f"No distributions successfully fit for parameter {param_value}"
                )
                continue
            
            # Select best fit based on AIC
            best_fit = min(fits, key=lambda x: x.aic)
            
            # Calculate additional goodness-of-fit metrics
            goodness_of_fit = {
                "r_squared": self._calculate_r_squared(data, best_fit),
                "rmse": self._calculate_rmse(data, best_fit),
                "mae": self._calculate_mae(data, best_fit)
            }
            
            # Generate Q-Q and P-P plot data
            qq_data = self.create_qq_plot_data(data, best_fit)
            pp_data = self.create_pp_plot_data(data, best_fit)
            
            results[param_value] = DistributionFitResult(
                metric=bootstrap_result.metric,
                parameter_value=param_value,
                best_fit=best_fit,
                all_fits=fits,
                goodness_of_fit=goodness_of_fit,
                qq_data=qq_data,
                pp_data=pp_data
            )
        
        return results

    def _calculate_r_squared(
        self,
        data: np.ndarray,
        fit: FittedDistribution
    ) -> float:
        """Calculate R-squared goodness of fit."""
        dist_obj = self.distributions[fit.name](**fit.params)
        expected = dist_obj.pdf(np.sort(data))
        hist, _ = np.histogram(data, bins='auto', density=True)
        ss_tot = np.sum((hist - np.mean(hist)) ** 2)
        ss_res = np.sum((hist - expected[:len(hist)]) ** 2)
        return 1 - (ss_res / ss_tot)

    def _calculate_rmse(
        self,
        data: np.ndarray,
        fit: FittedDistribution
    ) -> float:
        """Calculate Root Mean Square Error."""
        dist_obj = self.distributions[fit.name](**fit.params)
        expected = dist_obj.pdf(np.sort(data))
        hist, _ = np.histogram(data, bins='auto', density=True)
        return np.sqrt(np.mean((hist - expected[:len(hist)]) ** 2))

    def _calculate_mae(
        self,
        data: np.ndarray,
        fit: FittedDistribution
    ) -> float:
        """Calculate Mean Absolute Error."""
        dist_obj = self.distributions[fit.name](**fit.params)
        expected = dist_obj.pdf(np.sort(data))
        hist, _ = np.histogram(data, bins='auto', density=True)
        return np.mean(np.abs(hist - expected[:len(hist)]))

    def create_visualization(
        self,
        fit_result: DistributionFitResult
    ) -> go.Figure:
        """Create visualization of distribution fitting results."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Data vs Fitted Distribution",
                "Q-Q Plot",
                "P-P Plot",
                "Distribution Comparison"
            )
        )
        
        # Data vs fitted distribution
        hist_data = fit_result.best_fit.params
        x = np.linspace(
            min(hist_data.values()),
            max(hist_data.values()),
            100
        )
        dist_obj = self.distributions[fit_result.best_fit.name](
            **fit_result.best_fit.params
        )
        
        fig.add_trace(
            go.Histogram(
                x=x,
                histnorm='probability density',
                name="Data",
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=dist_obj.pdf(x),
                name=f"Fitted {fit_result.best_fit.name}",
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # Q-Q plot
        theoretical, empirical = fit_result.qq_data
        fig.add_trace(
            go.Scatter(
                x=theoretical,
                y=empirical,
                mode='markers',
                name='Q-Q Plot'
            ),
            row=1, col=2
        )
        
        # Add reference line
        min_val = min(theoretical.min(), empirical.min())
        max_val = max(theoretical.max(), empirical.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Reference Line',
                line=dict(dash='dash')
            ),
            row=1, col=2
        )
        
        # P-P plot
        theoretical, empirical = fit_result.pp_data
        fig.add_trace(
            go.Scatter(
                x=theoretical,
                y=empirical,
                mode='markers',
                name='P-P Plot'
            ),
            row=2, col=1
        )
        
        # Add reference line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Reference Line',
                line=dict(dash='dash')
            ),
            row=2, col=1
        )
        
        # Distribution comparison
        sorted_fits = sorted(
            fit_result.all_fits,
            key=lambda x: x.aic
        )
        for fit in sorted_fits[:3]:  # Show top 3 fits
            dist_obj = self.distributions[fit.name](**fit.params)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=dist_obj.pdf(x),
                    name=f"{fit.name} (AIC={fit.aic:.2f})"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text=(
                f"Distribution Analysis: {fit_result.metric} "
                f"(param={fit_result.parameter_value:.3f})"
            ),
            showlegend=True
        )
        
        return fig

    def generate_report(
        self,
        fit_results: Dict[float, DistributionFitResult]
    ) -> str:
        """Generate distribution analysis report."""
        lines = ["# Distribution Analysis Report", ""]
        
        # Summary statistics
        lines.extend([
            "## Analysis Summary",
            f"- Number of Parameter Values Analyzed: {len(fit_results)}",
            f"- Candidate Distributions: {', '.join(self.distributions.keys())}",
            ""
        ])
        
        # Results by parameter value
        for param_value, result in fit_results.items():
            lines.extend([
                f"## Parameter Value: {param_value:.3f}",
                "",
                "### Best Fit Distribution",
                f"- Distribution: {result.best_fit.name}",
                "- Parameters:",
                "  " + "\n  ".join(
                    f"{k}: {v:.3f}"
                    for k, v in result.best_fit.params.items()
                ),
                f"- AIC: {result.best_fit.aic:.3f}",
                f"- BIC: {result.best_fit.bic:.3f}",
                f"- KS Test p-value: {result.best_fit.p_value:.3f}",
                "",
                "### Goodness of Fit",
                f"- R-squared: {result.goodness_of_fit['r_squared']:.3f}",
                f"- RMSE: {result.goodness_of_fit['rmse']:.3f}",
                f"- MAE: {result.goodness_of_fit['mae']:.3f}",
                "",
                "### Alternative Fits",
                "Ranked by AIC:",
            ])
            
            sorted_fits = sorted(result.all_fits, key=lambda x: x.aic)
            for i, fit in enumerate(sorted_fits[1:4], 2):  # Show next 3 best fits
                lines.extend([
                    f"{i}. {fit.name}:",
                    f"   - AIC: {fit.aic:.3f}",
                    f"   - p-value: {fit.p_value:.3f}"
                ])
            lines.append("")
        
        # Recommendations
        lines.extend(["## Recommendations", ""])
        
        for param_value, result in fit_results.items():
            if result.best_fit.p_value < self.config["distributions"]["significance_level"]:
                lines.append(
                    f"- **Poor Fit** for parameter {param_value}: Consider "
                    "alternative distributions or non-parametric approaches"
                )
            
            if result.goodness_of_fit["r_squared"] < 0.9:
                lines.append(
                    f"- **Low R-squared** for parameter {param_value}: "
                    "Distribution may not capture all patterns"
                )
        
        # Best practices
        consistently_best = self._find_consistent_distribution(fit_results)
        if consistently_best:
            lines.extend([
                "",
                "### Best Practices",
                f"- The {consistently_best} distribution consistently provides "
                "good fits across parameter values",
                "- Consider standardizing on this distribution for future analyses"
            ])
        
        return "\n".join(lines)

    def _find_consistent_distribution(
        self,
        fit_results: Dict[float, DistributionFitResult]
    ) -> Optional[str]:
        """Find distribution that consistently performs well."""
        dist_counts = {}
        for result in fit_results.values():
            # Count how often each distribution is in top 2 by AIC
            sorted_fits = sorted(result.all_fits, key=lambda x: x.aic)[:2]
            for fit in sorted_fits:
                dist_counts[fit.name] = dist_counts.get(fit.name, 0) + 1
        
        # Check if any distribution is consistently good
        for dist_name, count in dist_counts.items():
            if count >= len(fit_results) * 0.8:  # At least 80% of the time
                return dist_name
        
        return None

def main() -> int:
    """Main entry point."""
    try:
        history_dir = Path("benchmark_results/performance_history")
        if not history_dir.exists():
            logger.error("No performance history directory found")
            return 1
        
        # Load bootstrap results
        bootstrap_dir = history_dir / "bootstrap_results"
        if not bootstrap_dir.exists():
            logger.error("No bootstrap results found")
            return 1
        
        # Process each bootstrap result file
        for result_file in bootstrap_dir.glob("*.json"):
            bootstrap_data = json.loads(result_file.read_text())
            bootstrap_result = BootstrapResult(**bootstrap_data)
            
            # Initialize analyzer
            analyzer = DistributionAnalyzer(history_dir)
            
            # Analyze distributions
            fit_results = analyzer.analyze_distributions(bootstrap_result)
            
            # Generate visualizations
            for param_value, result in fit_results.items():
                fig = analyzer.create_visualization(result)
                fig.write_html(
                    history_dir / 
                    f"distributions_{bootstrap_result.metric}_{param_value}.html"
                )
            
            # Generate report
            report = analyzer.generate_report(fit_results)
            report_file = history_dir / f"distributions_{bootstrap_result.metric}.md"
            report_file.write_text(report)
        
        logger.info(f"Distribution analysis complete. Reports written to {history_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during distribution analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
