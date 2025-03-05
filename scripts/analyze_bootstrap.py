#!/usr/bin/env python3
"""Bootstrap analysis for uncertainty quantification."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from scripts.analyze_uncertainty import UncertaintyResult, UncertaintyBounds
from scripts.analyze_sensitivity import SensitivityResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BootstrapDistribution:
    """Bootstrapped distribution of a metric."""
    samples: np.ndarray
    mean: float
    std: float
    percentiles: Dict[int, float]
    confidence_interval: Tuple[float, float]

@dataclass
class BootstrapResult:
    """Results of bootstrap analysis."""
    parameter: str
    metric: str
    distributions: Dict[float, BootstrapDistribution]
    stability_scores: Dict[float, float]
    convergence_metrics: Dict[str, float]
    empirical_cdf: Dict[float, np.ndarray]

class BootstrapAnalyzer:
    """Analyze uncertainty using bootstrap methods."""
    
    def __init__(
        self,
        history_dir: Path,
        config_file: Optional[Path] = None,
        workers: int = 4
    ):
        self.history_dir = history_dir
        self.config = self._load_config(config_file)
        self.workers = workers

    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load bootstrap analysis configuration."""
        default_config = {
            "bootstrap": {
                "n_iterations": 1000,
                "batch_size": 100,
                "confidence_level": 0.95,
                "percentiles": [1, 5, 25, 50, 75, 95, 99]
            },
            "convergence": {
                "min_batches": 5,
                "tolerance": 0.01,
                "max_relative_error": 0.05
            },
            "visualization": {
                "show_individual_samples": False,
                "violin_plots": True,
                "kernel_density": True
            }
        }

        if config_file and config_file.exists():
            custom_config = json.loads(config_file.read_text())
            default_config.update(custom_config)

        return default_config

    def _generate_bootstrap_samples(
        self,
        data: np.ndarray,
        n_iterations: int,
        batch_size: int
    ) -> np.ndarray:
        """Generate bootstrap samples from data."""
        n_samples = len(data)
        samples = np.zeros((n_iterations, batch_size))
        
        for i in range(n_iterations):
            indices = np.random.randint(0, n_samples, size=batch_size)
            samples[i] = data[indices]
        
        return samples

    def _calculate_distribution_stats(
        self,
        samples: np.ndarray,
        confidence_level: float
    ) -> BootstrapDistribution:
        """Calculate statistical properties of bootstrapped distribution."""
        mean = np.mean(samples)
        std = np.std(samples)
        
        # Calculate percentiles
        percentiles = {
            p: np.percentile(samples, p)
            for p in self.config["bootstrap"]["percentiles"]
        }
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(samples, alpha * 100 / 2)
        ci_upper = np.percentile(samples, 100 - alpha * 100 / 2)
        
        return BootstrapDistribution(
            samples=samples,
            mean=mean,
            std=std,
            percentiles=percentiles,
            confidence_interval=(ci_lower, ci_upper)
        )

    def _check_convergence(
        self,
        batch_means: np.ndarray,
        batch_stds: np.ndarray
    ) -> Tuple[bool, Dict[str, float]]:
        """Check if bootstrap has converged."""
        if len(batch_means) < self.config["convergence"]["min_batches"]:
            return False, {}
        
        # Calculate convergence metrics
        mean_rel_error = np.abs(np.diff(batch_means)) / batch_means[:-1]
        std_rel_error = np.abs(np.diff(batch_stds)) / batch_stds[:-1]
        
        metrics = {
            "mean_relative_error": float(mean_rel_error[-1]),
            "std_relative_error": float(std_rel_error[-1]),
            "mean_stability": float(np.std(batch_means) / np.mean(batch_means)),
            "std_stability": float(np.std(batch_stds) / np.mean(batch_stds))
        }
        
        # Check convergence criteria
        converged = (
            metrics["mean_relative_error"] < self.config["convergence"]["tolerance"]
            and metrics["std_relative_error"] < self.config["convergence"]["tolerance"]
            and metrics["mean_stability"] < self.config["convergence"]["max_relative_error"]
        )
        
        return converged, metrics

    def analyze_bootstrap(
        self,
        sensitivity_result: SensitivityResult,
        uncertainty_result: UncertaintyResult,
        metric: str
    ) -> BootstrapResult:
        """Perform bootstrap analysis for a metric."""
        parameter_values = np.array(sensitivity_result.values)
        impact_values = np.array(sensitivity_result.impacts[metric])
        bounds = uncertainty_result.bounds[metric]
        
        distributions = {}
        stability_scores = {}
        convergence_metrics = {}
        empirical_cdf = {}
        
        # Analyze each parameter value
        for i, param_value in enumerate(parameter_values):
            # Generate bootstrap samples
            samples = self._generate_bootstrap_samples(
                np.array([impact_values[i]]),
                self.config["bootstrap"]["n_iterations"],
                self.config["bootstrap"]["batch_size"]
            )
            
            # Calculate distribution stats
            distribution = self._calculate_distribution_stats(
                samples.flatten(),
                self.config["bootstrap"]["confidence_level"]
            )
            
            # Track batch statistics for convergence
            batch_size = self.config["bootstrap"]["batch_size"]
            n_batches = self.config["bootstrap"]["n_iterations"] // batch_size
            batch_means = np.array([
                np.mean(samples[i*batch_size:(i+1)*batch_size])
                for i in range(n_batches)
            ])
            batch_stds = np.array([
                np.std(samples[i*batch_size:(i+1)*batch_size])
                for i in range(n_batches)
            ])
            
            # Check convergence
            converged, conv_metrics = self._check_convergence(batch_means, batch_stds)
            
            # Calculate stability score
            stability = 1.0 - (distribution.std / abs(distribution.mean))
            
            # Calculate empirical CDF
            sorted_samples = np.sort(samples.flatten())
            ecdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
            
            # Store results
            distributions[param_value] = distribution
            stability_scores[param_value] = stability
            convergence_metrics.update({
                f"{param_value}_{k}": v
                for k, v in conv_metrics.items()
            })
            empirical_cdf[param_value] = (sorted_samples, ecdf)
        
        return BootstrapResult(
            parameter=sensitivity_result.parameter,
            metric=metric,
            distributions=distributions,
            stability_scores=stability_scores,
            convergence_metrics=convergence_metrics,
            empirical_cdf=empirical_cdf
        )

    def create_visualization(
        self,
        bootstrap_result: BootstrapResult
    ) -> go.Figure:
        """Create visualization of bootstrap analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Distribution Evolution",
                "Stability Analysis",
                "Convergence Metrics",
                "Empirical CDFs"
            )
        )
        
        # Distribution evolution
        param_values = sorted(bootstrap_result.distributions.keys())
        
        if self.config["visualization"]["violin_plots"]:
            violin_data = [
                dist.samples for dist in 
                [bootstrap_result.distributions[p] for p in param_values]
            ]
            
            fig.add_trace(
                go.Violin(
                    x=np.repeat(param_values, [len(s) for s in violin_data]),
                    y=np.concatenate(violin_data),
                    name="Distribution",
                    box_visible=True,
                    meanline_visible=True
                ),
                row=1, col=1
            )
        else:
            # Show mean and confidence intervals
            means = [
                bootstrap_result.distributions[p].mean
                for p in param_values
            ]
            ci_lower = [
                bootstrap_result.distributions[p].confidence_interval[0]
                for p in param_values
            ]
            ci_upper = [
                bootstrap_result.distributions[p].confidence_interval[1]
                for p in param_values
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=param_values,
                    y=means,
                    name="Mean",
                    mode="lines+markers"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=param_values + param_values[::-1],
                    y=ci_upper + ci_lower[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ),
                row=1, col=1
            )
        
        # Stability analysis
        stability_values = list(bootstrap_result.stability_scores.values())
        
        fig.add_trace(
            go.Scatter(
                x=param_values,
                y=stability_values,
                name="Stability",
                mode="lines+markers",
                marker=dict(
                    color=np.array(stability_values),
                    colorscale="RdYlGn",
                    showscale=True
                )
            ),
            row=1, col=2
        )
        
        # Convergence metrics
        metric_types = set(
            k.split('_')[1] for k in bootstrap_result.convergence_metrics.keys()
        )
        
        for metric in metric_types:
            values = [
                bootstrap_result.convergence_metrics[f"{p}_{metric}"]
                for p in param_values
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=param_values,
                    y=values,
                    name=metric,
                    mode="lines+markers"
                ),
                row=2, col=1
            )
        
        # Empirical CDFs
        for param_value in param_values:
            samples, ecdf = bootstrap_result.empirical_cdf[param_value]
            fig.add_trace(
                go.Scatter(
                    x=samples,
                    y=ecdf,
                    name=f"Param={param_value:.2f}",
                    mode="lines"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text=f"Bootstrap Analysis: {bootstrap_result.metric}",
            showlegend=True
        )
        
        return fig

    def generate_report(
        self,
        bootstrap_result: BootstrapResult
    ) -> str:
        """Generate bootstrap analysis report."""
        lines = ["# Bootstrap Analysis Report", ""]
        
        # Analysis parameters
        lines.extend([
            "## Analysis Parameters",
            f"- Parameter: {bootstrap_result.parameter}",
            f"- Metric: {bootstrap_result.metric}",
            f"- Bootstrap Iterations: {self.config['bootstrap']['n_iterations']}",
            f"- Batch Size: {self.config['bootstrap']['batch_size']}",
            f"- Confidence Level: {self.config['bootstrap']['confidence_level']:.1%}",
            ""
        ])
        
        # Distribution analysis
        lines.extend(["## Distribution Analysis", ""])
        
        param_values = sorted(bootstrap_result.distributions.keys())
        for param in param_values:
            dist = bootstrap_result.distributions[param]
            lines.extend([
                f"### Parameter Value: {param:.3f}",
                f"- Mean: {dist.mean:.3f}",
                f"- Standard Deviation: {dist.std:.3f}",
                f"- Confidence Interval: [{dist.confidence_interval[0]:.3f}, {dist.confidence_interval[1]:.3f}]",
                "- Percentiles:",
                "  " + ", ".join(
                    f"P{p}: {v:.3f}"
                    for p, v in dist.percentiles.items()
                ),
                ""
            ])
        
        # Stability analysis
        lines.extend(["## Stability Analysis", ""])
        
        sorted_stability = sorted(
            bootstrap_result.stability_scores.items(),
            key=lambda x: x[1]
        )
        
        lines.extend([
            "Most stable parameter values:",
            "  " + ", ".join(
                f"{param:.3f} ({score:.3f})"
                for param, score in sorted_stability[-3:]
            ),
            "",
            "Least stable parameter values:",
            "  " + ", ".join(
                f"{param:.3f} ({score:.3f})"
                for param, score in sorted_stability[:3]
            ),
            ""
        ])
        
        # Convergence analysis
        lines.extend(["## Convergence Analysis", ""])
        
        metric_types = set(
            k.split('_')[1] for k in bootstrap_result.convergence_metrics.keys()
        )
        
        for metric in metric_types:
            values = [
                bootstrap_result.convergence_metrics[f"{p}_{metric}"]
                for p in param_values
            ]
            avg_value = np.mean(values)
            max_value = np.max(values)
            
            lines.extend([
                f"### {metric.replace('_', ' ').title()}",
                f"- Average: {avg_value:.3f}",
                f"- Maximum: {max_value:.3f}",
                f"- Within tolerance: {'Yes' if max_value < self.config['convergence']['tolerance'] else 'No'}",
                ""
            ])
        
        # Recommendations
        lines.extend(["## Recommendations", ""])
        
        # Based on stability
        avg_stability = np.mean(list(bootstrap_result.stability_scores.values()))
        if avg_stability < 0.5:
            lines.append(
                "- **High Variability**: Consider increasing sample size or "
                "investigating sources of instability"
            )
        
        # Based on convergence
        if any(
            v > self.config["convergence"]["tolerance"]
            for v in bootstrap_result.convergence_metrics.values()
        ):
            lines.append(
                "- **Convergence Issues**: Consider increasing bootstrap iterations "
                "or batch size"
            )
        
        # Based on confidence intervals
        max_ci_width = max(
            d.confidence_interval[1] - d.confidence_interval[0]
            for d in bootstrap_result.distributions.values()
        )
        if max_ci_width > 0.5:  # Arbitrary threshold
            lines.append(
                "- **Wide Confidence Intervals**: Results may have high uncertainty. "
                "Consider additional data collection."
            )
        
        return "\n".join(lines)

def main() -> int:
    """Main entry point."""
    try:
        history_dir = Path("benchmark_results/performance_history")
        if not history_dir.exists():
            logger.error("No performance history directory found")
            return 1
        
        # Load sensitivity and uncertainty results
        sensitivity_file = history_dir / "sensitivity_results.json"
        uncertainty_file = history_dir / "uncertainty_results.json"
        
        if not sensitivity_file.exists() or not uncertainty_file.exists():
            logger.error("Missing required results files")
            return 1
        
        sensitivity_data = json.loads(sensitivity_file.read_text())
        uncertainty_data = json.loads(uncertainty_file.read_text())
        
        # Initialize analyzer
        analyzer = BootstrapAnalyzer(history_dir)
        
        # Process each parameter and metric
        for param, sensitivity_result in sensitivity_data.items():
            uncertainty_result = uncertainty_data[param]
            
            for metric in sensitivity_result["impacts"].keys():
                # Run bootstrap analysis
                bootstrap_result = analyzer.analyze_bootstrap(
                    sensitivity_result,
                    uncertainty_result,
                    metric
                )
                
                # Generate visualization
                fig = analyzer.create_visualization(bootstrap_result)
                fig.write_html(
                    history_dir / f"bootstrap_{param}_{metric}.html"
                )
                
                # Generate report
                report = analyzer.generate_report(bootstrap_result)
                report_file = history_dir / f"bootstrap_{param}_{metric}.md"
                report_file.write_text(report)
        
        logger.info(f"Bootstrap analysis complete. Reports written to {history_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during bootstrap analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
