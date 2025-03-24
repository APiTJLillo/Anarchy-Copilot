#!/usr/bin/env python3
"""Uncertainty quantification for sensitivity analysis."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import norm
from scripts.analyze_sensitivity import SensitivityAnalyzer, SensitivityResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UncertaintyBounds:
    """Confidence bounds for sensitivity analysis."""
    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    std: np.ndarray

@dataclass
class UncertaintyResult:
    """Results of uncertainty quantification."""
    parameter: str
    bounds: Dict[str, UncertaintyBounds]
    confidence_level: float
    monte_carlo_samples: int
    critical_points: Dict[str, List[float]]
    robustness_scores: Dict[str, float]

class UncertaintyAnalyzer:
    """Analyze uncertainty in sensitivity analysis."""
    
    def __init__(
        self,
        history_dir: Path,
        config_file: Optional[Path] = None,
        sensitivity_analyzer: Optional[SensitivityAnalyzer] = None
    ):
        self.history_dir = history_dir
        self.config = self._load_config(config_file)
        self.sensitivity_analyzer = sensitivity_analyzer or SensitivityAnalyzer(history_dir)
        self.gp_models: Dict[str, GaussianProcessRegressor] = {}

    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load uncertainty analysis configuration."""
        default_config = {
            "analysis": {
                "confidence_level": 0.95,
                "monte_carlo_samples": 1000,
                "bootstrap_iterations": 100,
                "critical_threshold": 0.1
            },
            "gaussian_process": {
                "kernel_length_scale": 1.0,
                "kernel_scale": 1.0,
                "noise_level": 0.1,
                "n_restarts": 5
            },
            "visualization": {
                "show_samples": True,
                "confidence_band_alpha": 0.2,
                "highlight_critical_points": True
            }
        }

        if config_file and config_file.exists():
            custom_config = json.loads(config_file.read_text())
            default_config.update(custom_config)

        return default_config

    def fit_gp_models(
        self,
        sensitivity_result: SensitivityResult
    ) -> None:
        """Fit Gaussian Process models to sensitivity data."""
        x = np.array(sensitivity_result.values).reshape(-1, 1)
        
        for metric, values in sensitivity_result.impacts.items():
            y = np.array(values)
            
            # Define kernel
            kernel = ConstantKernel(
                self.config["gaussian_process"]["kernel_scale"]
            ) * RBF(
                self.config["gaussian_process"]["kernel_length_scale"]
            )
            
            # Create and fit GP model
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.config["gaussian_process"]["noise_level"],
                n_restarts_optimizer=self.config["gaussian_process"]["n_restarts"],
                random_state=42
            )
            
            gp.fit(x, y)
            self.gp_models[metric] = gp

    def calculate_uncertainty_bounds(
        self,
        parameter_values: np.ndarray,
        metric: str
    ) -> UncertaintyBounds:
        """Calculate uncertainty bounds using GP model."""
        x = parameter_values.reshape(-1, 1)
        gp = self.gp_models[metric]
        
        # Predict mean and std
        mean, std = gp.predict(x, return_std=True)
        
        # Calculate confidence bounds
        z = norm.ppf((1 + self.config["analysis"]["confidence_level"]) / 2)
        lower = mean - z * std
        upper = mean + z * std
        
        return UncertaintyBounds(
            lower=lower,
            upper=upper,
            mean=mean,
            std=std
        )

    def run_monte_carlo(
        self,
        parameter_values: np.ndarray,
        metric: str,
        n_samples: int
    ) -> np.ndarray:
        """Run Monte Carlo simulation using GP model."""
        x = parameter_values.reshape(-1, 1)
        gp = self.gp_models[metric]
        
        # Generate random samples from posterior
        samples = gp.sample_y(x, n_samples=n_samples, random_state=42)
        return samples

    def find_critical_points(
        self,
        parameter_values: np.ndarray,
        bounds: UncertaintyBounds,
        threshold: float
    ) -> List[float]:
        """Find critical points where uncertainty is high."""
        # Calculate regions where uncertainty band crosses threshold
        uncertainty_width = bounds.upper - bounds.lower
        critical_mask = uncertainty_width > threshold
        
        # Find critical parameter values
        critical_points = parameter_values[critical_mask]
        
        return list(critical_points)

    def calculate_robustness_score(
        self,
        samples: np.ndarray,
        target_direction: str = "increase"
    ) -> float:
        """Calculate robustness score based on Monte Carlo samples."""
        if target_direction == "increase":
            successes = np.sum(samples > 0, axis=1)
        else:
            successes = np.sum(samples < 0, axis=1)
        
        # Calculate probability of desired outcome
        robustness = np.mean(successes / samples.shape[1])
        
        return float(robustness)

    def analyze_uncertainty(
        self,
        sensitivity_result: SensitivityResult
    ) -> UncertaintyResult:
        """Perform uncertainty analysis on sensitivity results."""
        # Fit GP models
        self.fit_gp_models(sensitivity_result)
        
        parameter_values = np.array(sensitivity_result.values)
        bounds = {}
        critical_points = {}
        robustness_scores = {}
        
        for metric in sensitivity_result.impacts.keys():
            # Calculate uncertainty bounds
            bounds[metric] = self.calculate_uncertainty_bounds(
                parameter_values,
                metric
            )
            
            # Run Monte Carlo simulation
            samples = self.run_monte_carlo(
                parameter_values,
                metric,
                self.config["analysis"]["monte_carlo_samples"]
            )
            
            # Find critical points
            critical_points[metric] = self.find_critical_points(
                parameter_values,
                bounds[metric],
                self.config["analysis"]["critical_threshold"]
            )
            
            # Calculate robustness score
            robustness_scores[metric] = self.calculate_robustness_score(
                samples,
                "increase"
            )
        
        return UncertaintyResult(
            parameter=sensitivity_result.parameter,
            bounds=bounds,
            confidence_level=self.config["analysis"]["confidence_level"],
            monte_carlo_samples=self.config["analysis"]["monte_carlo_samples"],
            critical_points=critical_points,
            robustness_scores=robustness_scores
        )

    def create_visualization(
        self,
        sensitivity_result: SensitivityResult,
        uncertainty_result: UncertaintyResult
    ) -> go.Figure:
        """Create visualization of uncertainty analysis."""
        n_metrics = len(sensitivity_result.impacts)
        fig = make_subplots(
            rows=n_metrics,
            cols=1,
            subplot_titles=list(sensitivity_result.impacts.keys())
        )

        for i, (metric, values) in enumerate(sensitivity_result.impacts.items(), 1):
            # Original sensitivity curve
            fig.add_trace(
                go.Scatter(
                    x=sensitivity_result.values,
                    y=values,
                    name=f"{metric} (mean)",
                    line=dict(color='blue'),
                    showlegend=i == 1
                ),
                row=i, col=1
            )

            bounds = uncertainty_result.bounds[metric]
            
            # Confidence bands
            fig.add_trace(
                go.Scatter(
                    x=sensitivity_result.values + sensitivity_result.values[::-1],
                    y=np.concatenate([bounds.upper, bounds.lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,0,255,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"{metric} ({uncertainty_result.confidence_level:.0%} CI)",
                    showlegend=i == 1
                ),
                row=i, col=1
            )

            # Monte Carlo samples if configured
            if self.config["visualization"]["show_samples"]:
                samples = self.run_monte_carlo(
                    np.array(sensitivity_result.values),
                    metric,
                    10  # Show 10 sample paths
                )
                
                for j in range(samples.shape[1]):
                    fig.add_trace(
                        go.Scatter(
                            x=sensitivity_result.values,
                            y=samples[:, j],
                            mode='lines',
                            line=dict(
                                color='rgba(0,0,255,0.1)',
                                width=1
                            ),
                            name="MC Sample" if i == 1 and j == 0 else None,
                            showlegend=i == 1 and j == 0
                        ),
                        row=i, col=1
                    )

            # Critical points
            if self.config["visualization"]["highlight_critical_points"]:
                for point in uncertainty_result.critical_points[metric]:
                    fig.add_vline(
                        x=point,
                        line_dash="dash",
                        line_color="red",
                        opacity=0.5,
                        row=i, col=1
                    )

            # Robustness score annotation
            fig.add_annotation(
                x=0.02,
                y=0.98,
                text=f"Robustness: {uncertainty_result.robustness_scores[metric]:.2f}",
                showarrow=False,
                xref=f"x{i}",
                yref=f"y{i}",
                xanchor="left",
                yanchor="top"
            )

        fig.update_layout(
            height=300 * n_metrics,
            title_text="Uncertainty Analysis",
            showlegend=True
        )

        return fig

    def generate_report(
        self,
        sensitivity_result: SensitivityResult,
        uncertainty_result: UncertaintyResult
    ) -> str:
        """Generate uncertainty analysis report."""
        lines = ["# Uncertainty Analysis Report", ""]
        
        # Analysis parameters
        lines.extend([
            "## Analysis Parameters",
            f"- Parameter: {uncertainty_result.parameter}",
            f"- Confidence Level: {uncertainty_result.confidence_level:.1%}",
            f"- Monte Carlo Samples: {uncertainty_result.monte_carlo_samples}",
            ""
        ])
        
        # Results by metric
        lines.append("## Metric Analysis")
        
        for metric in sensitivity_result.impacts.keys():
            bounds = uncertainty_result.bounds[metric]
            robustness = uncertainty_result.robustness_scores[metric]
            
            lines.extend([
                f"### {metric}",
                "#### Uncertainty Bounds",
                f"- Mean Range: [{bounds.mean.min():.3f}, {bounds.mean.max():.3f}]",
                f"- Maximum Uncertainty: ±{bounds.std.max():.3f}",
                "",
                "#### Critical Points",
                f"- Count: {len(uncertainty_result.critical_points[metric])}",
                "- Locations: " + ", ".join(
                    f"{p:.2f}" for p in uncertainty_result.critical_points[metric]
                ),
                "",
                "#### Robustness",
                f"- Score: {robustness:.3f}",
                f"- Interpretation: {'High' if robustness > 0.8 else 'Medium' if robustness > 0.5 else 'Low'} confidence in results",
                ""
            ])
        
        # Recommendations
        lines.extend(["## Recommendations", ""])
        
        for metric, robustness in uncertainty_result.robustness_scores.items():
            if robustness < 0.5:
                lines.append(
                    f"- **High Uncertainty** in {metric}: Consider additional "
                    "analysis or data collection"
                )
            elif len(uncertainty_result.critical_points[metric]) > 0:
                lines.append(
                    f"- **Critical Regions** in {metric}: Exercise caution around "
                    f"parameter values: {', '.join(f'{p:.2f}' for p in uncertainty_result.critical_points[metric])}"
                )
        
        if all(r > 0.8 for r in uncertainty_result.robustness_scores.values()):
            lines.append(
                "- **High Confidence** in overall results: Proceed with proposed "
                "parameter changes"
            )

        return "\n".join(lines)

def main() -> int:
    """Main entry point."""
    try:
        history_dir = Path("benchmark_results/performance_history")
        if not history_dir.exists():
            logger.error("No performance history directory found")
            return 1
        
        # Load sensitivity results if available
        sensitivity_file = history_dir / "sensitivity_results.json"
        if not sensitivity_file.exists():
            logger.error("No sensitivity results found")
            return 1
        
        sensitivity_data = json.loads(sensitivity_file.read_text())
        sensitivity_results = {}
        
        for param, data in sensitivity_data.items():
            sensitivity_results[param] = SensitivityResult(**data)
        
        # Initialize analyzers
        uncertainty_analyzer = UncertaintyAnalyzer(history_dir)
        
        # Analyze uncertainty for each parameter
        for param, result in sensitivity_results.items():
            # Run uncertainty analysis
            uncertainty_result = uncertainty_analyzer.analyze_uncertainty(result)
            
            # Generate visualization
            fig = uncertainty_analyzer.create_visualization(
                result,
                uncertainty_result
            )
            fig.write_html(history_dir / f"uncertainty_{param}.html")
            
            # Generate report
            report = uncertainty_analyzer.generate_report(
                result,
                uncertainty_result
            )
            report_file = history_dir / f"uncertainty_{param}.md"
            report_file.write_text(report)
        
        logger.info(f"Uncertainty analysis complete. Reports written to {history_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during uncertainty analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
