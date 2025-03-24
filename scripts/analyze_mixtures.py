#!/usr/bin/env python3
"""Mixture model analysis for performance metrics."""

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
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scripts.analyze_distributions import DistributionFitResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MixtureComponent:
    """Individual component in a mixture model."""
    weight: float
    params: Dict[str, float]
    distribution: str

@dataclass
class MixtureFit:
    """Results of mixture model fitting."""
    n_components: int
    components: List[MixtureComponent]
    bic: float
    aic: float
    likelihood: float
    entropy: float

@dataclass
class ClusterAssignment:
    """Cluster assignment information."""
    labels: np.ndarray
    probabilities: np.ndarray
    uncertainty: float

class MixtureAnalyzer:
    """Analyze mixture models for performance metrics."""
    
    def __init__(
        self,
        history_dir: Path,
        config_file: Optional[Path] = None
    ):
        self.history_dir = history_dir
        self.config = self._load_config(config_file)
        self.scaler = StandardScaler()

    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load mixture analysis configuration."""
        default_config = {
            "mixture": {
                "max_components": 5,
                "min_weight": 0.1,
                "convergence_tol": 1e-4,
                "n_init": 10,
                "max_iter": 1000
            },
            "model_selection": {
                "cv_folds": 5,
                "scoring": "bic",
                "refit": True
            },
            "distributions": {
                "candidates": ["normal", "lognormal", "gamma"],
                "fit_individual": True
            },
            "clustering": {
                "min_cluster_size": 50,
                "uncertainty_threshold": 0.2
            }
        }

        if config_file and config_file.exists():
            custom_config = json.loads(config_file.read_text())
            default_config.update(custom_config)

        return default_config

    def fit_gaussian_mixture(
        self,
        data: np.ndarray
    ) -> Tuple[GaussianMixture, float]:
        """Fit Gaussian mixture model with optimal components."""
        max_components = self.config["mixture"]["max_components"]
        
        # Define model grid
        param_grid = {
            "n_components": range(1, max_components + 1),
            "covariance_type": ["full", "tied", "diag", "spherical"]
        }
        
        # Initialize base model
        base_model = GaussianMixture(
            tol=self.config["mixture"]["convergence_tol"],
            n_init=self.config["mixture"]["n_init"],
            max_iter=self.config["mixture"]["max_iter"],
            random_state=42
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.config["model_selection"]["cv_folds"],
            scoring=self.config["model_selection"]["scoring"]
        )
        
        grid_search.fit(data.reshape(-1, 1))
        
        return grid_search.best_estimator_, grid_search.best_score_

    def fit_bayesian_mixture(
        self,
        data: np.ndarray
    ) -> BayesianGaussianMixture:
        """Fit Bayesian Gaussian mixture model."""
        model = BayesianGaussianMixture(
            n_components=self.config["mixture"]["max_components"],
            weight_concentration_prior=1.0/self.config["mixture"]["max_components"],
            covariance_type="full",
            tol=self.config["mixture"]["convergence_tol"],
            n_init=self.config["mixture"]["n_init"],
            max_iter=self.config["mixture"]["max_iter"],
            random_state=42
        )
        
        model.fit(data.reshape(-1, 1))
        return model

    def analyze_components(
        self,
        model: GaussianMixture,
        data: np.ndarray
    ) -> MixtureFit:
        """Analyze individual mixture components."""
        components = []
        active_components = 0
        
        for i, weight in enumerate(model.weights_):
            if weight > self.config["mixture"]["min_weight"]:
                active_components += 1
                component = MixtureComponent(
                    weight=weight,
                    params={
                        "mean": float(model.means_[i][0]),
                        "std": float(np.sqrt(model.covariances_[i][0][0]))
                    },
                    distribution="normal"
                )
                components.append(component)
        
        # Calculate fit metrics
        likelihood = model.score(data.reshape(-1, 1))
        labels = model.predict(data.reshape(-1, 1))
        probs = model.predict_proba(data.reshape(-1, 1))
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return MixtureFit(
            n_components=active_components,
            components=components,
            bic=model.bic(data.reshape(-1, 1)),
            aic=model.aic(data.reshape(-1, 1)),
            likelihood=likelihood,
            entropy=entropy
        )

    def assign_clusters(
        self,
        model: GaussianMixture,
        data: np.ndarray
    ) -> ClusterAssignment:
        """Assign data points to clusters."""
        probabilities = model.predict_proba(data.reshape(-1, 1))
        labels = model.predict(data.reshape(-1, 1))
        
        # Calculate assignment uncertainty
        uncertainty = -np.sum(
            probabilities * np.log(probabilities + 1e-10)
        ) / len(data)
        
        return ClusterAssignment(
            labels=labels,
            probabilities=probabilities,
            uncertainty=uncertainty
        )

    def characterize_clusters(
        self,
        data: np.ndarray,
        assignments: ClusterAssignment
    ) -> Dict[int, Dict[str, float]]:
        """Characterize statistical properties of clusters."""
        cluster_stats = {}
        
        for i in range(len(np.unique(assignments.labels))):
            cluster_data = data[assignments.labels == i]
            
            if len(cluster_data) >= self.config["clustering"]["min_cluster_size"]:
                stats = {
                    "size": len(cluster_data),
                    "mean": float(np.mean(cluster_data)),
                    "std": float(np.std(cluster_data)),
                    "skewness": float(stats.skew(cluster_data)),
                    "kurtosis": float(stats.kurtosis(cluster_data)),
                    "median": float(np.median(cluster_data)),
                    "iqr": float(np.percentile(cluster_data, 75) - np.percentile(cluster_data, 25))
                }
                cluster_stats[i] = stats
        
        return cluster_stats

    def create_visualization(
        self,
        data: np.ndarray,
        mixture_fit: MixtureFit,
        assignments: ClusterAssignment
    ) -> go.Figure:
        """Create visualization of mixture model analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Components and Data",
                "Component Contributions",
                "Cluster Assignments",
                "Uncertainty Analysis"
            )
        )
        
        # Components and data
        fig.add_trace(
            go.Histogram(
                x=data,
                name="Data",
                opacity=0.7,
                histnorm="probability density"
            ),
            row=1, col=1
        )
        
        x = np.linspace(data.min(), data.max(), 1000)
        for component in mixture_fit.components:
            y = stats.norm.pdf(
                x,
                loc=component.params["mean"],
                scale=component.params["std"]
            ) * component.weight
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=f"Component {len(fig.data)-1}",
                    line=dict(dash="dash")
                ),
                row=1, col=1
            )
        
        # Component contributions
        weights = [c.weight for c in mixture_fit.components]
        fig.add_trace(
            go.Bar(
                x=list(range(len(weights))),
                y=weights,
                name="Component Weights"
            ),
            row=1, col=2
        )
        
        # Cluster assignments
        for i in range(assignments.probabilities.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=data,
                    y=assignments.probabilities[:, i],
                    mode="markers",
                    name=f"Cluster {i}",
                    marker=dict(
                        size=5,
                        opacity=0.6
                    )
                ),
                row=2, col=1
            )
        
        # Uncertainty analysis
        uncertainties = -np.sum(
            assignments.probabilities *
            np.log(assignments.probabilities + 1e-10),
            axis=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=uncertainties,
                name="Assignment Uncertainty",
                nbinsx=30
            ),
            row=2, col=2
        )
        
        fig.add_vline(
            x=self.config["clustering"]["uncertainty_threshold"],
            line_dash="dash",
            line_color="red",
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"Mixture Model Analysis (n={mixture_fit.n_components})"
        )
        
        return fig

    def generate_report(
        self,
        mixture_fit: MixtureFit,
        cluster_stats: Dict[int, Dict[str, float]],
        assignments: ClusterAssignment
    ) -> str:
        """Generate mixture analysis report."""
        lines = ["# Mixture Model Analysis Report", ""]
        
        # Model summary
        lines.extend([
            "## Model Summary",
            f"- Number of Components: {mixture_fit.n_components}",
            f"- BIC: {mixture_fit.bic:.2f}",
            f"- AIC: {mixture_fit.aic:.2f}",
            f"- Log Likelihood: {mixture_fit.likelihood:.2f}",
            f"- Entropy: {mixture_fit.entropy:.2f}",
            ""
        ])
        
        # Component analysis
        lines.extend(["## Component Analysis", ""])
        
        for i, component in enumerate(mixture_fit.components):
            lines.extend([
                f"### Component {i}",
                f"- Weight: {component.weight:.3f}",
                f"- Distribution: {component.distribution}",
                "- Parameters:",
                "  " + "\n  ".join(
                    f"{k}: {v:.3f}"
                    for k, v in component.params.items()
                ),
                ""
            ])
        
        # Cluster statistics
        lines.extend(["## Cluster Statistics", ""])
        
        for cluster_id, stats in cluster_stats.items():
            lines.extend([
                f"### Cluster {cluster_id}",
                f"- Size: {stats['size']} samples",
                f"- Mean: {stats['mean']:.3f}",
                f"- Standard Deviation: {stats['std']:.3f}",
                f"- Skewness: {stats['skewness']:.3f}",
                f"- Kurtosis: {stats['kurtosis']:.3f}",
                f"- Median: {stats['median']:.3f}",
                f"- IQR: {stats['iqr']:.3f}",
                ""
            ])
        
        # Assignment analysis
        uncertain_assignments = np.sum(
            -np.sum(
                assignments.probabilities *
                np.log(assignments.probabilities + 1e-10),
                axis=1
            ) > self.config["clustering"]["uncertainty_threshold"]
        )
        
        lines.extend([
            "## Assignment Analysis",
            f"- Overall Uncertainty: {assignments.uncertainty:.3f}",
            f"- Uncertain Assignments: {uncertain_assignments} "
            f"({uncertain_assignments/len(assignments.labels)*100:.1f}%)",
            ""
        ])
        
        # Recommendations
        lines.extend(["## Recommendations", ""])
        
        if mixture_fit.n_components == 1:
            lines.append(
                "- Data appears to be unimodal. Consider using simpler "
                "single-distribution models."
            )
        
        if assignments.uncertainty > 0.5:
            lines.append(
                "- High assignment uncertainty suggests possible overlapping "
                "components. Consider reducing number of components."
            )
        
        if any(c.weight < 0.1 for c in mixture_fit.components):
            lines.append(
                "- Some components have very low weights. Consider removing them "
                "for a more parsimonious model."
            )
        
        return "\n".join(lines)

def main() -> int:
    """Main entry point."""
    try:
        history_dir = Path("benchmark_results/performance_history")
        if not history_dir.exists():
            logger.error("No performance history directory found")
            return 1
        
        # Load distribution analysis results
        dist_results_dir = history_dir / "distribution_results"
        if not dist_results_dir.exists():
            logger.error("No distribution analysis results found")
            return 1
        
        # Process each distribution result
        for result_file in dist_results_dir.glob("*.json"):
            dist_data = json.loads(result_file.read_text())
            distribution_result = DistributionFitResult(**dist_data)
            
            # Initialize analyzer
            analyzer = MixtureAnalyzer(history_dir)
            
            # Extract data
            data = np.array(distribution_result.qq_data[1])  # Use empirical data
            
            # Fit mixture model
            model, score = analyzer.fit_gaussian_mixture(data)
            
            # Analyze mixture
            mixture_fit = analyzer.analyze_components(model, data)
            assignments = analyzer.assign_clusters(model, data)
            cluster_stats = analyzer.characterize_clusters(data, assignments)
            
            # Generate visualization
            fig = analyzer.create_visualization(
                data, mixture_fit, assignments
            )
            fig.write_html(
                history_dir /
                f"mixture_{distribution_result.metric}_{distribution_result.parameter_value}.html"
            )
            
            # Generate report
            report = analyzer.generate_report(
                mixture_fit,
                cluster_stats,
                assignments
            )
            report_file = history_dir / f"mixture_{distribution_result.metric}.md"
            report_file.write_text(report)
        
        logger.info(f"Mixture analysis complete. Reports written to {history_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during mixture analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
