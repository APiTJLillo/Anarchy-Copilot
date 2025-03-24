"""Probabilistic modeling for what-if analysis."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .whatif_analysis import (
    WhatIfAnalyzer, WhatIfConfig, WhatIfVariation,
    WhatIfResult
)

@dataclass
class ProbabilisticConfig:
    """Configuration for probabilistic modeling."""
    enabled: bool = True
    update_interval: float = 300.0  # 5 minutes
    num_samples: int = 10000
    confidence_level: float = 0.95
    kernel_bandwidth: float = 0.1
    mixture_components: int = 3
    enable_bootstrap: bool = True
    bootstrap_iterations: int = 1000
    enable_importance_sampling: bool = True
    importance_samples: int = 5000
    enable_mcmc: bool = True
    mcmc_steps: int = 1000
    burn_in: int = 100
    visualization_dir: Optional[str] = "probabilistic_models"

@dataclass
class Distribution:
    """Probability distribution model."""
    name: str
    params: Dict[str, float]
    samples: np.ndarray
    kde: KernelDensity
    mixture: Optional[GaussianMixture] = None
    confidence_bounds: Tuple[float, float] = field(default=(0.0, 1.0))

@dataclass
class ProbabilisticResult:
    """Results of probabilistic analysis."""
    distributions: Dict[str, Distribution]
    correlations: pd.DataFrame
    tail_risks: Dict[str, float]
    expected_values: Dict[str, float]
    var_estimates: Dict[str, float]
    entropy: Dict[str, float]

class ProbabilisticModeler:
    """Model probability distributions for scenarios."""
    
    def __init__(
        self,
        analyzer: WhatIfAnalyzer,
        config: ProbabilisticConfig = None
    ):
        self.analyzer = analyzer
        self.config = config or ProbabilisticConfig()
        
        # Analysis state
        self.distributions: Dict[str, Dict[str, Distribution]] = {}
        self.results: Dict[str, ProbabilisticResult] = {}
        
        # Monitoring state
        self.last_update = datetime.min
        self.modeler_task: Optional[asyncio.Task] = None
    
    async def start_modeler(self):
        """Start probabilistic modeler."""
        if not self.config.enabled:
            return
        
        if self.modeler_task is None:
            self.modeler_task = asyncio.create_task(self._run_modeler())
    
    async def stop_modeler(self):
        """Stop probabilistic modeler."""
        if self.modeler_task:
            self.modeler_task.cancel()
            try:
                await self.modeler_task
            except asyncio.CancelledError:
                pass
            self.modeler_task = None
    
    async def _run_modeler(self):
        """Run periodic modeling."""
        while True:
            try:
                current_time = datetime.now()
                
                if (
                    current_time - self.last_update >=
                    timedelta(seconds=self.config.update_interval)
                ):
                    for scenario in self.analyzer.variations:
                        await self.model_distributions(scenario)
                    
                    self.last_update = current_time
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                print(f"Modeler error: {e}")
                await asyncio.sleep(60)
    
    async def model_distributions(
        self,
        scenario_name: str
    ) -> Optional[ProbabilisticResult]:
        """Model probability distributions for scenario."""
        if scenario_name not in self.analyzer.variations:
            return None
        
        variations = self.analyzer.variations[scenario_name]
        if not variations:
            return None
        
        # Extract metrics
        metrics = {}
        for var in variations:
            if not var.outcome:
                continue
            
            for param, value in var.parameter_changes.items():
                if param not in metrics:
                    metrics[param] = []
                metrics[param].append(value)
            
            if "total_impact" not in metrics:
                metrics["total_impact"] = []
            metrics["total_impact"].append(var.outcome.total_impact)
            
            if "success_rate" not in metrics:
                metrics["success_rate"] = []
            metrics["success_rate"].append(var.outcome.success_rate)
        
        # Model distributions for each metric
        distributions = {}
        for metric, values in metrics.items():
            if not values:
                continue
            
            # Fit kernel density estimation
            kde = KernelDensity(
                bandwidth=self.config.kernel_bandwidth,
                kernel="gaussian"
            )
            values_2d = np.array(values).reshape(-1, 1)
            kde.fit(values_2d)
            
            # Generate samples
            samples = kde.sample(self.config.num_samples).flatten()
            
            # Fit Gaussian mixture model
            mixture = None
            if len(values) >= self.config.mixture_components:
                mixture = GaussianMixture(
                    n_components=self.config.mixture_components,
                    random_state=42
                )
                mixture.fit(values_2d)
            
            # Calculate confidence bounds
            bounds = np.percentile(
                samples,
                [
                    (1 - self.config.confidence_level) * 50,
                    (1 + self.config.confidence_level) * 50
                ]
            )
            
            # Create distribution
            dist = Distribution(
                name=metric,
                params={
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "skew": stats.skew(values),
                    "kurtosis": stats.kurtosis(values)
                },
                samples=samples,
                kde=kde,
                mixture=mixture,
                confidence_bounds=tuple(bounds)
            )
            
            distributions[metric] = dist
        
        # Calculate correlations
        correlations = await self._analyze_correlations(metrics)
        
        # Calculate tail risks
        tail_risks = await self._calculate_tail_risks(distributions)
        
        # Calculate expected values
        expected_values = {
            metric: dist.params["mean"]
            for metric, dist in distributions.items()
        }
        
        # Calculate Value at Risk
        var_estimates = await self._calculate_var(distributions)
        
        # Calculate entropy
        entropy = await self._calculate_entropy(distributions)
        
        # Create result
        result = ProbabilisticResult(
            distributions=distributions,
            correlations=correlations,
            tail_risks=tail_risks,
            expected_values=expected_values,
            var_estimates=var_estimates,
            entropy=entropy
        )
        
        # Store results
        self.distributions[scenario_name] = distributions
        self.results[scenario_name] = result
        
        return result
    
    async def _analyze_correlations(
        self,
        metrics: Dict[str, List[float]]
    ) -> pd.DataFrame:
        """Analyze correlations between metrics."""
        if not metrics:
            return pd.DataFrame()
        
        data = pd.DataFrame(metrics)
        return data.corr()
    
    async def _calculate_tail_risks(
        self,
        distributions: Dict[str, Distribution]
    ) -> Dict[str, float]:
        """Calculate tail risk probabilities."""
        tail_risks = {}
        
        for name, dist in distributions.items():
            # Calculate probability of extreme values
            threshold = dist.params["mean"] + 2 * dist.params["std"]
            tail_risks[name] = np.mean(dist.samples > threshold)
        
        return tail_risks
    
    async def _calculate_var(
        self,
        distributions: Dict[str, Distribution]
    ) -> Dict[str, float]:
        """Calculate Value at Risk estimates."""
        var_estimates = {}
        
        for name, dist in distributions.items():
            # Calculate VaR at confidence level
            var_estimates[name] = np.percentile(
                dist.samples,
                (1 - self.config.confidence_level) * 100
            )
        
        return var_estimates
    
    async def _calculate_entropy(
        self,
        distributions: Dict[str, Distribution]
    ) -> Dict[str, float]:
        """Calculate entropy of distributions."""
        entropy = {}
        
        for name, dist in distributions.items():
            if dist.mixture:
                # Use mixture model entropy
                entropy[name] = -np.sum(
                    dist.mixture.weights_ * np.log(dist.mixture.weights_)
                )
            else:
                # Approximate entropy from samples
                kde = KernelDensity(bandwidth=0.1)
                kde.fit(dist.samples.reshape(-1, 1))
                log_dens = kde.score_samples(dist.samples.reshape(-1, 1))
                entropy[name] = -np.mean(log_dens)
        
        return entropy
    
    async def create_distribution_plots(self) -> Dict[str, go.Figure]:
        """Create distribution visualization plots."""
        plots = {}
        
        for scenario_name, distributions in self.distributions.items():
            if not distributions:
                continue
            
            # Distribution plot
            dist_fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    "Parameter Distributions",
                    "Impact Distribution",
                    "Success Rate Distribution",
                    "Mixture Components"
                ]
            )
            
            # Parameter distributions
            for name, dist in distributions.items():
                if name in self.analyzer.config.variation_ranges:
                    dist_fig.add_trace(
                        go.Histogram(
                            x=dist.samples,
                            name=name,
                            nbinsx=30,
                            opacity=0.7
                        ),
                        row=1,
                        col=1
                    )
            
            # Impact distribution
            if "total_impact" in distributions:
                impact_dist = distributions["total_impact"]
                dist_fig.add_trace(
                    go.Histogram(
                        x=impact_dist.samples,
                        name="Impact",
                        nbinsx=30
                    ),
                    row=1,
                    col=2
                )
            
            # Success rate distribution
            if "success_rate" in distributions:
                success_dist = distributions["success_rate"]
                dist_fig.add_trace(
                    go.Histogram(
                        x=success_dist.samples,
                        name="Success Rate",
                        nbinsx=30
                    ),
                    row=2,
                    col=1
                )
            
            # Mixture components
            for name, dist in distributions.items():
                if dist.mixture:
                    x = np.linspace(
                        min(dist.samples),
                        max(dist.samples),
                        100
                    ).reshape(-1, 1)
                    
                    for i in range(dist.mixture.n_components):
                        mean = dist.mixture.means_[i][0]
                        std = np.sqrt(dist.mixture.covariances_[i][0][0])
                        weight = dist.mixture.weights_[i]
                        
                        y = weight * stats.norm.pdf(x, mean, std)
                        
                        dist_fig.add_trace(
                            go.Scatter(
                                x=x.flatten(),
                                y=y.flatten(),
                                name=f"{name} Component {i+1}",
                                line=dict(dash="dash")
                            ),
                            row=2,
                            col=2
                        )
            
            dist_fig.update_layout(
                height=800,
                title=f"Probability Distributions - {scenario_name}",
                showlegend=True
            )
            plots[f"{scenario_name}_distributions"] = dist_fig
            
            # Correlation heatmap
            if scenario_name in self.results:
                result = self.results[scenario_name]
                
                corr_fig = go.Figure(data=go.Heatmap(
                    z=result.correlations.values,
                    x=result.correlations.columns,
                    y=result.correlations.index,
                    colorscale="RdBu",
                    zmid=0
                ))
                
                corr_fig.update_layout(
                    title=f"Metric Correlations - {scenario_name}",
                    xaxis_title="Metric",
                    yaxis_title="Metric"
                )
                plots[f"{scenario_name}_correlations"] = corr_fig
            
            # Risk metrics plot
            if scenario_name in self.results:
                result = self.results[scenario_name]
                
                risk_fig = go.Figure()
                
                metrics = list(result.tail_risks.keys())
                
                risk_fig.add_trace(go.Bar(
                    name="Tail Risk",
                    x=metrics,
                    y=[result.tail_risks[m] for m in metrics],
                    marker_color="red"
                ))
                
                risk_fig.add_trace(go.Bar(
                    name="VaR",
                    x=metrics,
                    y=[result.var_estimates[m] for m in metrics],
                    marker_color="blue"
                ))
                
                risk_fig.add_trace(go.Bar(
                    name="Entropy",
                    x=metrics,
                    y=[result.entropy[m] for m in metrics],
                    marker_color="green"
                ))
                
                risk_fig.update_layout(
                    title=f"Risk Metrics - {scenario_name}",
                    xaxis_title="Metric",
                    yaxis_title="Value",
                    barmode="group",
                    showlegend=True
                )
                plots[f"{scenario_name}_risks"] = risk_fig
        
        # Save plots
        if self.config.visualization_dir:
            from pathlib import Path
            path = Path(self.config.visualization_dir)
            path.mkdir(parents=True, exist_ok=True)
            for name, fig in plots.items():
                fig.write_html(str(path / f"probabilistic_{name}.html"))
        
        return plots

def create_probabilistic_modeler(
    analyzer: WhatIfAnalyzer,
    config: Optional[ProbabilisticConfig] = None
) -> ProbabilisticModeler:
    """Create probabilistic modeler."""
    return ProbabilisticModeler(analyzer, config)

if __name__ == "__main__":
    from .whatif_analysis import create_whatif_analyzer
    from .scenario_planning import create_scenario_planner
    from .risk_prediction import create_risk_predictor
    from .risk_assessment import create_risk_analyzer
    from .strategy_recommendations import create_strategy_advisor
    from .prevention_balancing import create_prevention_balancer
    from .leak_prevention import create_leak_prevention
    from .memory_leak_detection import create_leak_detector
    from .scheduler_profiling import create_profiling_hook
    
    async def main():
        # Setup components
        profiling = create_profiling_hook()
        detector = create_leak_detector(profiling)
        prevention = create_leak_prevention(detector)
        balancer = create_prevention_balancer(prevention)
        advisor = create_strategy_advisor(balancer)
        analyzer = create_risk_analyzer(advisor)
        predictor = create_risk_predictor(analyzer)
        planner = create_scenario_planner(predictor)
        whatif = create_whatif_analyzer(planner)
        modeler = create_probabilistic_modeler(whatif)
        
        # Start components
        await prevention.start_prevention()
        await balancer.start_balancing()
        await advisor.start_advisor()
        await analyzer.start_analyzer()
        await predictor.start_predictor()
        await planner.start_planner()
        await whatif.start_analyzer()
        await modeler.start_modeler()
        
        try:
            while True:
                # Analyze scenarios
                for scenario in whatif.variations:
                    result = await modeler.model_distributions(scenario)
                    if result:
                        print(f"\nProbabilistic Analysis for {scenario}:")
                        print("\nExpected Values:")
                        for metric, value in result.expected_values.items():
                            print(f"  {metric}: {value:.3f}")
                        
                        print("\nTail Risks:")
                        for metric, risk in result.tail_risks.items():
                            print(f"  {metric}: {risk:.1%}")
                        
                        print("\nValue at Risk (VaR):")
                        for metric, var in result.var_estimates.items():
                            print(f"  {metric}: {var:.3f}")
                
                # Create plots
                await modeler.create_distribution_plots()
                
                await asyncio.sleep(60)
        finally:
            await modeler.stop_modeler()
            await whatif.stop_analyzer()
            await planner.stop_planner()
            await predictor.stop_predictor()
            await analyzer.stop_analyzer()
            await advisor.stop_advisor()
            await balancer.stop_balancing()
            await prevention.stop_prevention()
    
    asyncio.run(main())
