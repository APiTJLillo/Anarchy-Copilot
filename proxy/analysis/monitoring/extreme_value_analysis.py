"""Extreme value analysis for risk modeling."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .probabilistic_modeling import (
    ProbabilisticModeler, ProbabilisticConfig, Distribution,
    ProbabilisticResult
)

@dataclass
class ExtremeValueConfig:
    """Configuration for extreme value analysis."""
    enabled: bool = True 
    update_interval: float = 300.0  # 5 minutes
    tail_fraction: float = 0.1  # Use top 10% of data
    block_size: int = 20  # For block maxima method
    min_exceedances: int = 10  # For peaks-over-threshold
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 50, 100])
    confidence_level: float = 0.95
    enable_dynamic_threshold: bool = True
    max_clusters: int = 3
    enable_declustering: bool = True
    cluster_separation: timedelta = timedelta(hours=1)
    visualization_dir: Optional[str] = "extreme_value_analysis"

@dataclass
class ExtremeParams:
    """Parameters for extreme value distributions."""
    location: float
    scale: float
    shape: float
    threshold: float
    confidence_intervals: Dict[str, Tuple[float, float]]

@dataclass
class ReturnLevel:
    """Return level estimates."""
    period: int
    level: float
    confidence_interval: Tuple[float, float]
    exceedance_probability: float

@dataclass
class ExtremeValueResult:
    """Results of extreme value analysis."""
    distribution_params: Dict[str, ExtremeParams]
    return_levels: Dict[str, List[ReturnLevel]]
    threshold_exceedances: Dict[str, np.ndarray]
    cluster_analysis: Dict[str, Dict[str, Any]]
    stability_metrics: Dict[str, float]
    diagnostic_stats: Dict[str, Dict[str, float]]

class ExtremeValueAnalyzer:
    """Analyze extreme values in probability distributions."""
    
    def __init__(
        self,
        modeler: ProbabilisticModeler,
        config: ExtremeValueConfig = None
    ):
        self.modeler = modeler
        self.config = config or ExtremeValueConfig()
        
        # Analysis state
        self.results: Dict[str, ExtremeValueResult] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}
        
        # Monitoring state
        self.last_update = datetime.min
        self.analyzer_task: Optional[asyncio.Task] = None
    
    async def start_analyzer(self):
        """Start extreme value analyzer."""
        if not self.config.enabled:
            return
        
        if self.analyzer_task is None:
            self.analyzer_task = asyncio.create_task(self._run_analyzer())
    
    async def stop_analyzer(self):
        """Stop extreme value analyzer."""
        if self.analyzer_task:
            self.analyzer_task.cancel()
            try:
                await self.analyzer_task
            except asyncio.CancelledError:
                pass
            self.analyzer_task = None
    
    async def _run_analyzer(self):
        """Run periodic analysis."""
        while True:
            try:
                current_time = datetime.now()
                
                if (
                    current_time - self.last_update >=
                    timedelta(seconds=self.config.update_interval)
                ):
                    for scenario in self.modeler.distributions:
                        await self.analyze_extremes(scenario)
                    
                    self.last_update = current_time
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                print(f"Extreme value analyzer error: {e}")
                await asyncio.sleep(60)
    
    async def analyze_extremes(
        self,
        scenario_name: str
    ) -> Optional[ExtremeValueResult]:
        """Analyze extreme values for scenario."""
        if scenario_name not in self.modeler.distributions:
            return None
        
        distributions = self.modeler.distributions[scenario_name]
        if not distributions:
            return None
        
        # Initialize results
        dist_params = {}
        return_levels = {}
        threshold_exceedances = {}
        cluster_analysis = {}
        stability_metrics = {}
        diagnostic_stats = {}
        
        for metric, dist in distributions.items():
            # Get exceedances over threshold
            threshold = await self._estimate_threshold(
                dist.samples,
                metric,
                scenario_name
            )
            
            exceedances = dist.samples[dist.samples > threshold]
            if len(exceedances) < self.config.min_exceedances:
                continue
            
            threshold_exceedances[metric] = exceedances
            
            # Fit extreme value distribution
            params = await self._fit_gpd(exceedances, threshold)
            dist_params[metric] = params
            
            # Calculate return levels
            levels = await self._calculate_return_levels(
                exceedances,
                params,
                threshold
            )
            return_levels[metric] = levels
            
            # Perform cluster analysis
            if self.config.enable_declustering:
                clusters = await self._analyze_clusters(
                    dist.samples,
                    threshold,
                    metric
                )
                cluster_analysis[metric] = clusters
            
            # Calculate stability metrics
            stability = await self._assess_stability(
                exceedances,
                params,
                threshold
            )
            stability_metrics[metric] = stability
            
            # Calculate diagnostic statistics
            diagnostics = await self._calculate_diagnostics(
                exceedances,
                params,
                threshold
            )
            diagnostic_stats[metric] = diagnostics
        
        # Create result
        result = ExtremeValueResult(
            distribution_params=dist_params,
            return_levels=return_levels,
            threshold_exceedances=threshold_exceedances,
            cluster_analysis=cluster_analysis,
            stability_metrics=stability_metrics,
            diagnostic_stats=diagnostic_stats
        )
        
        self.results[scenario_name] = result
        
        return result
    
    async def _estimate_threshold(
        self,
        samples: np.ndarray,
        metric: str,
        scenario: str
    ) -> float:
        """Estimate threshold for extreme value analysis."""
        if not self.config.enable_dynamic_threshold:
            # Use simple quantile
            return np.quantile(samples, 1 - self.config.tail_fraction)
        
        # Get historical threshold
        current_threshold = self.thresholds.get(
            scenario, {}
        ).get(metric)
        
        if current_threshold is None:
            # Initialize with quantile
            current_threshold = np.quantile(
                samples,
                1 - self.config.tail_fraction
            )
        
        # Analyze threshold stability
        exceedances = samples[samples > current_threshold]
        if len(exceedances) < self.config.min_exceedances:
            # Lower threshold if too few exceedances
            new_threshold = np.quantile(
                samples,
                1 - 2 * self.config.tail_fraction
            )
        elif len(exceedances) > 3 * self.config.min_exceedances:
            # Raise threshold if too many exceedances
            new_threshold = np.quantile(
                samples,
                1 - 0.5 * self.config.tail_fraction
            )
        else:
            new_threshold = current_threshold
        
        # Update threshold
        if scenario not in self.thresholds:
            self.thresholds[scenario] = {}
        self.thresholds[scenario][metric] = new_threshold
        
        return new_threshold
    
    async def _fit_gpd(
        self,
        exceedances: np.ndarray,
        threshold: float
    ) -> ExtremeParams:
        """Fit Generalized Pareto Distribution."""
        # Calculate shape and scale parameters
        excess = exceedances - threshold
        
        def negative_likelihood(params):
            shape, scale = params
            if scale <= 0:
                return float("inf")
            if shape == 0:
                return np.sum(np.log(scale) + excess / scale)
            z = 1 + shape * excess / scale
            if np.any(z <= 0):
                return float("inf")
            return np.sum(np.log(scale) + (1 + 1/shape) * np.log(z))
        
        # Optimize parameters
        result = minimize(
            negative_likelihood,
            x0=[0.1, np.std(excess)],
            method="Nelder-Mead"
        )
        
        shape, scale = result.x
        
        # Calculate confidence intervals
        hessian = result.hess_inv if hasattr(result, "hess_inv") else None
        if hessian is not None:
            # Use asymptotic normality
            std_errors = np.sqrt(np.diag(hessian))
            z = stats.norm.ppf(
                1 - (1 - self.config.confidence_level) / 2
            )
            confidence_intervals = {
                "shape": (
                    shape - z * std_errors[0],
                    shape + z * std_errors[0]
                ),
                "scale": (
                    scale - z * std_errors[1],
                    scale + z * std_errors[1]
                )
            }
        else:
            # Use bootstrap if Hessian not available
            confidence_intervals = await self._bootstrap_confidence(
                exceedances,
                threshold,
                shape,
                scale
            )
        
        return ExtremeParams(
            location=threshold,
            scale=scale,
            shape=shape,
            threshold=threshold,
            confidence_intervals=confidence_intervals
        )
    
    async def _bootstrap_confidence(
        self,
        exceedances: np.ndarray,
        threshold: float,
        shape: float,
        scale: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals."""
        n_samples = len(exceedances)
        bootstraps = 1000
        
        shape_samples = []
        scale_samples = []
        
        for _ in range(bootstraps):
            # Resample with replacement
            indices = np.random.randint(0, n_samples, n_samples)
            bootstrap_sample = exceedances[indices]
            
            # Fit GPD
            excess = bootstrap_sample - threshold
            try:
                result = minimize(
                    lambda params: -np.sum(stats.genpareto.logpdf(
                        excess,
                        c=params[0],
                        scale=params[1]
                    )),
                    x0=[shape, scale],
                    method="Nelder-Mead"
                )
                if result.success:
                    shape_samples.append(result.x[0])
                    scale_samples.append(result.x[1])
            except:
                continue
        
        if not shape_samples:
            return {
                "shape": (shape, shape),
                "scale": (scale, scale)
            }
        
        # Calculate percentile intervals
        alpha = (1 - self.config.confidence_level) / 2
        return {
            "shape": (
                np.percentile(shape_samples, alpha * 100),
                np.percentile(shape_samples, (1 - alpha) * 100)
            ),
            "scale": (
                np.percentile(scale_samples, alpha * 100),
                np.percentile(scale_samples, (1 - alpha) * 100)
            )
        }
    
    async def _calculate_return_levels(
        self,
        exceedances: np.ndarray,
        params: ExtremeParams,
        threshold: float
    ) -> List[ReturnLevel]:
        """Calculate return levels for different periods."""
        n_samples = len(exceedances)
        n_total = int(n_samples / self.config.tail_fraction)
        
        return_levels = []
        for period in self.config.return_periods:
            # Calculate return level
            p = 1 / period
            if params.shape == 0:
                level = threshold + params.scale * np.log(1/p)
            else:
                level = threshold + (
                    params.scale / params.shape *
                    ((n_total * p) ** -params.shape - 1)
                )
            
            # Calculate confidence interval
            ci = await self._return_level_confidence(
                level,
                params,
                period,
                n_total
            )
            
            # Calculate exceedance probability
            prob = 1 - stats.genpareto.cdf(
                level - threshold,
                c=params.shape,
                scale=params.scale
            )
            
            return_levels.append(ReturnLevel(
                period=period,
                level=level,
                confidence_interval=ci,
                exceedance_probability=prob
            ))
        
        return return_levels
    
    async def _return_level_confidence(
        self,
        level: float,
        params: ExtremeParams,
        period: int,
        n_total: int
    ) -> Tuple[float, float]:
        """Calculate confidence interval for return level."""
        # Use delta method
        var_shape = (
            (params.confidence_intervals["shape"][1] -
             params.confidence_intervals["shape"][0]) / 3.92
        )
        var_scale = (
            (params.confidence_intervals["scale"][1] -
             params.confidence_intervals["scale"][0]) / 3.92
        )
        
        # Calculate gradient
        if params.shape == 0:
            grad_shape = 0
            grad_scale = np.log(period)
        else:
            u = period ** -params.shape
            grad_shape = (
                params.scale / params.shape**2 *
                (1 - u + u * np.log(period))
            )
            grad_scale = u / params.shape
        
        # Calculate variance
        var_level = (
            grad_shape**2 * var_shape +
            grad_scale**2 * var_scale
        )
        
        # Calculate confidence interval
        z = stats.norm.ppf(
            1 - (1 - self.config.confidence_level) / 2
        )
        margin = z * np.sqrt(var_level)
        
        return (level - margin, level + margin)
    
    async def _analyze_clusters(
        self,
        samples: np.ndarray,
        threshold: float,
        metric: str
    ) -> Dict[str, Any]:
        """Analyze clusters of extreme values."""
        exceedances = np.where(samples > threshold)[0]
        
        if not self.config.enable_declustering:
            return {
                "n_clusters": 1,
                "cluster_sizes": [len(exceedances)],
                "cluster_maxima": [samples[exceedances].max()],
                "extremal_index": 1.0
            }
        
        # Find clusters using runs declustering
        clusters = []
        current_cluster = [exceedances[0]]
        
        for i in range(1, len(exceedances)):
            if (
                exceedances[i] - exceedances[i-1] >
                self.config.cluster_separation.total_seconds()
            ):
                clusters.append(current_cluster)
                current_cluster = []
            current_cluster.append(exceedances[i])
        
        if current_cluster:
            clusters.append(current_cluster)
        
        # Calculate cluster properties
        cluster_sizes = [len(c) for c in clusters]
        cluster_maxima = [
            samples[cluster].max()
            for cluster in clusters
        ]
        
        # Calculate extremal index
        n_clusters = len(clusters)
        n_exceedances = len(exceedances)
        extremal_index = n_clusters / n_exceedances if n_exceedances > 0 else 1.0
        
        return {
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes,
            "cluster_maxima": cluster_maxima,
            "extremal_index": extremal_index
        }
    
    async def _assess_stability(
        self,
        exceedances: np.ndarray,
        params: ExtremeParams,
        threshold: float
    ) -> float:
        """Assess stability of extreme value fit."""
        # Split data into two parts
        mid = len(exceedances) // 2
        first_half = exceedances[:mid]
        second_half = exceedances[mid:]
        
        # Fit GPD to both halves
        try:
            params1 = await self._fit_gpd(first_half, threshold)
            params2 = await self._fit_gpd(second_half, threshold)
            
            # Calculate parameter stability
            shape_diff = abs(params1.shape - params2.shape)
            scale_diff = abs(params1.scale - params2.scale)
            
            # Normalize differences
            max_shape = max(abs(params1.shape), abs(params2.shape))
            max_scale = max(abs(params1.scale), abs(params2.scale))
            
            if max_shape > 0 and max_scale > 0:
                stability = 1 - 0.5 * (
                    shape_diff / max_shape +
                    scale_diff / max_scale
                )
            else:
                stability = 0.0
            
        except:
            stability = 0.0
        
        return stability
    
    async def _calculate_diagnostics(
        self,
        exceedances: np.ndarray,
        params: ExtremeParams,
        threshold: float
    ) -> Dict[str, float]:
        """Calculate diagnostic statistics."""
        diagnostics = {}
        
        # Mean excess function
        excess = exceedances - threshold
        diagnostics["mean_excess"] = np.mean(excess)
        
        # Anderson-Darling test
        try:
            ad_stat, _ = stats.anderson_ksamp([
                excess,
                stats.genpareto.rvs(
                    c=params.shape,
                    scale=params.scale,
                    size=len(excess)
                )
            ])
            diagnostics["anderson_darling"] = ad_stat
        except:
            diagnostics["anderson_darling"] = float("inf")
        
        # Kolmogorov-Smirnov test
        try:
            ks_stat, _ = stats.kstest(
                excess,
                lambda x: stats.genpareto.cdf(
                    x,
                    c=params.shape,
                    scale=params.scale
                )
            )
            diagnostics["kolmogorov_smirnov"] = ks_stat
        except:
            diagnostics["kolmogorov_smirnov"] = float("inf")
        
        # L-moments ratio
        try:
            lmom = np.sort(excess)
            l1 = np.mean(lmom)
            l2 = np.mean(
                (np.arange(1, len(lmom) + 1) - 1) * lmom
            ) / len(lmom)
            diagnostics["l_moments_ratio"] = l2 / l1 if l1 != 0 else float("inf")
        except:
            diagnostics["l_moments_ratio"] = float("inf")
        
        return diagnostics
    
    async def create_extreme_plots(self) -> Dict[str, go.Figure]:
        """Create extreme value visualization plots."""
        plots = {}
        
        for scenario_name, result in self.results.items():
            if not result.distribution_params:
                continue
            
            # Return level plot
            level_fig = go.Figure()
            
            for metric, levels in result.return_levels.items():
                periods = [l.period for l in levels]
                values = [l.level for l in levels]
                lower = [l.confidence_interval[0] for l in levels]
                upper = [l.confidence_interval[1] for l in levels]
                
                level_fig.add_trace(go.Scatter(
                    x=periods,
                    y=values,
                    name=metric,
                    mode="lines+markers"
                ))
                
                level_fig.add_trace(go.Scatter(
                    x=periods + periods[::-1],
                    y=upper + lower[::-1],
                    fill="toself",
                    fillcolor="rgba(0,0,255,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{metric} Confidence"
                ))
            
            level_fig.update_layout(
                title=f"Return Level Plot - {scenario_name}",
                xaxis_title="Return Period (years)",
                yaxis_title="Return Level",
                xaxis_type="log",
                showlegend=True
            )
            plots[f"{scenario_name}_return_levels"] = level_fig
            
            # Diagnostic plots
            diag_fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    "Mean Excess Plot",
                    "Parameter Stability",
                    "QQ Plot",
                    "Cluster Analysis"
                ]
            )
            
            for metric, exceedances in result.threshold_exceedances.items():
                params = result.distribution_params[metric]
                
                # Mean excess plot
                thresholds = np.linspace(
                    params.threshold,
                    np.max(exceedances),
                    20
                )
                mean_excess = []
                
                for u in thresholds:
                    excess = exceedances[exceedances > u] - u
                    if len(excess) > 0:
                        mean_excess.append(np.mean(excess))
                    else:
                        mean_excess.append(0)
                
                diag_fig.add_trace(
                    go.Scatter(
                        x=thresholds,
                        y=mean_excess,
                        name=f"{metric} Mean Excess"
                    ),
                    row=1,
                    col=1
                )
                
                # Parameter stability plot
                if metric in result.stability_metrics:
                    diag_fig.add_trace(
                        go.Bar(
                            x=[metric],
                            y=[result.stability_metrics[metric]],
                            name=f"{metric} Stability"
                        ),
                        row=1,
                        col=2
                    )
                
                # QQ plot
                theoretical = stats.genpareto.ppf(
                    np.linspace(0.01, 0.99, len(exceedances)),
                    c=params.shape,
                    scale=params.scale
                )
                empirical = np.sort(exceedances - params.threshold)
                
                diag_fig.add_trace(
                    go.Scatter(
                        x=theoretical,
                        y=empirical,
                        name=f"{metric} QQ",
                        mode="markers"
                    ),
                    row=2,
                    col=1
                )
                
                # Reference line
                max_val = max(theoretical.max(), empirical.max())
                diag_fig.add_trace(
                    go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        name="Reference",
                        line=dict(dash="dash"),
                        showlegend=False
                    ),
                    row=2,
                    col=1
                )
                
                # Cluster analysis plot
                if metric in result.cluster_analysis:
                    clusters = result.cluster_analysis[metric]
                    diag_fig.add_trace(
                        go.Bar(
                            x=list(range(1, len(clusters["cluster_sizes"]) + 1)),
                            y=clusters["cluster_sizes"],
                            name=f"{metric} Clusters"
                        ),
                        row=2,
                        col=2
                    )
            
            diag_fig.update_layout(
                height=800,
                title=f"Diagnostic Plots - {scenario_name}",
                showlegend=True
            )
            plots[f"{scenario_name}_diagnostics"] = diag_fig
        
        # Save plots
        if self.config.visualization_dir:
            from pathlib import Path
            path = Path(self.config.visualization_dir)
            path.mkdir(parents=True, exist_ok=True)
            for name, fig in plots.items():
                fig.write_html(str(path / f"extreme_{name}.html"))
        
        return plots

def create_extreme_analyzer(
    modeler: ProbabilisticModeler,
    config: Optional[ExtremeValueConfig] = None
) -> ExtremeValueAnalyzer:
    """Create extreme value analyzer."""
    return ExtremeValueAnalyzer(modeler, config)

if __name__ == "__main__":
    from .probabilistic_modeling import create_probabilistic_modeler
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
        extreme = create_extreme_analyzer(modeler)
        
        # Start components
        await prevention.start_prevention()
        await balancer.start_balancing()
        await advisor.start_advisor()
        await analyzer.start_analyzer()
        await predictor.start_predictor()
        await planner.start_planner()
        await whatif.start_analyzer()
        await modeler.start_modeler()
        await extreme.start_analyzer()
        
        try:
            while True:
                # Analyze scenarios
                for scenario in modeler.distributions:
                    result = await extreme.analyze_extremes(scenario)
                    if result:
                        print(f"\nExtreme Value Analysis for {scenario}:")
                        
                        for metric in result.distribution_params:
                            params = result.distribution_params[metric]
                            print(f"\n{metric}:")
                            print(f"  Shape: {params.shape:.3f}")
                            print(f"  Scale: {params.scale:.3f}")
                            print(f"  Threshold: {params.threshold:.3f}")
                            
                            if metric in result.return_levels:
                                print("\n  Return Levels:")
                                for level in result.return_levels[metric]:
                                    print(
                                        f"    {level.period} year: "
                                        f"{level.level:.3f} "
                                        f"({level.exceedance_probability:.1%} "
                                        "exceedance)"
                                    )
                
                # Create plots
                await extreme.create_extreme_plots()
                
                await asyncio.sleep(60)
        finally:
            await extreme.stop_analyzer()
            await modeler.stop_modeler()
            await whatif.stop_analyzer()
            await planner.stop_planner()
            await predictor.stop_predictor()
            await analyzer.stop_analyzer()
            await advisor.stop_advisor()
            await balancer.stop_balancing()
            await prevention.stop_prevention()
    
    asyncio.run(main())
