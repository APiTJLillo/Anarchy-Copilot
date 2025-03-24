"""Intervention analysis for causal relationships."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .causal_extremes import (
    CausalAnalyzer, CausalConfig, CausalLink,
    CausalNetwork, CausalResult
)

@dataclass
class InterventionConfig:
    """Configuration for intervention analysis."""
    enabled: bool = True
    update_interval: float = 300.0  # 5 minutes
    min_effect_size: float = 0.1
    confidence_level: float = 0.95
    min_stability: float = 0.7
    max_interventions: int = 5
    min_intervention_gap: timedelta = timedelta(hours=1)
    enable_counterfactuals: bool = True
    counterfactual_samples: int = 1000
    enable_robustness: bool = True
    robustness_trials: int = 100
    enable_cost_analysis: bool = True
    optimization_iterations: int = 500
    visualization_dir: Optional[str] = "intervention_analysis"

@dataclass
class Intervention:
    """Intervention definition."""
    target: str
    action: str  # increase, decrease, stabilize
    magnitude: float
    timing: datetime
    duration: timedelta
    constraints: Dict[str, Tuple[float, float]]
    cost: float = 0.0
    priority: float = 1.0

@dataclass
class InterventionEffect:
    """Effect of intervention."""
    direct_impact: float
    indirect_impacts: Dict[str, float]
    total_impact: float
    confidence_interval: Tuple[float, float]
    stability_score: float
    success_probability: float
    recovery_time: float

@dataclass
class Counterfactual:
    """Counterfactual scenario."""
    intervention: Intervention
    baseline: Dict[str, float]
    outcomes: Dict[str, float]
    probability: float
    key_factors: Dict[str, float]

@dataclass
class InterventionResult:
    """Results of intervention analysis."""
    interventions: Dict[str, List[Intervention]]
    effects: Dict[str, Dict[str, InterventionEffect]]
    counterfactuals: Dict[str, List[Counterfactual]]
    optimization: Dict[str, Dict[str, Any]]
    robustness: Dict[str, Dict[str, float]]
    constraints: Dict[str, Dict[str, Any]]

class InterventionAnalyzer:
    """Analyze intervention effects in causal system."""
    
    def __init__(
        self,
        causal_analyzer: CausalAnalyzer,
        config: InterventionConfig = None
    ):
        self.causal_analyzer = causal_analyzer
        self.config = config or InterventionConfig()
        
        # Analysis state
        self.results: Dict[str, InterventionResult] = {}
        self.history: Dict[str, List[Intervention]] = {}
        self.scaler = StandardScaler()
        
        # Monitoring state
        self.last_update = datetime.min
        self.analyzer_task: Optional[asyncio.Task] = None
    
    async def start_analyzer(self):
        """Start intervention analyzer."""
        if not self.config.enabled:
            return
        
        if self.analyzer_task is None:
            self.analyzer_task = asyncio.create_task(self._run_analyzer())
    
    async def stop_analyzer(self):
        """Stop intervention analyzer."""
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
                    for scenario in self.causal_analyzer.results:
                        await self.analyze_interventions(scenario)
                    
                    self.last_update = current_time
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                print(f"Intervention analyzer error: {e}")
                await asyncio.sleep(60)
    
    async def analyze_interventions(
        self,
        scenario_name: str
    ) -> Optional[InterventionResult]:
        """Analyze potential interventions for scenario."""
        if scenario_name not in self.causal_analyzer.results:
            return None
        
        result = self.causal_analyzer.results[scenario_name]
        if not result.networks:
            return None
        
        # Generate potential interventions
        interventions = await self._generate_interventions(result)
        
        # Calculate intervention effects
        effects = await self._analyze_effects(
            result,
            interventions
        )
        
        # Generate counterfactuals
        counterfactuals = {}
        if self.config.enable_counterfactuals:
            counterfactuals = await self._generate_counterfactuals(
                result,
                interventions,
                effects
            )
        
        # Optimize interventions
        optimization = await self._optimize_interventions(
            result,
            interventions,
            effects
        )
        
        # Analyze robustness
        robustness = {}
        if self.config.enable_robustness:
            robustness = await self._analyze_robustness(
                result,
                interventions,
                effects
            )
        
        # Analyze constraints
        constraints = await self._analyze_constraints(
            result,
            interventions,
            effects
        )
        
        # Create result
        intervention_result = InterventionResult(
            interventions=interventions,
            effects=effects,
            counterfactuals=counterfactuals,
            optimization=optimization,
            robustness=robustness,
            constraints=constraints
        )
        
        self.results[scenario_name] = intervention_result
        
        return intervention_result
    
    async def _generate_interventions(
        self,
        result: CausalResult
    ) -> Dict[str, List[Intervention]]:
        """Generate potential interventions."""
        interventions = {}
        
        # Analyze each network
        for name, network in result.networks.items():
            network_interventions = []
            
            # Find central nodes
            central_nodes = sorted(
                network.centrality_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Generate interventions for top nodes
            for node, centrality in central_nodes[:self.config.max_interventions]:
                # Check node stability
                if network.stability_score >= self.config.min_stability:
                    # Generate different intervention types
                    for action in ["increase", "decrease", "stabilize"]:
                        # Calculate appropriate magnitude
                        magnitude = centrality * self.config.min_effect_size
                        
                        # Set timing and duration
                        timing = datetime.now() + timedelta(minutes=15)
                        duration = timedelta(hours=1)
                        
                        # Set constraints based on network structure
                        constraints = {}
                        for link in network.links:
                            if link.source == node:
                                constraints[link.target] = (
                                    -link.effect_size,
                                    link.effect_size
                                )
                        
                        intervention = Intervention(
                            target=node,
                            action=action,
                            magnitude=magnitude,
                            timing=timing,
                            duration=duration,
                            constraints=constraints,
                            cost=centrality,
                            priority=centrality
                        )
                        
                        network_interventions.append(intervention)
            
            interventions[name] = network_interventions
        
        return interventions
    
    async def _analyze_effects(
        self,
        result: CausalResult,
        interventions: Dict[str, List[Intervention]]
    ) -> Dict[str, Dict[str, InterventionEffect]]:
        """Analyze effects of interventions."""
        effects = {}
        
        for name, network in result.networks.items():
            network_effects = {}
            
            for intervention in interventions.get(name, []):
                # Calculate direct impact
                direct_impact = intervention.magnitude
                
                # Calculate indirect impacts
                indirect_impacts = {}
                for link in network.links:
                    if link.source == intervention.target:
                        impact = link.effect_size * intervention.magnitude
                        indirect_impacts[link.target] = impact
                
                # Calculate total impact
                total_impact = (
                    direct_impact +
                    sum(indirect_impacts.values())
                )
                
                # Calculate confidence interval
                ci = await self._calculate_confidence_interval(
                    total_impact,
                    network.stability_score
                )
                
                # Calculate stability score
                stability = await self._assess_stability(
                    network,
                    intervention
                )
                
                # Calculate success probability
                success_prob = await self._estimate_success_probability(
                    network,
                    intervention,
                    total_impact
                )
                
                # Estimate recovery time
                recovery_time = await self._estimate_recovery_time(
                    network,
                    intervention
                )
                
                effect = InterventionEffect(
                    direct_impact=direct_impact,
                    indirect_impacts=indirect_impacts,
                    total_impact=total_impact,
                    confidence_interval=ci,
                    stability_score=stability,
                    success_probability=success_prob,
                    recovery_time=recovery_time
                )
                
                network_effects[intervention.target] = effect
            
            effects[name] = network_effects
        
        return effects
    
    async def _generate_counterfactuals(
        self,
        result: CausalResult,
        interventions: Dict[str, List[Intervention]],
        effects: Dict[str, Dict[str, InterventionEffect]]
    ) -> Dict[str, List[Counterfactual]]:
        """Generate counterfactual scenarios."""
        counterfactuals = {}
        
        for name, network in result.networks.items():
            network_counterfactuals = []
            
            for intervention in interventions.get(name, []):
                # Get current state as baseline
                baseline = {
                    node: 0.0
                    for node in network.nodes
                }
                
                # Generate samples
                for _ in range(self.config.counterfactual_samples):
                    # Simulate alternative outcome
                    outcomes = baseline.copy()
                    
                    # Apply intervention effect
                    effect = effects[name][intervention.target]
                    outcomes[intervention.target] = effect.direct_impact
                    
                    # Propagate effects
                    for target, impact in effect.indirect_impacts.items():
                        outcomes[target] = impact
                    
                    # Add random variations
                    for node in outcomes:
                        outcomes[node] += np.random.normal(
                            0,
                            1 - network.stability_score
                        )
                    
                    # Calculate probability
                    prob = np.exp(
                        -sum(abs(outcomes[n] - baseline[n])
                        for n in network.nodes)
                    )
                    
                    # Identify key factors
                    key_factors = {}
                    for node in network.nodes:
                        if abs(outcomes[node] - baseline[node]) > self.config.min_effect_size:
                            key_factors[node] = outcomes[node] - baseline[node]
                    
                    counterfactual = Counterfactual(
                        intervention=intervention,
                        baseline=baseline,
                        outcomes=outcomes,
                        probability=prob,
                        key_factors=key_factors
                    )
                    
                    network_counterfactuals.append(counterfactual)
            
            counterfactuals[name] = network_counterfactuals
        
        return counterfactuals
    
    async def _optimize_interventions(
        self,
        result: CausalResult,
        interventions: Dict[str, List[Intervention]],
        effects: Dict[str, Dict[str, InterventionEffect]]
    ) -> Dict[str, Dict[str, Any]]:
        """Optimize intervention parameters."""
        optimization = {}
        
        for name, network in result.networks.items():
            network_optimization = {}
            
            for intervention in interventions.get(name, []):
                effect = effects[name][intervention.target]
                
                # Optimize magnitude
                best_magnitude = intervention.magnitude
                best_impact = effect.total_impact
                
                for _ in range(self.config.optimization_iterations):
                    # Try different magnitude
                    test_magnitude = best_magnitude * (
                        1 + np.random.normal(0, 0.1)
                    )
                    
                    # Calculate new impact
                    test_impact = test_magnitude * sum(
                        1 + link.effect_size
                        for link in network.links
                        if link.source == intervention.target
                    )
                    
                    # Update if better
                    if (
                        test_impact > best_impact and
                        await self._check_constraints(
                            network,
                            intervention,
                            test_magnitude
                        )
                    ):
                        best_magnitude = test_magnitude
                        best_impact = test_impact
                
                # Optimize timing
                best_timing = intervention.timing
                min_gap = self.config.min_intervention_gap
                
                # Check historical interventions
                if name in self.history:
                    for hist_intervention in self.history[name]:
                        gap = abs(hist_intervention.timing - best_timing)
                        if gap < min_gap:
                            best_timing += min_gap - gap
                
                network_optimization[intervention.target] = {
                    "optimal_magnitude": best_magnitude,
                    "optimal_timing": best_timing,
                    "optimal_impact": best_impact
                }
            
            optimization[name] = network_optimization
        
        return optimization
    
    async def _analyze_robustness(
        self,
        result: CausalResult,
        interventions: Dict[str, List[Intervention]],
        effects: Dict[str, Dict[str, InterventionEffect]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze robustness of interventions."""
        robustness = {}
        
        for name, network in result.networks.items():
            network_robustness = {}
            
            for intervention in interventions.get(name, []):
                effect = effects[name][intervention.target]
                
                # Calculate success rate under perturbations
                successes = 0
                
                for _ in range(self.config.robustness_trials):
                    # Add random noise to network
                    noise = np.random.normal(
                        0,
                        1 - network.stability_score,
                        len(network.nodes)
                    )
                    
                    # Calculate perturbed impact
                    perturbed_impact = effect.total_impact + np.mean(noise)
                    
                    # Check if still effective
                    if (
                        abs(perturbed_impact) >=
                        abs(effect.total_impact) * self.config.min_effect_size
                    ):
                        successes += 1
                
                robustness_score = successes / self.config.robustness_trials
                network_robustness[intervention.target] = robustness_score
            
            robustness[name] = network_robustness
        
        return robustness
    
    async def _analyze_constraints(
        self,
        result: CausalResult,
        interventions: Dict[str, List[Intervention]],
        effects: Dict[str, Dict[str, InterventionEffect]]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze intervention constraints."""
        constraints = {}
        
        for name, network in result.networks.items():
            network_constraints = {}
            
            for intervention in interventions.get(name, []):
                effect = effects[name][intervention.target]
                
                # Check stability constraints
                stability_violated = (
                    effect.stability_score <
                    self.config.min_stability
                )
                
                # Check impact bounds
                impact_violated = any(
                    abs(impact) > constraint[1]
                    for impact, constraint in zip(
                        effect.indirect_impacts.values(),
                        intervention.constraints.values()
                    )
                )
                
                # Check timing constraints
                timing_violated = False
                if name in self.history:
                    for hist_intervention in self.history[name]:
                        if abs(
                            hist_intervention.timing -
                            intervention.timing
                        ) < self.config.min_intervention_gap:
                            timing_violated = True
                            break
                
                network_constraints[intervention.target] = {
                    "stability_violated": stability_violated,
                    "impact_violated": impact_violated,
                    "timing_violated": timing_violated,
                    "total_violations": sum([
                        stability_violated,
                        impact_violated,
                        timing_violated
                    ])
                }
            
            constraints[name] = network_constraints
        
        return constraints
    
    async def _calculate_confidence_interval(
        self,
        impact: float,
        stability: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for impact."""
        std_error = abs(impact) * (1 - stability)
        z_score = stats.norm.ppf(
            1 - (1 - self.config.confidence_level) / 2
        )
        margin = z_score * std_error
        
        return (impact - margin, impact + margin)
    
    async def _assess_stability(
        self,
        network: CausalNetwork,
        intervention: Intervention
    ) -> float:
        """Assess stability of intervention."""
        # Base stability on network properties
        stability = network.stability_score
        
        # Adjust for intervention magnitude
        stability *= np.exp(-abs(intervention.magnitude))
        
        # Adjust for network centrality
        centrality = network.centrality_scores[intervention.target]
        stability *= (1 - centrality)
        
        return min(1.0, max(0.0, stability))
    
    async def _estimate_success_probability(
        self,
        network: CausalNetwork,
        intervention: Intervention,
        impact: float
    ) -> float:
        """Estimate probability of intervention success."""
        # Base probability on network stability
        prob = network.stability_score
        
        # Adjust for impact size
        prob *= np.exp(-abs(impact))
        
        # Adjust for intervention constraints
        n_constraints = len(intervention.constraints)
        if n_constraints > 0:
            prob *= np.exp(-n_constraints * 0.1)
        
        return min(1.0, max(0.0, prob))
    
    async def _estimate_recovery_time(
        self,
        network: CausalNetwork,
        intervention: Intervention
    ) -> float:
        """Estimate system recovery time."""
        # Base time on intervention duration
        recovery_time = intervention.duration.total_seconds()
        
        # Adjust for network stability
        recovery_time *= (2 - network.stability_score)
        
        # Adjust for network complexity
        n_nodes = len(network.nodes)
        n_edges = len(network.links)
        complexity = np.log1p(n_nodes * n_edges)
        recovery_time *= complexity
        
        return recovery_time
    
    async def _check_constraints(
        self,
        network: CausalNetwork,
        intervention: Intervention,
        magnitude: float
    ) -> bool:
        """Check if magnitude satisfies constraints."""
        for target, (min_val, max_val) in intervention.constraints.items():
            # Calculate propagated effect
            effect = magnitude
            for link in network.links:
                if link.source == intervention.target and link.target == target:
                    effect *= link.effect_size
            
            if not min_val <= effect <= max_val:
                return False
        
        return True
    
    async def create_intervention_plots(self) -> Dict[str, go.Figure]:
        """Create intervention visualization plots."""
        plots = {}
        
        for scenario_name, result in self.results.items():
            # Impact analysis plot
            impact_fig = go.Figure()
            
            for name, effects in result.effects.items():
                if not effects:
                    continue
                
                targets = list(effects.keys())
                direct_impacts = [
                    effects[t].direct_impact
                    for t in targets
                ]
                total_impacts = [
                    effects[t].total_impact
                    for t in targets
                ]
                
                impact_fig.add_trace(go.Bar(
                    name="Direct Impact",
                    x=targets,
                    y=direct_impacts,
                    marker_color="blue"
                ))
                
                impact_fig.add_trace(go.Bar(
                    name="Total Impact",
                    x=targets,
                    y=total_impacts,
                    marker_color="red"
                ))
            
            impact_fig.update_layout(
                title=f"Intervention Impacts - {scenario_name}",
                xaxis_title="Target",
                yaxis_title="Impact",
                barmode="group",
                showlegend=True
            )
            plots[f"{scenario_name}_impacts"] = impact_fig
            
            # Robustness analysis plot
            if result.robustness:
                robust_fig = go.Figure()
                
                for name, scores in result.robustness.items():
                    if not scores:
                        continue
                    
                    targets = list(scores.keys())
                    robustness = list(scores.values())
                    
                    robust_fig.add_trace(go.Bar(
                        name=name,
                        x=targets,
                        y=robustness,
                        text=[f"{v:.1%}" for v in robustness]
                    ))
                
                robust_fig.update_layout(
                    title=f"Intervention Robustness - {scenario_name}",
                    xaxis_title="Target",
                    yaxis_title="Robustness Score",
                    showlegend=True
                )
                plots[f"{scenario_name}_robustness"] = robust_fig
            
            # Constraint analysis plot
            if result.constraints:
                const_fig = go.Figure()
                
                for name, constraints in result.constraints.items():
                    if not constraints:
                        continue
                    
                    targets = list(constraints.keys())
                    violations = [
                        c["total_violations"]
                        for c in constraints.values()
                    ]
                    
                    const_fig.add_trace(go.Bar(
                        name=name,
                        x=targets,
                        y=violations,
                        text=violations
                    ))
                
                const_fig.update_layout(
                    title=f"Constraint Violations - {scenario_name}",
                    xaxis_title="Target",
                    yaxis_title="Number of Violations",
                    showlegend=True
                )
                plots[f"{scenario_name}_constraints"] = const_fig
            
            # Optimization results plot
            if result.optimization:
                opt_fig = go.Figure()
                
                for name, optimizations in result.optimization.items():
                    if not optimizations:
                        continue
                    
                    targets = list(optimizations.keys())
                    impacts = [
                        opt["optimal_impact"]
                        for opt in optimizations.values()
                    ]
                    magnitudes = [
                        opt["optimal_magnitude"]
                        for opt in optimizations.values()
                    ]
                    
                    opt_fig.add_trace(go.Scatter(
                        name=f"{name} Impact",
                        x=targets,
                        y=impacts,
                        mode="lines+markers"
                    ))
                    
                    opt_fig.add_trace(go.Scatter(
                        name=f"{name} Magnitude",
                        x=targets,
                        y=magnitudes,
                        mode="lines+markers",
                        yaxis="y2"
                    ))
                
                opt_fig.update_layout(
                    title=f"Optimization Results - {scenario_name}",
                    xaxis_title="Target",
                    yaxis_title="Optimal Impact",
                    yaxis2=dict(
                        title="Optimal Magnitude",
                        overlaying="y",
                        side="right"
                    ),
                    showlegend=True
                )
                plots[f"{scenario_name}_optimization"] = opt_fig
        
        # Save plots
        if self.config.visualization_dir:
            from pathlib import Path
            path = Path(self.config.visualization_dir)
            path.mkdir(parents=True, exist_ok=True)
            for name, fig in plots.items():
                fig.write_html(str(path / f"intervention_{name}.html"))
        
        return plots

def create_intervention_analyzer(
    causal_analyzer: CausalAnalyzer,
    config: Optional[InterventionConfig] = None
) -> InterventionAnalyzer:
    """Create intervention analyzer."""
    return InterventionAnalyzer(causal_analyzer, config)

if __name__ == "__main__":
    from .causal_extremes import create_causal_analyzer
    from .temporal_extremes import create_temporal_analyzer
    from .extreme_value_analysis import create_extreme_analyzer
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
        temporal = create_temporal_analyzer(extreme)
        causal = create_causal_analyzer(temporal)
        intervention = create_intervention_analyzer(causal)
        
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
        await temporal.start_analyzer()
        await causal.start_analyzer()
        await intervention.start_analyzer()
        
        try:
            while True:
                # Analyze scenarios
                for scenario in causal.results:
                    result = await intervention.analyze_interventions(scenario)
                    if result:
                        print(f"\nIntervention Analysis for {scenario}:")
                        
                        for name, effects in result.effects.items():
                            print(f"\n{name} Effects:")
                            for target, effect in effects.items():
                                print(f"\n{target}:")
                                print(f"  Direct Impact: {effect.direct_impact:.3f}")
                                print(f"  Total Impact: {effect.total_impact:.3f}")
                                print(f"  Success Probability: {effect.success_probability:.1%}")
                                print(f"  Recovery Time: {effect.recovery_time:.1f}s")
                        
                        if result.optimization:
                            print("\nOptimization Results:")
                            for name, opts in result.optimization.items():
                                print(f"\n{name}:")
                                for target, opt in opts.items():
                                    print(
                                        f"  {target}: "
                                        f"magnitude={opt['optimal_magnitude']:.3f}, "
                                        f"impact={opt['optimal_impact']:.3f}"
                                    )
                        
                        if result.robustness:
                            print("\nRobustness Analysis:")
                            for name, scores in result.robustness.items():
                                print(f"\n{name}:")
                                for target, score in scores.items():
                                    print(f"  {target}: {score:.1%}")
                        
                        if result.constraints:
                            print("\nConstraint Violations:")
                            for name, constraints in result.constraints.items():
                                print(f"\n{name}:")
                                for target, violations in constraints.items():
                                    if violations["total_violations"] > 0:
                                        print(
                                            f"  {target}: "
                                            f"{violations['total_violations']} violations"
                                        )
                
                # Create plots
                await intervention.create_intervention_plots()
                
                await asyncio.sleep(60)
        finally:
            await intervention.stop_analyzer()
            await causal.stop_analyzer()
            await temporal.stop_analyzer()
            await extreme.stop_analyzer()
            await modeler.stop_modeler()
            await whatif.stop_analyzer()
            await planner.stop_planner()
            await predictor.stop_predictor()
            await analyzer.stop_analyzer()
            await advisor.stop_analyzer()
            await balancer.stop_balancing()
            await prevention.stop_prevention()
    
    asyncio.run(main())
