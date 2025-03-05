"""Scenario planning for risk prediction."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict

from .risk_prediction import RiskPredictor, PredictionConfig, RiskPrediction

@dataclass
class ScenarioConfig:
    """Configuration for scenario planning."""
    enabled: bool = True
    update_interval: float = 600.0  # 10 minutes
    max_scenarios: int = 10
    parallel_limit: int = 5
    simulation_runs: int = 1000
    sensitivity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "memory_impact": 0.2,
        "cpu_impact": 0.3,
        "latency_impact": 0.25,
        "stability_impact": 0.15
    })
    enable_auto_adjustment: bool = True
    correlation_threshold: float = 0.7
    confidence_level: float = 0.95
    visualization_dir: Optional[str] = "scenario_analysis"

@dataclass
class ScenarioCondition:
    """Condition for scenario."""
    metric: str
    operator: str  # gt, lt, eq, between
    value: Union[float, Tuple[float, float]]
    weight: float = 1.0

@dataclass
class Scenario:
    """Scenario definition."""
    name: str
    description: str
    conditions: List[ScenarioCondition]
    probability: float = 0.0
    impact: float = 0.0
    mitigation_actions: List[str] = field(default_factory=list)

@dataclass
class ScenarioOutcome:
    """Outcome of scenario simulation."""
    scenario: Scenario
    risk_predictions: List[RiskPrediction]
    total_impact: float
    success_rate: float
    recovery_time: float
    resource_usage: Dict[str, float]
    confidence_interval: Tuple[float, float]
    sensitivity_scores: Dict[str, float]

class ScenarioPlanner:
    """Plan and analyze risk scenarios."""
    
    def __init__(
        self,
        predictor: RiskPredictor,
        config: ScenarioConfig = None
    ):
        self.predictor = predictor
        self.config = config or ScenarioConfig()
        
        # Scenario storage
        self.scenarios: Dict[str, Scenario] = {}
        self.outcomes: Dict[str, ScenarioOutcome] = {}
        
        # Analysis state
        self.last_update = datetime.min
        self.planner_task: Optional[asyncio.Task] = None
    
    async def start_planner(self):
        """Start scenario planner."""
        if not self.config.enabled:
            return
        
        if self.planner_task is None:
            self.planner_task = asyncio.create_task(self._run_planner())
    
    async def stop_planner(self):
        """Stop scenario planner."""
        if self.planner_task:
            self.planner_task.cancel()
            try:
                await self.planner_task
            except asyncio.CancelledError:
                pass
            self.planner_task = None
    
    async def add_scenario(
        self,
        scenario: Scenario
    ) -> bool:
        """Add new scenario for analysis."""
        if len(self.scenarios) >= self.config.max_scenarios:
            return False
        
        if scenario.name in self.scenarios:
            return False
        
        self.scenarios[scenario.name] = scenario
        
        # Analyze new scenario
        await self._analyze_scenario(scenario)
        
        return True
    
    async def remove_scenario(
        self,
        name: str
    ) -> bool:
        """Remove scenario from analysis."""
        if name not in self.scenarios:
            return False
        
        del self.scenarios[name]
        if name in self.outcomes:
            del self.outcomes[name]
        
        return True
    
    async def _run_planner(self):
        """Run periodic scenario planning."""
        while True:
            try:
                current_time = datetime.now()
                
                if (
                    current_time - self.last_update >=
                    timedelta(seconds=self.config.update_interval)
                ):
                    scenarios = list(self.scenarios.values())
                    chunks = [
                        scenarios[i:i + self.config.parallel_limit]
                        for i in range(0, len(scenarios), self.config.parallel_limit)
                    ]
                    
                    for chunk in chunks:
                        await asyncio.gather(*(
                            self._analyze_scenario(scenario)
                            for scenario in chunk
                        ))
                    
                    if self.config.enable_auto_adjustment:
                        await self._adjust_scenarios()
                    
                    self.last_update = current_time
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                print(f"Planner error: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_scenario(
        self,
        scenario: Scenario
    ):
        """Analyze scenario impacts and outcomes."""
        # Get strategy predictions
        predictions = []
        for strategy in self.predictor.analyzer.assessments:
            result = await self.predictor.predict_risks(strategy)
            if result.predictions:
                predictions.extend(result.predictions)
        
        if not predictions:
            return
        
        # Apply scenario conditions
        filtered_predictions = []
        for pred in predictions:
            if await self._check_conditions(pred, scenario.conditions):
                filtered_predictions.append(pred)
        
        if not filtered_predictions:
            return
        
        # Calculate impacts
        total_impact = np.mean([
            sum(v for v in p.predicted_impacts.values())
            for p in filtered_predictions
        ])
        
        success_rate = np.mean([
            1 - p.failure_probability
            for p in filtered_predictions
        ])
        
        recovery_time = np.mean([
            self.predictor.analyzer._estimate_recovery_time(
                self.predictor.analyzer.assessments[p.strategy]
            )
            for p in filtered_predictions
        ])
        
        # Calculate resource usage
        resource_usage = defaultdict(list)
        for pred in filtered_predictions:
            for resource, value in pred.predicted_impacts.items():
                resource_usage[resource].append(value)
        
        avg_resource_usage = {
            resource: np.mean(values)
            for resource, values in resource_usage.items()
        }
        
        # Calculate confidence interval
        risks = [p.total_risk for p in filtered_predictions]
        ci = stats.t.interval(
            self.config.confidence_level,
            len(risks) - 1,
            loc=np.mean(risks),
            scale=stats.sem(risks)
        )
        
        # Calculate sensitivity scores
        sensitivity = await self._calculate_sensitivity(
            filtered_predictions
        )
        
        # Create outcome
        outcome = ScenarioOutcome(
            scenario=scenario,
            risk_predictions=filtered_predictions,
            total_impact=total_impact,
            success_rate=success_rate,
            recovery_time=recovery_time,
            resource_usage=avg_resource_usage,
            confidence_interval=ci,
            sensitivity_scores=sensitivity
        )
        
        self.outcomes[scenario.name] = outcome
    
    async def _check_conditions(
        self,
        prediction: RiskPrediction,
        conditions: List[ScenarioCondition]
    ) -> bool:
        """Check if prediction meets scenario conditions."""
        for condition in conditions:
            value = prediction.predicted_impacts.get(condition.metric)
            if value is None:
                return False
            
            if condition.operator == "gt":
                if not value > condition.value:
                    return False
            elif condition.operator == "lt":
                if not value < condition.value:
                    return False
            elif condition.operator == "eq":
                if not abs(value - condition.value) < 0.01:
                    return False
            elif condition.operator == "between":
                low, high = condition.value
                if not low <= value <= high:
                    return False
        
        return True
    
    async def _calculate_sensitivity(
        self,
        predictions: List[RiskPrediction]
    ) -> Dict[str, float]:
        """Calculate sensitivity scores for metrics."""
        sensitivity = {}
        
        for metric in self.config.sensitivity_thresholds:
            values = [
                p.predicted_impacts.get(metric, 0.0)
                for p in predictions
            ]
            
            if not values:
                continue
            
            # Calculate sensitivity as normalized variance
            variance = np.var(values)
            max_value = max(values)
            if max_value > 0:
                sensitivity[metric] = min(
                    1.0,
                    variance / (max_value * self.config.sensitivity_thresholds[metric])
                )
            else:
                sensitivity[metric] = 0.0
        
        return sensitivity
    
    async def _adjust_scenarios(self):
        """Automatically adjust scenarios based on outcomes."""
        if not self.outcomes:
            return
        
        # Find correlated scenarios
        correlations = await self._analyze_correlations()
        
        # Merge similar scenarios
        for (s1, s2), corr in correlations.items():
            if corr > self.config.correlation_threshold:
                # Keep scenario with higher impact
                if (
                    self.outcomes[s1].total_impact >
                    self.outcomes[s2].total_impact
                ):
                    await self.remove_scenario(s2)
                else:
                    await self.remove_scenario(s1)
        
        # Update probabilities
        total_impact = sum(
            o.total_impact for o in self.outcomes.values()
        )
        
        if total_impact > 0:
            for name, outcome in self.outcomes.items():
                self.scenarios[name].probability = (
                    outcome.total_impact / total_impact
                )
                self.scenarios[name].impact = outcome.total_impact
    
    async def _analyze_correlations(
        self
    ) -> Dict[Tuple[str, str], float]:
        """Analyze correlations between scenarios."""
        correlations = {}
        scenarios = list(self.outcomes.keys())
        
        for i, s1 in enumerate(scenarios):
            outcome1 = self.outcomes[s1]
            risks1 = [p.total_risk for p in outcome1.risk_predictions]
            
            for s2 in scenarios[i+1:]:
                outcome2 = self.outcomes[s2]
                risks2 = [p.total_risk for p in outcome2.risk_predictions]
                
                # Truncate to same length
                min_len = min(len(risks1), len(risks2))
                correlation, _ = stats.pearsonr(
                    risks1[:min_len],
                    risks2[:min_len]
                )
                
                correlations[(s1, s2)] = correlation
        
        return correlations
    
    async def create_scenario_plots(self) -> Dict[str, go.Figure]:
        """Create scenario visualization plots."""
        plots = {}
        
        if not self.outcomes:
            return plots
        
        # Impact comparison plot
        impact_fig = go.Figure()
        
        scenarios = list(self.outcomes.keys())
        impacts = [self.outcomes[s].total_impact for s in scenarios]
        probabilities = [self.scenarios[s].probability for s in scenarios]
        
        impact_fig.add_trace(go.Bar(
            x=scenarios,
            y=impacts,
            name="Impact",
            marker_color="blue"
        ))
        
        impact_fig.add_trace(go.Scatter(
            x=scenarios,
            y=probabilities,
            name="Probability",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="red")
        ))
        
        impact_fig.update_layout(
            title="Scenario Impacts and Probabilities",
            xaxis_title="Scenario",
            yaxis_title="Impact",
            yaxis2=dict(
                title="Probability",
                overlaying="y",
                side="right"
            ),
            showlegend=True
        )
        plots["impacts"] = impact_fig
        
        # Resource usage plot
        resource_fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Memory Usage",
                "CPU Usage",
                "Latency",
                "Stability"
            ]
        )
        
        for scenario in scenarios:
            outcome = self.outcomes[scenario]
            
            resource_fig.add_trace(
                go.Bar(
                    x=[scenario],
                    y=[outcome.resource_usage.get("memory", 0)],
                    name=scenario
                ),
                row=1,
                col=1
            )
            
            resource_fig.add_trace(
                go.Bar(
                    x=[scenario],
                    y=[outcome.resource_usage.get("cpu", 0)],
                    name=scenario
                ),
                row=1,
                col=2
            )
            
            resource_fig.add_trace(
                go.Bar(
                    x=[scenario],
                    y=[outcome.resource_usage.get("latency", 0)],
                    name=scenario
                ),
                row=2,
                col=1
            )
            
            resource_fig.add_trace(
                go.Bar(
                    x=[scenario],
                    y=[outcome.resource_usage.get("stability", 0)],
                    name=scenario
                ),
                row=2,
                col=2
            )
        
        resource_fig.update_layout(
            height=800,
            showlegend=True,
            title="Resource Usage by Scenario"
        )
        plots["resources"] = resource_fig
        
        # Sensitivity analysis plot
        sensitivity_fig = go.Figure()
        
        metrics = set()
        for outcome in self.outcomes.values():
            metrics.update(outcome.sensitivity_scores.keys())
        
        for metric in sorted(metrics):
            values = [
                self.outcomes[s].sensitivity_scores.get(metric, 0)
                for s in scenarios
            ]
            
            sensitivity_fig.add_trace(go.Bar(
                name=metric,
                x=scenarios,
                y=values,
                text=[f"{v:.2%}" for v in values]
            ))
        
        sensitivity_fig.update_layout(
            title="Metric Sensitivity Analysis",
            xaxis_title="Scenario",
            yaxis_title="Sensitivity Score",
            barmode="group",
            showlegend=True
        )
        plots["sensitivity"] = sensitivity_fig
        
        # Save plots
        if self.config.visualization_dir:
            from pathlib import Path
            path = Path(self.config.visualization_dir)
            path.mkdir(parents=True, exist_ok=True)
            for name, fig in plots.items():
                fig.write_html(str(path / f"scenario_{name}.html"))
        
        return plots

def create_scenario_planner(
    predictor: RiskPredictor,
    config: Optional[ScenarioConfig] = None
) -> ScenarioPlanner:
    """Create scenario planner."""
    return ScenarioPlanner(predictor, config)

if __name__ == "__main__":
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
        
        # Start components
        await prevention.start_prevention()
        await balancer.start_balancing()
        await advisor.start_advisor()
        await analyzer.start_analyzer()
        await predictor.start_predictor()
        await planner.start_planner()
        
        try:
            # Add test scenarios
            test_scenarios = [
                Scenario(
                    name="high_memory_pressure",
                    description="High memory usage scenario",
                    conditions=[
                        ScenarioCondition(
                            metric="memory_impact",
                            operator="gt",
                            value=0.8
                        )
                    ]
                ),
                Scenario(
                    name="cpu_latency_spike",
                    description="High CPU and latency impact",
                    conditions=[
                        ScenarioCondition(
                            metric="cpu_impact",
                            operator="gt",
                            value=0.7
                        ),
                        ScenarioCondition(
                            metric="latency_impact",
                            operator="gt",
                            value=0.5
                        )
                    ]
                )
            ]
            
            for scenario in test_scenarios:
                await planner.add_scenario(scenario)
            
            while True:
                # Print scenario analysis
                for name, outcome in planner.outcomes.items():
                    print(f"\nScenario: {name}")
                    print(f"Impact: {outcome.total_impact:.2%}")
                    print(f"Success Rate: {outcome.success_rate:.2%}")
                    print(f"Recovery Time: {outcome.recovery_time:.1f}s")
                    print("\nResource Usage:")
                    for resource, usage in outcome.resource_usage.items():
                        print(f"  {resource}: {usage:.2%}")
                    print("\nSensitivity Scores:")
                    for metric, score in outcome.sensitivity_scores.items():
                        print(f"  {metric}: {score:.2%}")
                
                # Create plots
                await planner.create_scenario_plots()
                
                await asyncio.sleep(60)
        finally:
            await planner.stop_planner()
            await predictor.stop_predictor()
            await analyzer.stop_analyzer()
            await advisor.stop_advisor()
            await balancer.stop_balancing()
            await prevention.stop_prevention()
    
    asyncio.run(main())
