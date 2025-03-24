"""What-if analysis for scenario planning."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .scenario_planning import (
    ScenarioPlanner, ScenarioConfig, Scenario, 
    ScenarioCondition, ScenarioOutcome
)
from .risk_prediction import RiskPrediction

@dataclass
class WhatIfConfig:
    """Configuration for what-if analysis."""
    enabled: bool = True
    update_interval: float = 300.0  # 5 minutes
    max_variations: int = 20
    max_combinations: int = 100
    simulation_runs: int = 1000
    variation_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "memory_impact": (0.1, 1.0),
        "cpu_impact": (0.1, 1.0),
        "latency_impact": (0.1, 1.0),
        "stability_impact": (0.1, 1.0)
    })
    step_sizes: Dict[str, float] = field(default_factory=lambda: {
        "memory_impact": 0.1,
        "cpu_impact": 0.1,
        "latency_impact": 0.1,
        "stability_impact": 0.1
    })
    optimization_target: str = "total_impact"
    enable_constraints: bool = True
    constraints: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "success_rate": (0.8, 1.0),
        "recovery_time": (0, 300)  # seconds
    })
    visualization_dir: Optional[str] = "whatif_analysis"

@dataclass
class WhatIfVariation:
    """Variation of scenario parameters."""
    base_scenario: str
    parameter_changes: Dict[str, float]
    impact_delta: float = 0.0
    success_probability: float = 0.0
    risk_factors: Dict[str, float] = field(default_factory=dict)
    outcome: Optional[ScenarioOutcome] = None

@dataclass
class WhatIfResult:
    """Results of what-if analysis."""
    variations: List[WhatIfVariation]
    optimal_parameters: Dict[str, float]
    sensitivity_ranking: Dict[str, float]
    correlation_matrix: pd.DataFrame
    key_insights: List[str]

class WhatIfAnalyzer:
    """Analyze what-if scenarios."""
    
    def __init__(
        self,
        planner: ScenarioPlanner,
        config: WhatIfConfig = None
    ):
        self.planner = planner
        self.config = config or WhatIfConfig()
        
        # Analysis state
        self.variations: Dict[str, List[WhatIfVariation]] = {}
        self.results: Dict[str, WhatIfResult] = {}
        self.predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Monitoring state
        self.last_update = datetime.min
        self.analyzer_task: Optional[asyncio.Task] = None
    
    async def start_analyzer(self):
        """Start what-if analyzer."""
        if not self.config.enabled:
            return
        
        if self.analyzer_task is None:
            self.analyzer_task = asyncio.create_task(self._run_analyzer())
    
    async def stop_analyzer(self):
        """Stop what-if analyzer."""
        if self.analyzer_task:
            self.analyzer_task.cancel()
            try:
                await self.analyzer_task
            except asyncio.CancelledError:
                pass
            self.analyzer_task = None
    
    async def analyze_scenario(
        self,
        scenario_name: str
    ) -> Optional[WhatIfResult]:
        """Analyze what-if variations of scenario."""
        if scenario_name not in self.planner.scenarios:
            return None
        
        # Generate variations
        variations = await self._generate_variations(scenario_name)
        
        # Analyze each variation
        for variation in variations:
            # Apply parameter changes
            modified_scenario = await self._modify_scenario(
                self.planner.scenarios[scenario_name],
                variation.parameter_changes
            )
            
            # Analyze modified scenario
            await self.planner._analyze_scenario(modified_scenario)
            
            # Store outcome
            if modified_scenario.name in self.planner.outcomes:
                variation.outcome = self.planner.outcomes[modified_scenario.name]
                
                # Calculate impact delta
                base_impact = (
                    self.planner.outcomes[scenario_name].total_impact
                    if scenario_name in self.planner.outcomes else 0.0
                )
                variation.impact_delta = (
                    variation.outcome.total_impact - base_impact
                )
                
                # Calculate success probability
                variation.success_probability = variation.outcome.success_rate
                
                # Extract risk factors
                variation.risk_factors = {
                    metric: score
                    for metric, score in variation.outcome.sensitivity_scores.items()
                }
        
        # Find optimal parameters
        optimal_params = await self._optimize_parameters(variations)
        
        # Calculate sensitivity ranking
        sensitivity = await self._analyze_sensitivity(variations)
        
        # Generate correlation matrix
        correlations = await self._analyze_correlations(variations)
        
        # Generate insights
        insights = await self._generate_insights(
            variations,
            optimal_params,
            sensitivity
        )
        
        # Store results
        result = WhatIfResult(
            variations=variations,
            optimal_parameters=optimal_params,
            sensitivity_ranking=sensitivity,
            correlation_matrix=correlations,
            key_insights=insights
        )
        
        self.variations[scenario_name] = variations
        self.results[scenario_name] = result
        
        return result
    
    async def _generate_variations(
        self,
        scenario_name: str
    ) -> List[WhatIfVariation]:
        """Generate parameter variations."""
        variations = []
        
        # Generate single parameter variations
        for metric, (min_val, max_val) in self.config.variation_ranges.items():
            step = self.config.step_sizes[metric]
            values = np.arange(min_val, max_val + step, step)
            
            for value in values:
                variations.append(WhatIfVariation(
                    base_scenario=scenario_name,
                    parameter_changes={metric: value}
                ))
        
        # Generate combinations if within limit
        if len(variations) < self.config.max_combinations:
            # Add selected combinations
            metrics = list(self.config.variation_ranges.keys())
            for i, metric1 in enumerate(metrics[:-1]):
                for metric2 in metrics[i+1:]:
                    step1 = self.config.step_sizes[metric1]
                    step2 = self.config.step_sizes[metric2]
                    
                    for v1 in np.arange(
                        self.config.variation_ranges[metric1][0],
                        self.config.variation_ranges[metric1][1] + step1,
                        step1 * 2
                    ):
                        for v2 in np.arange(
                            self.config.variation_ranges[metric2][0],
                            self.config.variation_ranges[metric2][1] + step2,
                            step2 * 2
                        ):
                            variations.append(WhatIfVariation(
                                base_scenario=scenario_name,
                                parameter_changes={
                                    metric1: v1,
                                    metric2: v2
                                }
                            ))
                            
                            if len(variations) >= self.config.max_variations:
                                break
                        
                        if len(variations) >= self.config.max_variations:
                            break
                    
                    if len(variations) >= self.config.max_variations:
                        break
                
                if len(variations) >= self.config.max_variations:
                    break
        
        return variations[:self.config.max_variations]
    
    async def _modify_scenario(
        self,
        base_scenario: Scenario,
        changes: Dict[str, float]
    ) -> Scenario:
        """Create modified version of scenario."""
        new_conditions = base_scenario.conditions.copy()
        
        # Modify or add conditions
        for metric, value in changes.items():
            found = False
            for condition in new_conditions:
                if condition.metric == metric:
                    condition.value = value
                    found = True
                    break
            
            if not found:
                new_conditions.append(ScenarioCondition(
                    metric=metric,
                    operator="eq",
                    value=value
                ))
        
        return Scenario(
            name=f"{base_scenario.name}_variation_{len(self.variations.get(base_scenario.name, []))}",
            description=f"Variation of {base_scenario.name}",
            conditions=new_conditions
        )
    
    async def _optimize_parameters(
        self,
        variations: List[WhatIfVariation]
    ) -> Dict[str, float]:
        """Find optimal parameter values."""
        if not variations:
            return {}
        
        # Prepare training data
        X = []
        y = []
        
        for var in variations:
            if not var.outcome:
                continue
            
            features = []
            for metric in self.config.variation_ranges:
                features.append(var.parameter_changes.get(metric, 0.0))
            
            X.append(features)
            
            # Get target metric
            if self.config.optimization_target == "total_impact":
                target = var.outcome.total_impact
            elif self.config.optimization_target == "success_rate":
                target = var.outcome.success_rate
            else:
                target = var.outcome.total_impact
            
            y.append(target)
        
        if not X or not y:
            return {}
        
        # Train predictor
        self.predictor.fit(X, y)
        
        # Find optimal parameters
        best_params = {}
        best_score = float("-inf")
        
        for _ in range(self.config.simulation_runs):
            # Random parameter set
            test_params = {}
            for metric, (min_val, max_val) in self.config.variation_ranges.items():
                test_params[metric] = np.random.uniform(min_val, max_val)
            
            # Predict score
            features = [
                test_params.get(metric, 0.0)
                for metric in self.config.variation_ranges
            ]
            score = self.predictor.predict([features])[0]
            
            # Check constraints if enabled
            if self.config.enable_constraints:
                valid = True
                for var in variations:
                    if not var.outcome:
                        continue
                    
                    for metric, (min_val, max_val) in self.config.constraints.items():
                        if metric == "success_rate":
                            value = var.outcome.success_rate
                        elif metric == "recovery_time":
                            value = var.outcome.recovery_time
                        else:
                            continue
                        
                        if not min_val <= value <= max_val:
                            valid = False
                            break
                    
                    if not valid:
                        break
                
                if not valid:
                    continue
            
            # Update best parameters
            if score > best_score:
                best_score = score
                best_params = test_params
        
        return best_params
    
    async def _analyze_sensitivity(
        self,
        variations: List[WhatIfVariation]
    ) -> Dict[str, float]:
        """Analyze parameter sensitivity."""
        sensitivity = {}
        
        if not variations:
            return sensitivity
        
        # Group variations by parameter
        by_parameter = defaultdict(list)
        for var in variations:
            if not var.outcome:
                continue
            
            for param, value in var.parameter_changes.items():
                by_parameter[param].append((
                    value,
                    var.outcome.total_impact
                ))
        
        # Calculate sensitivity for each parameter
        for param, values in by_parameter.items():
            if not values:
                continue
            
            x = [v[0] for v in values]
            y = [v[1] for v in values]
            
            # Calculate correlation
            correlation, _ = stats.pearsonr(x, y)
            
            # Calculate variability
            std = np.std(y)
            
            # Combine metrics
            sensitivity[param] = abs(correlation) * std
        
        # Normalize
        total = sum(sensitivity.values()) or 1.0
        return {
            k: v / total
            for k, v in sensitivity.items()
        }
    
    async def _analyze_correlations(
        self,
        variations: List[WhatIfVariation]
    ) -> pd.DataFrame:
        """Analyze parameter correlations."""
        if not variations:
            return pd.DataFrame()
        
        # Extract parameters and metrics
        data = []
        for var in variations:
            if not var.outcome:
                continue
            
            row = {}
            row.update(var.parameter_changes)
            row["total_impact"] = var.outcome.total_impact
            row["success_rate"] = var.outcome.success_rate
            row["recovery_time"] = var.outcome.recovery_time
            
            data.append(row)
        
        if not data:
            return pd.DataFrame()
        
        # Create correlation matrix
        df = pd.DataFrame(data)
        return df.corr()
    
    async def _generate_insights(
        self,
        variations: List[WhatIfVariation],
        optimal_params: Dict[str, float],
        sensitivity: Dict[str, float]
    ) -> List[str]:
        """Generate key insights from analysis."""
        insights = []
        
        if not variations:
            return insights
        
        # Add sensitivity insights
        for param, score in sorted(
            sensitivity.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            insights.append(
                f"Parameter {param} has {score:.1%} influence on outcomes"
            )
        
        # Add optimal parameter insights
        if optimal_params:
            insights.append("\nOptimal parameters:")
            for param, value in optimal_params.items():
                insights.append(f"  {param}: {value:.2f}")
        
        # Add performance insights
        best_variation = max(
            [v for v in variations if v.outcome],
            key=lambda x: x.outcome.total_impact,
            default=None
        )
        
        if best_variation:
            insights.append(
                f"\nBest variation achieves {best_variation.outcome.total_impact:.1%} "
                f"impact with {best_variation.success_probability:.1%} success rate"
            )
        
        # Add risk insights
        high_risk_variations = [
            v for v in variations
            if v.outcome and v.outcome.total_impact > 0.8
        ]
        
        if high_risk_variations:
            avg_recovery = np.mean([
                v.outcome.recovery_time for v in high_risk_variations
            ])
            insights.append(
                f"\nHigh-impact variations require {avg_recovery:.1f}s "
                "average recovery time"
            )
        
        return insights
    
    async def create_whatif_plots(self) -> Dict[str, go.Figure]:
        """Create what-if visualization plots."""
        plots = {}
        
        for scenario_name, variations in self.variations.items():
            if not variations:
                continue
            
            # Parameter impact plot
            impact_fig = go.Figure()
            
            for param in self.config.variation_ranges:
                param_vars = [
                    v for v in variations
                    if param in v.parameter_changes and v.outcome
                ]
                if not param_vars:
                    continue
                
                impact_fig.add_trace(go.Scatter(
                    x=[v.parameter_changes[param] for v in param_vars],
                    y=[v.outcome.total_impact for v in param_vars],
                    name=param,
                    mode="lines+markers"
                ))
            
            impact_fig.update_layout(
                title=f"Parameter Impact Analysis - {scenario_name}",
                xaxis_title="Parameter Value",
                yaxis_title="Total Impact",
                showlegend=True
            )
            plots[f"{scenario_name}_impact"] = impact_fig
            
            # Trade-off plot
            tradeoff_fig = go.Figure()
            
            valid_vars = [v for v in variations if v.outcome]
            if valid_vars:
                tradeoff_fig.add_trace(go.Scatter(
                    x=[v.outcome.success_rate for v in valid_vars],
                    y=[v.outcome.total_impact for v in valid_vars],
                    mode="markers",
                    text=[
                        f"Parameters: {v.parameter_changes}"
                        for v in valid_vars
                    ],
                    marker=dict(
                        size=[
                            v.outcome.recovery_time / 10 + 5
                            for v in valid_vars
                        ],
                        color=[
                            sum(v.risk_factors.values())
                            for v in valid_vars
                        ],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Risk Score")
                    )
                ))
            
            tradeoff_fig.update_layout(
                title=f"Performance Trade-offs - {scenario_name}",
                xaxis_title="Success Rate",
                yaxis_title="Total Impact",
                showlegend=False
            )
            plots[f"{scenario_name}_tradeoff"] = tradeoff_fig
            
            # Correlation heatmap
            if scenario_name in self.results:
                result = self.results[scenario_name]
                
                corr_fig = go.Figure(data=go.Heatmap(
                    z=result.correlation_matrix.values,
                    x=result.correlation_matrix.columns,
                    y=result.correlation_matrix.index,
                    colorscale="RdBu",
                    zmid=0
                ))
                
                corr_fig.update_layout(
                    title=f"Parameter Correlations - {scenario_name}",
                    xaxis_title="Parameter",
                    yaxis_title="Parameter"
                )
                plots[f"{scenario_name}_correlation"] = corr_fig
        
        # Save plots
        if self.config.visualization_dir:
            from pathlib import Path
            path = Path(self.config.visualization_dir)
            path.mkdir(parents=True, exist_ok=True)
            for name, fig in plots.items():
                fig.write_html(str(path / f"whatif_{name}.html"))
        
        return plots

def create_whatif_analyzer(
    planner: ScenarioPlanner,
    config: Optional[WhatIfConfig] = None
) -> WhatIfAnalyzer:
    """Create what-if analyzer."""
    return WhatIfAnalyzer(planner, config)

if __name__ == "__main__":
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
        
        # Start components
        await prevention.start_prevention()
        await balancer.start_balancing()
        await advisor.start_advisor()
        await analyzer.start_analyzer()
        await predictor.start_predictor()
        await planner.start_planner()
        await whatif.start_analyzer()
        
        try:
            # Add test scenarios
            test_scenario = Scenario(
                name="resource_pressure",
                description="Resource pressure scenario",
                conditions=[
                    ScenarioCondition(
                        metric="memory_impact",
                        operator="gt",
                        value=0.7
                    ),
                    ScenarioCondition(
                        metric="cpu_impact", 
                        operator="gt",
                        value=0.6
                    )
                ]
            )
            
            await planner.add_scenario(test_scenario)
            
            while True:
                # Analyze scenarios
                for scenario in planner.scenarios:
                    result = await whatif.analyze_scenario(scenario)
                    if result:
                        print(f"\nWhat-if Analysis for {scenario}:")
                        print("\nKey Insights:")
                        for insight in result.key_insights:
                            print(insight)
                        
                        print("\nSensitivity Ranking:")
                        for param, score in sorted(
                            result.sensitivity_ranking.items(),
                            key=lambda x: x[1],
                            reverse=True
                        ):
                            print(f"  {param}: {score:.2%}")
                
                # Create plots
                await whatif.create_whatif_plots()
                
                await asyncio.sleep(60)
        finally:
            await whatif.stop_analyzer()
            await planner.stop_planner()
            await predictor.stop_predictor()
            await analyzer.stop_analyzer()
            await advisor.stop_advisor()
            await balancer.stop_balancing()
            await prevention.stop_prevention()
    
    asyncio.run(main())
