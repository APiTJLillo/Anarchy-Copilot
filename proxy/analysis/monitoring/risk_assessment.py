"""Risk assessment for strategy changes."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .strategy_recommendations import (
    StrategyAdvisor, StrategyRecommendation, RecommendationConfig
)
from .prevention_balancing import PreventionBalancer

@dataclass
class RiskConfig:
    """Configuration for risk assessment."""
    enabled: bool = True
    update_interval: float = 300.0  # 5 minutes
    min_history: int = 50
    risk_threshold: float = 0.7
    confidence_interval: float = 0.95
    monte_carlo_samples: int = 1000
    max_impact_threshold: float = 0.3
    enable_rollback: bool = True
    rollback_threshold: float = 0.8
    visualization_dir: Optional[str] = "risk_analysis"

@dataclass
class RiskAssessment:
    """Risk assessment for strategy change."""
    recommendation: StrategyRecommendation
    risk_score: float
    impact_probability: float
    recovery_time: float
    failure_modes: List[str]
    mitigation_steps: List[str]
    rollback_plan: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ChangeImpact:
    """Impact analysis of strategy change."""
    memory_impact: float
    cpu_impact: float
    latency_impact: float
    stability_impact: float
    resource_conflicts: List[str]
    cascading_effects: List[str]

class RiskAnalyzer:
    """Analyze risks of strategy changes."""
    
    def __init__(
        self,
        advisor: StrategyAdvisor,
        config: RiskConfig = None
    ):
        self.advisor = advisor
        self.config = config or RiskConfig()
        
        # Analysis state 
        self.assessments: Dict[str, RiskAssessment] = {}
        self.impact_history: List[ChangeImpact] = []
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # Monitoring state
        self.last_update = datetime.min
        self.analyzer_task: Optional[asyncio.Task] = None
    
    async def start_analyzer(self):
        """Start risk analyzer."""
        if not self.config.enabled:
            return
        
        if self.analyzer_task is None:
            self.analyzer_task = asyncio.create_task(self._run_analyzer())
    
    async def stop_analyzer(self):
        """Stop risk analyzer."""
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
                    recommendations = self.advisor.get_recommendations()
                    for rec in recommendations:
                        await self._assess_risk(rec)
                    
                    self.last_update = current_time
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                print(f"Risk analyzer error: {e}")
                await asyncio.sleep(60)
    
    async def _assess_risk(
        self,
        recommendation: StrategyRecommendation
    ) -> RiskAssessment:
        """Assess risk of recommendation."""
        # Analyze historical impact
        impact = await self._analyze_impact(recommendation)
        
        # Calculate risk score
        risk_score = await self._calculate_risk_score(recommendation, impact)
        
        # Identify failure modes
        failure_modes = await self._identify_failure_modes(recommendation, impact)
        
        # Plan mitigation steps
        mitigation_steps = await self._plan_mitigation(recommendation, failure_modes)
        
        # Create rollback plan
        rollback_plan = await self._create_rollback_plan(recommendation)
        
        # Create assessment
        assessment = RiskAssessment(
            recommendation=recommendation,
            risk_score=risk_score,
            impact_probability=self._estimate_impact_probability(impact),
            recovery_time=self._estimate_recovery_time(impact),
            failure_modes=failure_modes,
            mitigation_steps=mitigation_steps,
            rollback_plan=rollback_plan
        )
        
        # Store assessment
        self.assessments[recommendation.strategy] = assessment
        
        return assessment
    
    async def _analyze_impact(
        self,
        recommendation: StrategyRecommendation
    ) -> ChangeImpact:
        """Analyze potential impact of change."""
        # Get historical data
        history = self.advisor.balancer.history[-self.config.min_history:]
        if not history:
            return ChangeImpact(
                memory_impact=0.0,
                cpu_impact=0.0,
                latency_impact=0.0,
                stability_impact=0.0,
                resource_conflicts=[],
                cascading_effects=[]
            )
        
        # Analyze memory impact
        memory_changes = [
            h.metrics[recommendation.strategy].memory_savings
            for h in history
        ]
        memory_impact = np.mean(memory_changes) * recommendation.impact
        
        # Analyze CPU impact
        cpu_changes = [
            h.metrics[recommendation.strategy].cpu_overhead
            for h in history
        ]
        cpu_impact = np.mean(cpu_changes) * recommendation.impact
        
        # Analyze latency impact
        latency_changes = [
            h.metrics[recommendation.strategy].execution_time
            for h in history
        ]
        latency_impact = np.mean(latency_changes) * recommendation.impact
        
        # Analyze stability
        stability_scores = [
            h.metrics[recommendation.strategy].stability_score
            for h in history
        ]
        stability_impact = 1.0 - np.mean(stability_scores)
        
        # Find resource conflicts
        resource_conflicts = await self._find_resource_conflicts(
            recommendation,
            cpu_impact,
            memory_impact
        )
        
        # Analyze cascading effects
        cascading_effects = await self._analyze_cascading_effects(
            recommendation,
            history
        )
        
        return ChangeImpact(
            memory_impact=memory_impact,
            cpu_impact=cpu_impact,
            latency_impact=latency_impact,
            stability_impact=stability_impact,
            resource_conflicts=resource_conflicts,
            cascading_effects=cascading_effects
        )
    
    async def _calculate_risk_score(
        self,
        recommendation: StrategyRecommendation,
        impact: ChangeImpact
    ) -> float:
        """Calculate overall risk score."""
        # Base risk from impact
        impact_risk = np.mean([
            impact.memory_impact,
            impact.cpu_impact,
            impact.latency_impact,
            impact.stability_impact
        ])
        
        # Adjust for resource conflicts
        resource_risk = len(impact.resource_conflicts) * 0.1
        
        # Adjust for cascading effects
        cascade_risk = len(impact.cascading_effects) * 0.15
        
        # Adjust for change magnitude
        change_risk = abs(recommendation.impact) * 0.2
        
        # Combine risks
        total_risk = (
            0.4 * impact_risk +
            0.2 * resource_risk +
            0.2 * cascade_risk +
            0.2 * change_risk
        )
        
        return min(1.0, total_risk)
    
    async def _find_resource_conflicts(
        self,
        recommendation: StrategyRecommendation,
        cpu_impact: float,
        memory_impact: float
    ) -> List[str]:
        """Find potential resource conflicts."""
        conflicts = []
        
        # Check CPU conflicts
        if cpu_impact > 0.5:  # 50% CPU usage
            conflicts.append("High CPU contention")
        
        # Check memory conflicts
        if memory_impact > 0.4:  # 40% memory usage
            conflicts.append("High memory pressure")
        
        # Check strategy conflicts
        for other_rec in self.advisor.get_recommendations():
            if other_rec.strategy != recommendation.strategy:
                if other_rec.action == recommendation.action:
                    conflicts.append(
                        f"Competing change with {other_rec.strategy}"
                    )
        
        return conflicts
    
    async def _analyze_cascading_effects(
        self,
        recommendation: StrategyRecommendation,
        history: List[Any]
    ) -> List[str]:
        """Analyze potential cascading effects."""
        effects = []
        
        # Check for correlated strategies
        correlations = await self.advisor._analyze_correlations()
        for (s1, s2), corr in correlations.items():
            if recommendation.strategy in (s1, s2) and abs(corr) > 0.6:
                other = s2 if recommendation.strategy == s1 else s1
                effects.append(f"Strong correlation with {other}")
        
        # Check for dependent strategies
        for strategy, metrics in self.advisor.balancer.strategy_metrics.items():
            if strategy != recommendation.strategy:
                if metrics.success_rate < 0.5:
                    effects.append(f"May impact struggling strategy {strategy}")
        
        # Check for timing effects
        if recommendation.action in ("disable", "decrease"):
            effects.append("May increase cleanup intervals")
        elif recommendation.action in ("enable", "increase"):
            effects.append("May increase resource consumption")
        
        return effects
    
    def _estimate_impact_probability(
        self,
        impact: ChangeImpact
    ) -> float:
        """Estimate probability of significant impact."""
        # Convert impact metrics to features
        features = np.array([[
            impact.memory_impact,
            impact.cpu_impact,
            impact.latency_impact,
            impact.stability_impact,
            len(impact.resource_conflicts),
            len(impact.cascading_effects)
        ]])
        
        # Use anomaly detection
        score = self.anomaly_detector.score_samples(features)[0]
        
        # Convert to probability (higher score = lower probability of impact)
        return 1.0 - stats.norm.cdf(score)
    
    def _estimate_recovery_time(
        self,
        impact: ChangeImpact
    ) -> float:
        """Estimate recovery time in seconds."""
        # Base recovery time
        base_time = 60.0  # 1 minute
        
        # Add time for resource conflicts
        base_time += len(impact.resource_conflicts) * 30.0
        
        # Add time for cascading effects
        base_time += len(impact.cascading_effects) * 60.0
        
        # Scale by impact severity
        severity = max(
            impact.memory_impact,
            impact.cpu_impact,
            impact.latency_impact,
            impact.stability_impact
        )
        
        return base_time * (1.0 + severity)
    
    async def _identify_failure_modes(
        self,
        recommendation: StrategyRecommendation,
        impact: ChangeImpact
    ) -> List[str]:
        """Identify potential failure modes."""
        modes = []
        
        # Resource exhaustion
        if impact.memory_impact > self.config.max_impact_threshold:
            modes.append("Memory exhaustion")
        if impact.cpu_impact > self.config.max_impact_threshold:
            modes.append("CPU saturation")
        
        # Performance degradation
        if impact.latency_impact > self.config.max_impact_threshold:
            modes.append("High latency")
        if impact.stability_impact > self.config.max_impact_threshold:
            modes.append("Stability loss")
        
        # Strategy conflicts
        for conflict in impact.resource_conflicts:
            modes.append(f"Resource conflict: {conflict}")
        
        # Cascading failures
        for effect in impact.cascading_effects:
            modes.append(f"Cascade risk: {effect}")
        
        return modes
    
    async def _plan_mitigation(
        self,
        recommendation: StrategyRecommendation,
        failure_modes: List[str]
    ) -> List[str]:
        """Plan mitigation steps."""
        steps = []
        
        # General precautions
        steps.append("Monitor system metrics closely")
        steps.append("Enable detailed logging")
        
        # Specific mitigations
        for mode in failure_modes:
            if "Memory" in mode:
                steps.append("Set memory limits")
                steps.append("Enable aggressive garbage collection")
            elif "CPU" in mode:
                steps.append("Set CPU quotas")
                steps.append("Enable load shedding")
            elif "latency" in mode:
                steps.append("Set timeout limits")
                steps.append("Enable circuit breakers")
            elif "Stability" in mode:
                steps.append("Enable automatic rollback triggers")
                steps.append("Set stability thresholds")
        
        # Add rollback preparation
        if self.config.enable_rollback:
            steps.append("Save current configuration")
            steps.append("Prepare rollback script")
        
        return steps
    
    async def _create_rollback_plan(
        self,
        recommendation: StrategyRecommendation
    ) -> Dict[str, Any]:
        """Create rollback plan."""
        return {
            "trigger_threshold": self.config.rollback_threshold,
            "wait_time": 300,  # 5 minutes
            "steps": [
                {
                    "action": "restore_weight",
                    "strategy": recommendation.strategy,
                    "original_weight": self.advisor.balancer.strategy_weights[
                        recommendation.strategy
                    ]
                },
                {
                    "action": "verify_metrics",
                    "timeout": 60
                },
                {
                    "action": "notify_status",
                    "channels": ["ops", "monitoring"]
                }
            ],
            "verification": {
                "required_metrics": [
                    "memory_used",
                    "cpu_used",
                    "success_rate"
                ],
                "acceptance_criteria": {
                    "memory_used": "< 85%",
                    "cpu_used": "< 75%",
                    "success_rate": "> 0.9"
                }
            }
        }
    
    async def create_risk_plots(self) -> Dict[str, go.Figure]:
        """Create risk visualization plots."""
        plots = {}
        
        if not self.assessments:
            return plots
        
        # Risk overview plot
        overview_fig = go.Figure()
        
        strategies = list(self.assessments.keys())
        risk_scores = [a.risk_score for a in self.assessments.values()]
        impact_probs = [a.impact_probability for a in self.assessments.values()]
        
        overview_fig.add_trace(go.Bar(
            x=strategies,
            y=risk_scores,
            name="Risk Score",
            marker_color="red"
        ))
        
        overview_fig.add_trace(go.Scatter(
            x=strategies,
            y=impact_probs,
            name="Impact Probability",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="blue")
        ))
        
        overview_fig.update_layout(
            title="Risk Assessment Overview",
            xaxis_title="Strategy",
            yaxis_title="Risk Score",
            yaxis2=dict(
                title="Impact Probability",
                overlaying="y",
                side="right"
            ),
            showlegend=True
        )
        plots["overview"] = overview_fig
        
        # Impact analysis plot
        impact_fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Memory Impact",
                "CPU Impact",
                "Latency Impact",
                "Stability Impact"
            ]
        )
        
        for assessment in self.assessments.values():
            impact = await self._analyze_impact(assessment.recommendation)
            
            impact_fig.add_trace(
                go.Bar(
                    x=[assessment.recommendation.strategy],
                    y=[impact.memory_impact],
                    name="Memory"
                ),
                row=1,
                col=1
            )
            
            impact_fig.add_trace(
                go.Bar(
                    x=[assessment.recommendation.strategy],
                    y=[impact.cpu_impact],
                    name="CPU"
                ),
                row=1,
                col=2
            )
            
            impact_fig.add_trace(
                go.Bar(
                    x=[assessment.recommendation.strategy],
                    y=[impact.latency_impact],
                    name="Latency"
                ),
                row=2,
                col=1
            )
            
            impact_fig.add_trace(
                go.Bar(
                    x=[assessment.recommendation.strategy],
                    y=[impact.stability_impact],
                    name="Stability"
                ),
                row=2,
                col=2
            )
        
        impact_fig.update_layout(
            height=800,
            showlegend=True,
            title="Impact Analysis"
        )
        plots["impact"] = impact_fig
        
        # Failure modes plot
        modes_fig = go.Figure()
        
        mode_counts = defaultdict(int)
        for assessment in self.assessments.values():
            for mode in assessment.failure_modes:
                mode_counts[mode] += 1
        
        modes_fig.add_trace(go.Bar(
            x=list(mode_counts.keys()),
            y=list(mode_counts.values()),
            name="Failure Modes"
        ))
        
        modes_fig.update_layout(
            title="Potential Failure Modes",
            xaxis_title="Failure Mode",
            yaxis_title="Count",
            showlegend=True
        )
        plots["failure_modes"] = modes_fig
        
        # Save plots
        if self.config.visualization_dir:
            from pathlib import Path
            path = Path(self.config.visualization_dir)
            path.mkdir(parents=True, exist_ok=True)
            for name, fig in plots.items():
                fig.write_html(str(path / f"risk_{name}.html"))
        
        return plots

def create_risk_analyzer(
    advisor: StrategyAdvisor,
    config: Optional[RiskConfig] = None
) -> RiskAnalyzer:
    """Create risk analyzer."""
    return RiskAnalyzer(advisor, config)

if __name__ == "__main__":
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
        
        # Start components
        await prevention.start_prevention()
        await balancer.start_balancing()
        await advisor.start_advisor()
        await analyzer.start_analyzer()
        
        try:
            while True:
                # Get current assessments
                for strategy, assessment in analyzer.assessments.items():
                    print(f"\nRisk Assessment for {strategy}:")
                    print(f"Risk Score: {assessment.risk_score:.2%}")
                    print(f"Impact Probability: {assessment.impact_probability:.2%}")
                    print(f"Recovery Time: {assessment.recovery_time:.1f}s")
                    print("\nFailure Modes:")
                    for mode in assessment.failure_modes:
                        print(f"- {mode}")
                    print("\nMitigation Steps:")
                    for step in assessment.mitigation_steps:
                        print(f"- {step}")
                
                # Create plots
                await analyzer.create_risk_plots()
                
                await asyncio.sleep(60)
        finally:
            await analyzer.stop_analyzer()
            await advisor.stop_advisor()
            await balancer.stop_balancing()
            await prevention.stop_prevention()
    
    asyncio.run(main())
