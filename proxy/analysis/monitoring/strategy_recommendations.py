"""Strategy recommendations for leak prevention."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .prevention_balancing import PreventionBalancer, BalancingConfig, StrategyMetrics
from .leak_prevention import LeakPrevention, PreventionConfig

@dataclass
class RecommendationConfig:
    """Configuration for strategy recommendations."""
    enabled: bool = True
    update_interval: float = 600.0  # 10 minutes
    min_history: int = 20
    confidence_threshold: float = 0.95
    impact_threshold: float = 0.1
    max_recommendations: int = 5
    enable_auto_apply: bool = False
    lookback_window: timedelta = timedelta(hours=24)
    visualization_dir: Optional[str] = "recommendations"

@dataclass
class StrategyRecommendation:
    """Recommendation for strategy adjustment."""
    strategy: str
    action: str  # increase, decrease, disable, enable
    confidence: float
    impact: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    applied: bool = False

class StrategyAdvisor:
    """Generate recommendations for strategy optimization."""
    
    def __init__(
        self,
        balancer: PreventionBalancer,
        config: RecommendationConfig = None
    ):
        self.balancer = balancer
        self.config = config or RecommendationConfig()
        
        # Recommendation storage
        self.recommendations: List[StrategyRecommendation] = []
        self.last_update = datetime.min
        self.advisor_task: Optional[asyncio.Task] = None
        
        # Analysis state
        self.trend_cache: Dict[str, Dict[str, Any]] = {}
        self.correlation_cache: Dict[Tuple[str, str], float] = {}
    
    async def start_advisor(self):
        """Start recommendation advisor."""
        if not self.config.enabled:
            return
        
        if self.advisor_task is None:
            self.advisor_task = asyncio.create_task(self._run_advisor())
    
    async def stop_advisor(self):
        """Stop recommendation advisor."""
        if self.advisor_task:
            self.advisor_task.cancel()
            try:
                await self.advisor_task
            except asyncio.CancelledError:
                pass
            self.advisor_task = None
    
    async def _run_advisor(self):
        """Run periodic advisor."""
        while True:
            try:
                current_time = datetime.now()
                
                if (
                    current_time - self.last_update >=
                    timedelta(seconds=self.config.update_interval)
                ):
                    await self._generate_recommendations()
                    if self.config.enable_auto_apply:
                        await self._apply_recommendations()
                    
                    self.last_update = current_time
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                print(f"Advisor error: {e}")
                await asyncio.sleep(60)
    
    async def _generate_recommendations(self):
        """Generate strategy recommendations."""
        if len(self.balancer.history) < self.config.min_history:
            return
        
        recommendations = []
        
        # Analyze trends
        trends = await self._analyze_trends()
        for strategy, trend in trends.items():
            if trend["confidence"] >= self.config.confidence_threshold:
                if trend["slope"] < 0 and trend["impact"] >= self.config.impact_threshold:
                    # Strategy is declining in effectiveness
                    recommendations.append(StrategyRecommendation(
                        strategy=strategy,
                        action="decrease",
                        confidence=trend["confidence"],
                        impact=trend["impact"],
                        reason=f"Declining effectiveness: {trend['slope']:.2%} per hour"
                    ))
                elif trend["slope"] > 0 and trend["performance"] > 0.8:
                    # Strategy is performing well
                    recommendations.append(StrategyRecommendation(
                        strategy=strategy,
                        action="increase",
                        confidence=trend["confidence"],
                        impact=trend["impact"],
                        reason=f"High performance: {trend['performance']:.2%}"
                    ))
        
        # Analyze correlations
        correlations = await self._analyze_correlations()
        for (strategy1, strategy2), corr in correlations.items():
            if abs(corr) >= 0.8:
                # Strong correlation suggests redundancy
                metric1 = self.balancer.strategy_metrics[strategy1]
                metric2 = self.balancer.strategy_metrics[strategy2]
                
                if metric1.success_rate > metric2.success_rate:
                    better, worse = strategy1, strategy2
                else:
                    better, worse = strategy2, strategy1
                
                recommendations.append(StrategyRecommendation(
                    strategy=worse,
                    action="disable",
                    confidence=abs(corr),
                    impact=self.balancer.strategy_weights[worse],
                    reason=f"Redundant with {better} (correlation: {corr:.2f})"
                ))
        
        # Check resource impact
        stats = self.balancer.get_strategy_stats()
        for strategy, data in stats.items():
            metrics = data["metrics"]
            if metrics["cpu_overhead"] > 0.3:  # 30% CPU usage
                recommendations.append(StrategyRecommendation(
                    strategy=strategy,
                    action="decrease",
                    confidence=0.9,
                    impact=metrics["cpu_overhead"],
                    reason=f"High CPU usage: {metrics['cpu_overhead']:.1%}"
                ))
        
        # Sort and filter recommendations
        recommendations.sort(key=lambda r: r.impact * r.confidence, reverse=True)
        self.recommendations = recommendations[:self.config.max_recommendations]
    
    async def _analyze_trends(self) -> Dict[str, Dict[str, Any]]:
        """Analyze strategy performance trends."""
        current_time = datetime.now()
        cutoff_time = current_time - self.config.lookback_window
        
        # Get relevant history
        history = [
            h for h in self.balancer.history
            if h.timestamp >= cutoff_time
        ]
        
        if not history:
            return {}
        
        trends = {}
        for strategy in self.balancer.strategy_weights:
            # Extract metrics
            times = [
                (h.timestamp - cutoff_time).total_seconds() / 3600  # hours
                for h in history
            ]
            metrics = [h.metrics[strategy] for h in history]
            success_rates = [m.success_rate for m in metrics]
            memory_savings = [m.memory_savings for m in metrics]
            
            # Fit trend lines
            success_slope, _, r_value, p_value, _ = stats.linregress(
                times,
                success_rates
            )
            memory_slope, _, _, _, _ = stats.linregress(
                times,
                memory_savings
            )
            
            # Calculate impact
            impact = (
                np.mean(success_rates) *
                np.mean(memory_savings) *
                self.balancer.strategy_weights[strategy]
            )
            
            trends[strategy] = {
                "slope": success_slope,
                "confidence": 1 - p_value,
                "correlation": r_value ** 2,
                "impact": impact,
                "performance": np.mean(success_rates),
                "memory_slope": memory_slope
            }
        
        return trends
    
    async def _analyze_correlations(
        self
    ) -> Dict[Tuple[str, str], float]:
        """Analyze correlations between strategies."""
        correlations = {}
        strategies = list(self.balancer.strategy_weights.keys())
        
        for i, strategy1 in enumerate(strategies):
            metrics1 = [
                h.metrics[strategy1].success_rate
                for h in self.balancer.history
            ]
            
            for strategy2 in strategies[i+1:]:
                metrics2 = [
                    h.metrics[strategy2].success_rate
                    for h in self.balancer.history
                ]
                
                correlation, _ = stats.pearsonr(metrics1, metrics2)
                correlations[(strategy1, strategy2)] = correlation
        
        return correlations
    
    async def _apply_recommendations(self):
        """Apply recommended changes."""
        if not self.recommendations:
            return
        
        for recommendation in self.recommendations:
            if recommendation.applied:
                continue
            
            strategy = recommendation.strategy
            current_weight = self.balancer.strategy_weights[strategy]
            
            if recommendation.action == "increase":
                new_weight = min(1.0, current_weight * 1.2)
            elif recommendation.action == "decrease":
                new_weight = max(0.1, current_weight * 0.8)
            elif recommendation.action == "disable":
                new_weight = 0.0
            elif recommendation.action == "enable":
                new_weight = max(0.1, current_weight)
            else:
                continue
            
            # Update weight
            self.balancer.strategy_weights[strategy] = new_weight
            recommendation.applied = True
    
    def get_recommendations(
        self,
        limit: Optional[int] = None
    ) -> List[StrategyRecommendation]:
        """Get current recommendations."""
        recs = [r for r in self.recommendations if not r.applied]
        if limit:
            recs = recs[:limit]
        return recs
    
    async def create_recommendation_plots(self) -> Dict[str, go.Figure]:
        """Create recommendation visualization plots."""
        plots = {}
        
        if not self.balancer.history:
            return plots
        
        # Strategy effectiveness plot
        effectiveness_fig = go.Figure()
        
        df = pd.DataFrame([
            {
                "timestamp": h.timestamp,
                **{
                    f"{s}_success": h.metrics[s].success_rate
                    for s in self.balancer.strategy_weights
                }
            }
            for h in self.balancer.history
        ])
        
        for strategy in self.balancer.strategy_weights:
            effectiveness_fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[f"{strategy}_success"],
                    name=strategy,
                    mode="lines"
                )
            )
        
        effectiveness_fig.update_layout(
            title="Strategy Effectiveness Over Time",
            xaxis_title="Time",
            yaxis_title="Success Rate",
            showlegend=True
        )
        plots["effectiveness"] = effectiveness_fig
        
        # Correlation matrix plot
        correlations = await self._analyze_correlations()
        strategies = list(self.balancer.strategy_weights.keys())
        matrix = np.zeros((len(strategies), len(strategies)))
        
        for i, s1 in enumerate(strategies):
            for j, s2 in enumerate(strategies):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    key = (s1, s2) if (s1, s2) in correlations else (s2, s1)
                    matrix[i][j] = correlations.get(key, 0.0)
        
        correlation_fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=strategies,
            y=strategies,
            colorscale="RdBu",
            zmid=0
        ))
        
        correlation_fig.update_layout(
            title="Strategy Correlation Matrix",
            xaxis_title="Strategy",
            yaxis_title="Strategy"
        )
        plots["correlations"] = correlation_fig
        
        # Impact plot
        trends = await self._analyze_trends()
        impact_fig = go.Figure()
        
        strategies = list(trends.keys())
        impacts = [trends[s]["impact"] for s in strategies]
        confidences = [trends[s]["confidence"] for s in strategies]
        
        impact_fig.add_trace(go.Bar(
            x=strategies,
            y=impacts,
            name="Impact",
            marker_color="blue"
        ))
        
        impact_fig.add_trace(go.Scatter(
            x=strategies,
            y=confidences,
            name="Confidence",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="red")
        ))
        
        impact_fig.update_layout(
            title="Strategy Impact and Confidence",
            xaxis_title="Strategy",
            yaxis_title="Impact",
            yaxis2=dict(
                title="Confidence",
                overlaying="y",
                side="right"
            ),
            showlegend=True
        )
        plots["impact"] = impact_fig
        
        # Save plots
        if self.config.visualization_dir:
            from pathlib import Path
            path = Path(self.config.visualization_dir)
            path.mkdir(parents=True, exist_ok=True)
            for name, fig in plots.items():
                fig.write_html(str(path / f"recommendations_{name}.html"))
        
        return plots

def create_strategy_advisor(
    balancer: PreventionBalancer,
    config: Optional[RecommendationConfig] = None
) -> StrategyAdvisor:
    """Create strategy advisor."""
    return StrategyAdvisor(balancer, config)

if __name__ == "__main__":
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
        
        # Start components
        await prevention.start_prevention()
        await balancer.start_balancing()
        await advisor.start_advisor()
        
        try:
            while True:
                # Get recommendations
                recommendations = advisor.get_recommendations()
                if recommendations:
                    print("\nStrategy Recommendations:")
                    for rec in recommendations:
                        print(
                            f"\n{rec.strategy}:"
                            f"\n  Action: {rec.action}"
                            f"\n  Confidence: {rec.confidence:.2%}"
                            f"\n  Impact: {rec.impact:.2%}"
                            f"\n  Reason: {rec.reason}"
                        )
                
                # Create plots
                await advisor.create_recommendation_plots()
                
                await asyncio.sleep(60)
        finally:
            await advisor.stop_advisor()
            await balancer.stop_balancing()
            await prevention.stop_prevention()
    
    asyncio.run(main())
