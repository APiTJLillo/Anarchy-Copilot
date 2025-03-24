"""Performance balancing for leak prevention strategies."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .leak_prevention import LeakPrevention, PreventionConfig, PreventionStats
from .memory_leak_detection import LeakDetector

@dataclass
class BalancingConfig:
    """Configuration for prevention balancing."""
    enabled: bool = True
    update_interval: float = 300.0  # 5 minutes
    learning_rate: float = 0.1
    min_samples: int = 10
    max_history: int = 1000
    optimization_window: timedelta = timedelta(hours=1)
    enable_auto_tuning: bool = True
    performance_weight: float = 0.6
    stability_weight: float = 0.4
    min_strategy_weight: float = 0.1
    strategy_cooldown: timedelta = timedelta(minutes=5)
    visualization_dir: Optional[str] = "balancing_results"

@dataclass
class StrategyMetrics:
    """Metrics for prevention strategy."""
    success_rate: float = 0.0
    memory_savings: float = 0.0
    execution_time: float = 0.0
    cpu_overhead: float = 0.0
    stability_score: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class BalancingHistory:
    """Historical data for balancing."""
    timestamp: datetime
    strategy_weights: Dict[str, float]
    metrics: Dict[str, StrategyMetrics]
    performance_score: float
    stability_score: float

class PreventionBalancer:
    """Balance and optimize leak prevention strategies."""
    
    def __init__(
        self,
        prevention: LeakPrevention,
        config: BalancingConfig = None
    ):
        self.prevention = prevention
        self.config = config or BalancingConfig()
        
        # Strategy state
        self.strategy_weights: Dict[str, float] = {
            strategy: 1.0
            for strategy in self.prevention.config.mitigation_strategies
        }
        
        self.strategy_metrics: Dict[str, StrategyMetrics] = {
            strategy: StrategyMetrics()
            for strategy in self.prevention.config.mitigation_strategies
        }
        
        # History tracking
        self.history: List[BalancingHistory] = []
        self.last_update = datetime.min
        self.balancing_task: Optional[asyncio.Task] = None
    
    async def start_balancing(self):
        """Start balancing task."""
        if not self.config.enabled:
            return
        
        if self.balancing_task is None:
            self.balancing_task = asyncio.create_task(self._run_balancing())
    
    async def stop_balancing(self):
        """Stop balancing task."""
        if self.balancing_task:
            self.balancing_task.cancel()
            try:
                await self.balancing_task
            except asyncio.CancelledError:
                pass
            self.balancing_task = None
    
    async def _run_balancing(self):
        """Run periodic balancing."""
        while True:
            try:
                current_time = datetime.now()
                
                if (
                    current_time - self.last_update >=
                    timedelta(seconds=self.config.update_interval)
                ):
                    await self._update_metrics()
                    if self.config.enable_auto_tuning:
                        await self._optimize_weights()
                    
                    self.last_update = current_time
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                print(f"Balancing error: {e}")
                await asyncio.sleep(60)
    
    async def _update_metrics(self):
        """Update strategy metrics."""
        stats = self.prevention.stats
        
        for strategy in self.strategy_metrics:
            metrics = self.strategy_metrics[strategy]
            count = stats.mitigation_counts.get(strategy, 0)
            
            if count > 0:
                # Calculate success rate
                success_rate = min(
                    1.0,
                    stats.memory_freed /
                    (count * stats.total_size if stats.total_size else 1)
                )
                
                # Calculate memory savings
                memory_savings = (
                    stats.memory_freed / count
                    if count else 0
                )
                
                # Calculate execution overhead
                execution_time = (
                    stats.collection_time / count
                    if count else 0
                )
                
                # Update metrics with exponential moving average
                lr = self.config.learning_rate
                metrics.success_rate = (
                    (1 - lr) * metrics.success_rate +
                    lr * success_rate
                )
                metrics.memory_savings = (
                    (1 - lr) * metrics.memory_savings +
                    lr * memory_savings
                )
                metrics.execution_time = (
                    (1 - lr) * metrics.execution_time +
                    lr * execution_time
                )
                metrics.cpu_overhead = (
                    (1 - lr) * metrics.cpu_overhead +
                    lr * (stats.collection_time / self.config.update_interval)
                )
                
                # Calculate stability score
                metrics.stability_score = self._calculate_stability(strategy)
                metrics.last_update = datetime.now()
    
    def _calculate_stability(
        self,
        strategy: str
    ) -> float:
        """Calculate strategy stability score."""
        if len(self.history) < self.config.min_samples:
            return 0.0
        
        # Get recent metrics
        recent = [
            h.metrics[strategy].success_rate
            for h in self.history[-self.config.min_samples:]
        ]
        
        # Calculate stability based on variance
        return 1.0 - min(1.0, np.std(recent))
    
    async def _optimize_weights(self):
        """Optimize strategy weights."""
        if len(self.history) < self.config.min_samples:
            return
        
        # Prepare optimization data
        data = pd.DataFrame([
            {
                "strategy": strategy,
                "success_rate": metrics.success_rate,
                "memory_savings": metrics.memory_savings,
                "execution_time": metrics.execution_time,
                "cpu_overhead": metrics.cpu_overhead,
                "stability": metrics.stability_score
            }
            for strategy, metrics in self.strategy_metrics.items()
        ])
        
        # Define objective function
        def objective(weights):
            # Performance score
            performance = np.sum(
                weights * (
                    data["success_rate"] * data["memory_savings"] /
                    (data["execution_time"] * data["cpu_overhead"])
                )
            )
            
            # Stability score
            stability = np.sum(weights * data["stability"])
            
            # Combined score
            return -(
                self.config.performance_weight * performance +
                self.config.stability_weight * stability
            )
        
        # Define constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},  # Sum to 1
            {"type": "ineq", "fun": lambda x: x - self.config.min_strategy_weight}  # Min weight
        ]
        
        # Initial weights
        x0 = np.array(list(self.strategy_weights.values()))
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            constraints=constraints,
            bounds=[(self.config.min_strategy_weight, 1.0)] * len(self.strategy_weights)
        )
        
        if result.success:
            # Update weights
            new_weights = {
                strategy: float(weight)
                for strategy, weight in zip(self.strategy_weights.keys(), result.x)
            }
            
            # Apply cooldown
            current_time = datetime.now()
            for strategy, metrics in self.strategy_metrics.items():
                if (
                    current_time - metrics.last_update <
                    self.config.strategy_cooldown
                ):
                    new_weights[strategy] = self.strategy_weights[strategy]
            
            self.strategy_weights = new_weights
            
            # Update prevention config
            self.prevention.config.mitigation_strategies = {
                strategy
                for strategy, weight in self.strategy_weights.items()
                if weight > self.config.min_strategy_weight
            }
            
            # Store history
            self.history.append(BalancingHistory(
                timestamp=datetime.now(),
                strategy_weights=self.strategy_weights.copy(),
                metrics={
                    strategy: StrategyMetrics(**metrics.__dict__)
                    for strategy, metrics in self.strategy_metrics.items()
                },
                performance_score=-result.fun * self.config.performance_weight,
                stability_score=-result.fun * self.config.stability_weight
            ))
            
            # Trim history
            while len(self.history) > self.config.max_history:
                self.history.pop(0)
    
    def get_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get strategy statistics."""
        return {
            strategy: {
                "weight": self.strategy_weights[strategy],
                "metrics": self.strategy_metrics[strategy].__dict__
            }
            for strategy in self.strategy_weights
        }
    
    async def create_balancing_plots(self) -> Dict[str, go.Figure]:
        """Create balancing visualization plots."""
        plots = {}
        
        if not self.history:
            return plots
        
        # Strategy weights plot
        weights_fig = go.Figure()
        
        df = pd.DataFrame([
            {
                "timestamp": h.timestamp,
                **h.strategy_weights
            }
            for h in self.history
        ])
        
        for strategy in self.strategy_weights:
            weights_fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[strategy],
                    name=strategy,
                    mode="lines"
                )
            )
        
        weights_fig.update_layout(
            title="Strategy Weights Over Time",
            xaxis_title="Time",
            yaxis_title="Weight",
            showlegend=True
        )
        plots["weights"] = weights_fig
        
        # Performance metrics plot
        metrics_fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Success Rate",
                "Memory Savings",
                "Execution Time",
                "CPU Overhead"
            ]
        )
        
        for strategy in self.strategy_weights:
            metrics = [h.metrics[strategy] for h in self.history]
            timestamps = [h.timestamp for h in self.history]
            
            metrics_fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[m.success_rate for m in metrics],
                    name=f"{strategy} Success",
                    line=dict(dash="solid")
                ),
                row=1,
                col=1
            )
            
            metrics_fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[m.memory_savings for m in metrics],
                    name=f"{strategy} Memory",
                    line=dict(dash="solid")
                ),
                row=1,
                col=2
            )
            
            metrics_fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[m.execution_time for m in metrics],
                    name=f"{strategy} Time",
                    line=dict(dash="solid")
                ),
                row=2,
                col=1
            )
            
            metrics_fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[m.cpu_overhead for m in metrics],
                    name=f"{strategy} CPU",
                    line=dict(dash="solid")
                ),
                row=2,
                col=2
            )
        
        metrics_fig.update_layout(
            height=800,
            showlegend=True,
            title="Strategy Performance Metrics"
        )
        plots["metrics"] = metrics_fig
        
        # Score plot
        score_fig = go.Figure()
        
        score_fig.add_trace(
            go.Scatter(
                x=[h.timestamp for h in self.history],
                y=[h.performance_score for h in self.history],
                name="Performance Score",
                line=dict(color="blue")
            )
        )
        
        score_fig.add_trace(
            go.Scatter(
                x=[h.timestamp for h in self.history],
                y=[h.stability_score for h in self.history],
                name="Stability Score",
                line=dict(color="green")
            )
        )
        
        score_fig.update_layout(
            title="Optimization Scores",
            xaxis_title="Time",
            yaxis_title="Score",
            showlegend=True
        )
        plots["scores"] = score_fig
        
        # Save plots
        if self.config.visualization_dir:
            from pathlib import Path
            for name, fig in plots.items():
                path = Path(self.config.visualization_dir)
                path.mkdir(parents=True, exist_ok=True)
                fig.write_html(str(path / f"balancing_{name}.html"))
        
        return plots

def create_prevention_balancer(
    prevention: LeakPrevention,
    config: Optional[BalancingConfig] = None
) -> PreventionBalancer:
    """Create prevention balancer."""
    return PreventionBalancer(prevention, config)

if __name__ == "__main__":
    from .leak_prevention import create_leak_prevention
    from .memory_leak_detection import create_leak_detector
    from .scheduler_profiling import create_profiling_hook
    
    async def main():
        # Setup components
        profiling = create_profiling_hook()
        detector = create_leak_detector(profiling)
        prevention = create_leak_prevention(detector)
        balancer = create_prevention_balancer(prevention)
        
        # Start components
        await prevention.start_prevention()
        await balancer.start_balancing()
        
        try:
            while True:
                # Print current weights
                print("\nStrategy Weights:")
                stats = balancer.get_strategy_stats()
                for strategy, data in stats.items():
                    print(
                        f"\n{strategy}:"
                        f"\n  Weight: {data['weight']:.3f}"
                        f"\n  Success Rate: {data['metrics']['success_rate']:.2%}"
                        f"\n  Memory Savings: {data['metrics']['memory_savings']:,} bytes"
                        f"\n  Execution Time: {data['metrics']['execution_time']:.3f}s"
                    )
                
                # Create plots
                await balancer.create_balancing_plots()
                
                await asyncio.sleep(60)
        finally:
            await balancer.stop_balancing()
            await prevention.stop_prevention()
    
    asyncio.run(main())
