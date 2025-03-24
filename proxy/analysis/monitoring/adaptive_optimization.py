"""Adaptive optimization for validation filters."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import time
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import heapq

from .filter_optimization import FilterOptimizer, OptimizationConfig
from .validation_filters import FilterManager, ValidationFilter

logger = logging.getLogger(__name__)

@dataclass
class AdaptiveConfig:
    """Configuration for adaptive optimization."""
    learning_rate: float = 0.1
    min_samples: int = 100
    max_history: int = 1000
    update_interval: float = 60.0
    warm_up_period: float = 300.0
    auto_tune: bool = True
    decision_threshold: float = 0.1
    max_strategies: int = 5
    output_path: Optional[Path] = None

class PerformanceMetrics:
    """Track filter performance metrics."""
    
    def __init__(self):
        self.execution_times: List[float] = []
        self.throughput: List[float] = []
        self.cache_hits: List[int] = []
        self.result_counts: List[int] = []
        self.timestamps: List[datetime] = []
        self.strategies: List[str] = []
    
    def add_measurement(
        self,
        execution_time: float,
        count: int,
        cache_hit: int,
        strategy: str
    ):
        """Add performance measurement."""
        self.execution_times.append(execution_time)
        self.throughput.append(count / execution_time if execution_time > 0 else 0)
        self.cache_hits.append(cache_hit)
        self.result_counts.append(count)
        self.timestamps.append(datetime.now())
        self.strategies.append(strategy)
        
        # Trim history if needed
        if len(self.execution_times) > 1000:
            self.execution_times = self.execution_times[-1000:]
            self.throughput = self.throughput[-1000:]
            self.cache_hits = self.cache_hits[-1000:]
            self.result_counts = self.result_counts[-1000:]
            self.timestamps = self.timestamps[-1000:]
            self.strategies = self.strategies[-1000:]
    
    def get_recent_performance(
        self,
        window: int = 100
    ) -> Dict[str, float]:
        """Get recent performance metrics."""
        if not self.execution_times:
            return {}
        
        window = min(window, len(self.execution_times))
        recent = {
            "avg_time": np.mean(self.execution_times[-window:]),
            "avg_throughput": np.mean(self.throughput[-window:]),
            "cache_hit_rate": (
                np.sum(self.cache_hits[-window:]) /
                np.sum(self.result_counts[-window:])
                if np.sum(self.result_counts[-window:]) > 0 else 0
            ),
            "strategy": self.strategies[-1]
        }
        
        return recent

class OptimizationStrategy:
    """Optimization strategy for filters."""
    
    def __init__(
        self,
        name: str,
        conditions: Dict[str, Any],
        actions: Dict[str, Any]
    ):
        self.name = name
        self.conditions = conditions
        self.actions = actions
        self.success_rate = 0.0
        self.attempts = 0
        self.successes = 0
    
    def matches(
        self,
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ) -> bool:
        """Check if strategy matches current conditions."""
        for key, condition in self.conditions.items():
            if key not in metrics:
                continue
            
            op, value = condition
            if op == "gt" and metrics[key] <= value:
                return False
            elif op == "lt" and metrics[key] >= value:
                return False
            elif op == "eq" and metrics[key] != value:
                return False
        
        return True
    
    def apply(
        self,
        optimizer: FilterOptimizer,
        filter_name: str
    ):
        """Apply optimization actions."""
        for action, value in self.actions.items():
            if hasattr(optimizer.config, action):
                setattr(optimizer.config, action, value)
        
        # Re-optimize filter
        optimizer.optimize_filter(filter_name)
    
    def update_success(self, success: bool):
        """Update strategy success rate."""
        self.attempts += 1
        if success:
            self.successes += 1
        self.success_rate = self.successes / self.attempts if self.attempts > 0 else 0

class AdaptiveOptimizer:
    """Adaptive optimization for validation filters."""
    
    def __init__(
        self,
        optimizer: FilterOptimizer,
        config: AdaptiveConfig
    ):
        self.optimizer = optimizer
        self.config = config
        self.metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self.strategies: Dict[str, List[OptimizationStrategy]] = defaultdict(list)
        self.active_strategies: Dict[str, OptimizationStrategy] = {}
        self.update_thread: Optional[threading.Thread] = None
        self.running = False
        self.performance_queue = queue.Queue()
        
        self._initialize_strategies()
    
    def start_monitoring(self):
        """Start monitoring and adaptation."""
        if self.update_thread is not None:
            return
        
        self.running = True
        self.update_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.update_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
            self.update_thread = None
    
    def apply_filter(
        self,
        filter_name: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply filter with adaptive optimization."""
        if filter_name not in self.optimizer.filter_manager.filters:
            raise KeyError(f"Filter not found: {filter_name}")
        
        start_time = time.time()
        cache_hits = 0
        
        try:
            # Apply current strategy
            if filter_name in self.active_strategies:
                self.active_strategies[filter_name].apply(
                    self.optimizer,
                    filter_name
                )
            
            # Apply filter
            filtered = self.optimizer.apply_optimized_filter(
                filter_name,
                results
            )
            
            # Record performance
            execution_time = time.time() - start_time
            self.metrics[filter_name].add_measurement(
                execution_time,
                len(filtered),
                cache_hits,
                self.active_strategies.get(
                    filter_name,
                    OptimizationStrategy("default", {}, {})
                ).name
            )
            
            # Queue performance update
            self.performance_queue.put((
                filter_name,
                execution_time,
                len(filtered),
                cache_hits
            ))
            
            return filtered
            
        except Exception as e:
            logger.error(f"Filter application failed: {e}")
            raise
    
    def add_strategy(
        self,
        filter_name: str,
        strategy: OptimizationStrategy
    ):
        """Add optimization strategy."""
        self.strategies[filter_name].append(strategy)
        
        # Keep only top strategies
        self.strategies[filter_name] = heapq.nlargest(
            self.config.max_strategies,
            self.strategies[filter_name],
            key=lambda s: s.success_rate
        )
    
    def get_performance_report(
        self,
        filter_name: str
    ) -> Dict[str, Any]:
        """Get performance report for filter."""
        if filter_name not in self.metrics:
            return {}
        
        metrics = self.metrics[filter_name]
        recent = metrics.get_recent_performance()
        
        return {
            "current_strategy": self.active_strategies.get(
                filter_name,
                "none"
            ),
            "performance": recent,
            "strategies": [
                {
                    "name": strategy.name,
                    "success_rate": strategy.success_rate,
                    "attempts": strategy.attempts
                }
                for strategy in self.strategies[filter_name]
            ]
        }
    
    def _initialize_strategies(self):
        """Initialize optimization strategies."""
        default_strategies = [
            OptimizationStrategy(
                "parallel_large",
                {"avg_time": ("gt", 1.0), "result_counts": ("gt", 5000)},
                {"parallel_threshold": 1000, "max_workers": 4}
            ),
            OptimizationStrategy(
                "indexed_frequent",
                {"cache_hit_rate": ("gt", 0.8)},
                {"enable_indexing": True, "cache_filters": True}
            ),
            OptimizationStrategy(
                "batch_medium",
                {"result_counts": ("gt", 1000)},
                {"min_batch_size": 500}
            ),
            OptimizationStrategy(
                "simple_small",
                {"result_counts": ("lt", 100)},
                {"enable_indexing": False, "cache_filters": False}
            )
        ]
        
        for filter_name in self.optimizer.filter_manager.filters:
            for strategy in default_strategies:
                self.add_strategy(filter_name, strategy)
    
    def _monitor_loop(self):
        """Monitor and adapt optimization."""
        warm_up_end = time.time() + self.config.warm_up_period
        
        while self.running:
            try:
                # Process performance updates
                while not self.performance_queue.empty():
                    update = self.performance_queue.get_nowait()
                    self._process_performance_update(*update)
                
                # Skip adaptation during warm-up
                if time.time() < warm_up_end:
                    time.sleep(1)
                    continue
                
                # Adapt strategies
                if self.config.auto_tune:
                    self._adapt_strategies()
                
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(1)
    
    def _process_performance_update(
        self,
        filter_name: str,
        execution_time: float,
        result_count: int,
        cache_hits: int
    ):
        """Process performance update."""
        # Update strategy success
        if filter_name in self.active_strategies:
            strategy = self.active_strategies[filter_name]
            
            # Calculate success based on performance improvement
            baseline = self.metrics[filter_name].get_recent_performance()
            if baseline:
                improvement = (
                    baseline["avg_time"] - execution_time
                ) / baseline["avg_time"]
                
                strategy.update_success(improvement > self.config.decision_threshold)
    
    def _adapt_strategies(self):
        """Adapt optimization strategies."""
        for filter_name in self.optimizer.filter_manager.filters:
            metrics = self.metrics[filter_name].get_recent_performance()
            if not metrics:
                continue
            
            # Find best matching strategy
            best_strategy = None
            best_score = -1
            
            for strategy in self.strategies[filter_name]:
                if strategy.matches(metrics, self.optimizer.config.__dict__):
                    score = strategy.success_rate
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy
            
            # Apply best strategy if it's different from current
            if (
                best_strategy and
                best_strategy.name != self.active_strategies.get(
                    filter_name,
                    OptimizationStrategy("none", {}, {})
                ).name
            ):
                self.active_strategies[filter_name] = best_strategy
                best_strategy.apply(self.optimizer, filter_name)
                
                logger.info(
                    f"Switched to strategy {best_strategy.name} "
                    f"for filter {filter_name}"
                )
    
    def save_state(self):
        """Save optimizer state."""
        if not self.config.output_path:
            return
        
        try:
            state = {
                "metrics": {
                    name: {
                        "execution_times": metrics.execution_times,
                        "throughput": metrics.throughput,
                        "cache_hits": metrics.cache_hits,
                        "result_counts": metrics.result_counts,
                        "timestamps": [
                            t.isoformat() for t in metrics.timestamps
                        ],
                        "strategies": metrics.strategies
                    }
                    for name, metrics in self.metrics.items()
                },
                "strategies": {
                    name: [
                        {
                            "name": s.name,
                            "conditions": s.conditions,
                            "actions": s.actions,
                            "success_rate": s.success_rate,
                            "attempts": s.attempts,
                            "successes": s.successes
                        }
                        for s in strategies
                    ]
                    for name, strategies in self.strategies.items()
                },
                "active_strategies": {
                    name: strategy.name
                    for name, strategy in self.active_strategies.items()
                }
            }
            
            output_file = self.config.output_path / "adaptive_state.json"
            with open(output_file, "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved state to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load_state(self):
        """Load optimizer state."""
        if not self.config.output_path:
            return
        
        try:
            state_file = self.config.output_path / "adaptive_state.json"
            if not state_file.exists():
                return
            
            with open(state_file) as f:
                state = json.load(f)
            
            # Restore metrics
            for name, data in state["metrics"].items():
                metrics = PerformanceMetrics()
                metrics.execution_times = data["execution_times"]
                metrics.throughput = data["throughput"]
                metrics.cache_hits = data["cache_hits"]
                metrics.result_counts = data["result_counts"]
                metrics.timestamps = [
                    datetime.fromisoformat(t)
                    for t in data["timestamps"]
                ]
                metrics.strategies = data["strategies"]
                self.metrics[name] = metrics
            
            # Restore strategies
            for name, strategies in state["strategies"].items():
                self.strategies[name] = [
                    OptimizationStrategy(
                        s["name"],
                        s["conditions"],
                        s["actions"]
                    )
                    for s in strategies
                ]
                
                # Restore strategy stats
                for s, data in zip(self.strategies[name], strategies):
                    s.success_rate = data["success_rate"]
                    s.attempts = data["attempts"]
                    s.successes = data["successes"]
            
            # Restore active strategies
            for name, strategy_name in state["active_strategies"].items():
                matching = [
                    s for s in self.strategies[name]
                    if s.name == strategy_name
                ]
                if matching:
                    self.active_strategies[name] = matching[0]
            
            logger.info(f"Loaded state from {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

def create_adaptive_optimizer(
    optimizer: FilterOptimizer,
    output_path: Optional[Path] = None
) -> AdaptiveOptimizer:
    """Create adaptive optimizer."""
    config = AdaptiveConfig(output_path=output_path)
    return AdaptiveOptimizer(optimizer, config)

if __name__ == "__main__":
    # Example usage
    from .filter_optimization import create_filter_optimizer
    from .validation_filters import create_filter_manager
    from .preset_validation import create_preset_validator
    from .visualization_presets import create_preset_manager
    
    # Create components
    preset_manager = create_preset_manager()
    validator = create_preset_validator(preset_manager)
    filter_manager = create_filter_manager(validator)
    optimizer = create_filter_optimizer(filter_manager)
    
    # Create adaptive optimizer
    adaptive = create_adaptive_optimizer(
        optimizer,
        output_path=Path("adaptive_optimization")
    )
    
    # Start monitoring
    adaptive.start_monitoring()
    
    # Create and apply filter
    filter_manager.create_error_filter(
        "critical_errors",
        ["schema", "value_range"]
    )
    
    results = adaptive.apply_filter(
        "critical_errors",
        validator.validation_results.values()
    )
    
    # Get performance report
    report = adaptive.get_performance_report("critical_errors")
    print(json.dumps(report, indent=2))
    
    # Stop monitoring
    adaptive.stop_monitoring()
    adaptive.save_state()
