"""Automated tuning for visualization performance optimization."""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

import pytest
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from .test_costbenefit_optimization import (
    OptimizedVisualizer,
    OptimizationConfig,
    ErrorRecoveryConfig
)

@dataclass
class TuningConfig:
    """Configuration for automated tuning."""
    enable_autotuning: bool = True
    tuning_interval: float = 3600.0  # 1 hour
    min_samples: int = 100
    max_trials: int = 50
    exploration_ratio: float = 0.2
    performance_window: int = 100
    min_improvement: float = 0.05
    max_regression: float = 0.1
    save_history: bool = True
    history_path: str = "tuning_history.json"
    metrics: List[str] = field(default_factory=lambda: [
        "response_time",
        "memory_usage",
        "error_rate",
        "cpu_usage"
    ])

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    response_time: float
    memory_usage: float
    error_rate: float
    cpu_usage: float
    timestamp: datetime = field(default_factory=datetime.now)

class AutoTuner:
    """Automated tuning for visualization performance."""

    def __init__(
        self,
        visualizer: OptimizedVisualizer,
        config: TuningConfig
    ):
        self.visualizer = visualizer
        self.config = config
        
        # State
        self.performance_history: List[PerformanceMetrics] = []
        self.parameter_history: List[Dict[str, float]] = []
        self.model: Optional[GaussianProcessRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.tuning_task: Optional[asyncio.Task] = None
        self.best_params: Optional[Dict[str, float]] = None
        self.best_score: float = float('inf')

    async def start_tuning(self) -> None:
        """Start automated tuning process."""
        if self.tuning_task is None:
            self.tuning_task = asyncio.create_task(self._run_tuning())
            await self._load_history()

    async def stop_tuning(self) -> None:
        """Stop automated tuning process."""
        if self.tuning_task:
            self.tuning_task.cancel()
            try:
                await self.tuning_task
            except asyncio.CancelledError:
                pass
            self.tuning_task = None
            await self._save_history()

    async def _run_tuning(self) -> None:
        """Run continuous tuning process."""
        while True:
            try:
                # Collect performance metrics
                metrics = await self._collect_metrics()
                self.performance_history.append(metrics)
                
                # Update model with new data
                if len(self.performance_history) >= self.config.min_samples:
                    await self._update_model()
                    
                    # Optimize parameters if needed
                    if self._should_optimize():
                        new_params = await self._optimize_parameters()
                        await self._apply_parameters(new_params)
                
                # Save history periodically
                await self._save_history()
                
                # Wait for next tuning interval
                await asyncio.sleep(self.config.tuning_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in tuning process: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        import psutil
        
        process = psutil.Process()
        
        # Calculate average response time from recent operations
        response_times = [
            op_time for op_time in self.visualizer.base_visualizer.operation_times[-100:]
        ] if hasattr(self.visualizer.base_visualizer, "operation_times") else [0.0]
        
        return PerformanceMetrics(
            response_time=np.mean(response_times),
            memory_usage=process.memory_info().rss / (1024 * 1024),  # MB
            error_rate=sum(self.visualizer.error_counts.values()) / max(
                sum(len(self.visualizer.cache) for _ in self.visualizer.cache), 1
            ),
            cpu_usage=process.cpu_percent() / 100.0
        )

    async def _update_model(self) -> None:
        """Update the performance prediction model."""
        # Prepare training data
        X = np.array(self._extract_parameters(self.parameter_history))
        y = np.array(self._extract_metrics(self.performance_history))
        
        # Initialize or update scaler
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Train model
        if self.model is None:
            self.model = GaussianProcessRegressor(
                random_state=42,
                normalize_y=True
            )
        
        self.model.fit(X_scaled, y)

    def _should_optimize(self) -> bool:
        """Determine if optimization is needed."""
        if len(self.performance_history) < self.config.performance_window:
            return False
        
        recent_metrics = self.performance_history[-self.config.performance_window:]
        
        # Calculate performance trends
        trends = {}
        for metric in self.config.metrics:
            values = [getattr(m, metric) for m in recent_metrics]
            trends[metric] = np.polyfit(range(len(values)), values, 1)[0]
        
        # Check for performance degradation
        for metric, trend in trends.items():
            if trend > 0 and metric != "error_rate":  # Increasing trend (bad)
                return True
            if trend < 0 and metric == "error_rate":  # Decreasing trend (good)
                return True
        
        return False

    async def _optimize_parameters(self) -> Dict[str, float]:
        """Optimize performance parameters."""
        current_params = self._get_current_parameters()
        
        def objective(x: np.ndarray) -> float:
            """Objective function for optimization."""
            params = dict(zip(current_params.keys(), x))
            return self._predict_performance(params)
        
        # Define bounds
        bounds = self._get_parameter_bounds()
        
        # Run optimization
        result = minimize(
            objective,
            x0=list(current_params.values()),
            bounds=bounds,
            method="L-BFGS-B"
        )
        
        # Return optimized parameters
        return dict(zip(current_params.keys(), result.x))

    def _predict_performance(self, params: Dict[str, float]) -> float:
        """Predict performance score for parameters."""
        if self.model is None or self.scaler is None:
            return float('inf')
        
        # Scale parameters
        x = self.scaler.transform(np.array([list(params.values())]))
        
        # Predict metrics
        predicted_metrics = self.model.predict(x)[0]
        
        # Calculate weighted score
        weights = {
            "response_time": 0.4,
            "memory_usage": 0.3,
            "error_rate": 0.2,
            "cpu_usage": 0.1
        }
        
        return sum(
            metric * weights[name]
            for metric, name in zip(predicted_metrics, self.config.metrics)
        )

    async def _apply_parameters(self, params: Dict[str, float]) -> None:
        """Apply new parameters if they improve performance."""
        # Measure current performance
        current_metrics = await self._collect_metrics()
        current_score = self._calculate_score(current_metrics)
        
        # Apply new parameters
        old_params = self._get_current_parameters()
        await self._update_parameters(params)
        
        # Measure new performance
        await asyncio.sleep(10)  # Wait for changes to take effect
        new_metrics = await self._collect_metrics()
        new_score = self._calculate_score(new_metrics)
        
        # Revert if performance degraded
        if new_score > current_score * (1 + self.config.max_regression):
            await self._update_parameters(old_params)
        elif new_score < self.best_score:
            self.best_score = new_score
            self.best_params = params.copy()

    def _calculate_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate performance score from metrics."""
        weights = {
            "response_time": 0.4,
            "memory_usage": 0.3,
            "error_rate": 0.2,
            "cpu_usage": 0.1
        }
        
        return sum(
            getattr(metrics, metric) * weight
            for metric, weight in weights.items()
        )

    async def _load_history(self) -> None:
        """Load tuning history from disk."""
        try:
            with open(self.config.history_path, 'r') as f:
                data = json.load(f)
                self.parameter_history = data.get("parameters", [])
                self.best_params = data.get("best_params")
                self.best_score = data.get("best_score", float('inf'))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    async def _save_history(self) -> None:
        """Save tuning history to disk."""
        if not self.config.save_history:
            return
        
        data = {
            "parameters": self.parameter_history,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.config.history_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_current_parameters(self) -> Dict[str, float]:
        """Get current optimization parameters."""
        config = self.visualizer.optimization_config
        return {
            "compression_ratio": config.compression_ratio,
            "batch_size": float(config.batch_size),
            "cache_size": float(config.cache_size),
            "downsample_threshold": float(config.downsample_threshold)
        }

    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for optimization parameters."""
        return [
            (0.01, 0.5),    # compression_ratio
            (10, 200),      # batch_size
            (100, 5000),    # cache_size
            (100, 10000)    # downsample_threshold
        ]

    async def _update_parameters(self, params: Dict[str, float]) -> None:
        """Update optimization parameters."""
        config = self.visualizer.optimization_config
        
        config.compression_ratio = params["compression_ratio"]
        config.batch_size = int(params["batch_size"])
        config.cache_size = int(params["cache_size"])
        config.downsample_threshold = int(params["downsample_threshold"])

@pytest.fixture
def auto_tuner(optimized_visualizer):
    """Create auto-tuner for testing."""
    config = TuningConfig()
    return AutoTuner(optimized_visualizer, config)

@pytest.mark.asyncio
async def test_performance_optimization(auto_tuner):
    """Test automated performance optimization."""
    # Start tuning
    await auto_tuner.start_tuning()
    
    # Generate some test load
    for _ in range(10):
        metrics = await auto_tuner._collect_metrics()
        auto_tuner.performance_history.append(metrics)
    
    # Allow time for optimization
    await asyncio.sleep(1)
    
    # Verify optimization occurred
    assert len(auto_tuner.performance_history) > 0
    if len(auto_tuner.performance_history) >= auto_tuner.config.min_samples:
        assert auto_tuner.model is not None
    
    # Stop tuning
    await auto_tuner.stop_tuning()

@pytest.mark.asyncio
async def test_parameter_persistence(auto_tuner, tmp_path):
    """Test parameter persistence."""
    # Set temporary path
    auto_tuner.config.history_path = str(tmp_path / "tuning_history.json")
    
    # Add test data
    auto_tuner.best_params = {"test_param": 1.0}
    auto_tuner.best_score = 0.5
    
    # Save and reload
    await auto_tuner._save_history()
    auto_tuner.best_params = None
    auto_tuner.best_score = float('inf')
    await auto_tuner._load_history()
    
    # Verify persistence
    assert auto_tuner.best_params == {"test_param": 1.0}
    assert auto_tuner.best_score == 0.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
