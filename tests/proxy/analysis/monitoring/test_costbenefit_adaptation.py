"""Real-time adaptation for visualization performance."""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging
from scipy.stats import norm

from .test_costbenefit_tuning import (
    AutoTuner,
    TuningConfig,
    PerformanceMetrics
)

@dataclass
class AdaptationConfig:
    """Configuration for real-time adaptation."""
    enable_adaptation: bool = True
    min_adaptation_interval: float = 1.0  # seconds
    learning_rate: float = 0.1
    momentum: float = 0.9
    min_confidence: float = 0.8
    max_change_rate: float = 0.2
    change_threshold: float = 0.05
    smoothing_window: int = 10
    trend_window: int = 30
    stability_threshold: float = 0.1
    recovery_threshold: float = 0.3
    recovery_multiplier: float = 2.0
    adaptive_window: bool = True

@dataclass
class AdaptationMetrics:
    """Metrics for adaptation performance."""
    adaptation_time: float
    improvement_ratio: float
    stability_score: float
    confidence_level: float
    change_magnitude: float
    recovery_count: int
    timestamp: datetime = field(default_factory=datetime.now)

class RealTimeAdapter:
    """Real-time performance adaptation."""

    def __init__(
        self,
        tuner: AutoTuner,
        config: AdaptationConfig
    ):
        self.tuner = tuner
        self.config = config
        
        # State
        self.adaptation_history: List[AdaptationMetrics] = []
        self.parameter_velocity: Dict[str, float] = {}
        self.last_adaptation: Optional[datetime] = None
        self.adaptation_task: Optional[asyncio.Task] = None
        self.recovery_mode: bool = False
        self.stability_buffer: Dict[str, List[float]] = {}
        
        # Initialize logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup adaptation logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    async def start_adaptation(self) -> None:
        """Start real-time adaptation."""
        if self.adaptation_task is None:
            self.adaptation_task = asyncio.create_task(self._run_adaptation())

    async def stop_adaptation(self) -> None:
        """Stop real-time adaptation."""
        if self.adaptation_task:
            self.adaptation_task.cancel()
            try:
                await self.adaptation_task
            except asyncio.CancelledError:
                pass
            self.adaptation_task = None

    async def _run_adaptation(self) -> None:
        """Run continuous adaptation process."""
        while True:
            try:
                current_time = datetime.now()
                if self.last_adaptation:
                    time_since_last = (
                        current_time - self.last_adaptation
                    ).total_seconds()
                    if time_since_last < self.config.min_adaptation_interval:
                        await asyncio.sleep(
                            self.config.min_adaptation_interval - time_since_last
                        )
                        continue

                # Check if adaptation is needed
                if await self._should_adapt():
                    metrics = await self._adapt_parameters()
                    self.adaptation_history.append(metrics)
                    self.last_adaptation = current_time
                
                await asyncio.sleep(0.1)  # Small delay to prevent CPU overuse
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Adaptation error: {e}")
                await asyncio.sleep(1.0)

    async def _should_adapt(self) -> bool:
        """Determine if adaptation is needed."""
        if not self.config.enable_adaptation:
            return False
        
        if len(self.tuner.performance_history) < self.config.trend_window:
            return False
        
        # Check performance trends
        recent_metrics = self.tuner.performance_history[-self.config.trend_window:]
        performance_change = self._calculate_trend(recent_metrics)
        
        # Check stability
        if not self._is_stable():
            return False
        
        return (
            abs(performance_change) > self.config.change_threshold or
            self.recovery_mode
        )

    def _calculate_trend(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate performance trend."""
        if not metrics:
            return 0.0
        
        # Calculate weighted average of metric trends
        weights = {
            "response_time": 0.4,
            "memory_usage": 0.3,
            "error_rate": 0.2,
            "cpu_usage": 0.1
        }
        
        trends = {}
        for metric_name in weights:
            values = [getattr(m, metric_name) for m in metrics]
            trends[metric_name] = np.polyfit(range(len(values)), values, 1)[0]
        
        return sum(
            trend * weights[metric]
            for metric, trend in trends.items()
        )

    def _is_stable(self) -> bool:
        """Check if system is stable enough for adaptation."""
        if not self.stability_buffer:
            return True
        
        for values in self.stability_buffer.values():
            if len(values) < self.config.smoothing_window:
                continue
            
            # Calculate stability score
            std = np.std(values[-self.config.smoothing_window:])
            mean = np.mean(values[-self.config.smoothing_window:])
            cv = std / mean if mean != 0 else float('inf')
            
            if cv > self.config.stability_threshold:
                return False
        
        return True

    async def _adapt_parameters(self) -> AdaptationMetrics:
        """Adapt parameters in real-time."""
        start_time = datetime.now()
        
        # Get current performance baseline
        baseline_metrics = await self.tuner._collect_metrics()
        baseline_score = self.tuner._calculate_score(baseline_metrics)
        
        # Calculate parameter updates
        updates = await self._calculate_parameter_updates()
        
        # Apply updates with momentum
        for param, update in updates.items():
            current_value = self.tuner._get_current_parameters()[param]
            velocity = self.parameter_velocity.get(param, 0.0)
            
            # Update velocity with momentum
            new_velocity = (
                self.config.momentum * velocity +
                self.config.learning_rate * update
            )
            self.parameter_velocity[param] = new_velocity
            
            # Calculate new parameter value
            change = new_velocity
            if abs(change) > self.config.max_change_rate * current_value:
                change = np.sign(change) * self.config.max_change_rate * current_value
            
            new_value = current_value + change
            
            # Apply bounds
            bounds = dict(zip(
                self.tuner._get_current_parameters().keys(),
                self.tuner._get_parameter_bounds()
            ))
            new_value = np.clip(new_value, bounds[param][0], bounds[param][1])
            
            # Update parameter
            await self.tuner._update_parameters({param: new_value})
        
        # Measure impact
        new_metrics = await self.tuner._collect_metrics()
        new_score = self.tuner._calculate_score(new_metrics)
        
        # Calculate improvement
        improvement_ratio = (baseline_score - new_score) / baseline_score
        
        # Check if recovery is needed
        if improvement_ratio < -self.config.recovery_threshold:
            self.recovery_mode = True
            # Revert changes with increased magnitude
            for param, update in updates.items():
                await self.tuner._update_parameters({
                    param: self.tuner._get_current_parameters()[param] - 
                    update * self.config.recovery_multiplier
                })
        else:
            self.recovery_mode = False
        
        # Update stability buffer
        for param, value in self.tuner._get_current_parameters().items():
            if param not in self.stability_buffer:
                self.stability_buffer[param] = []
            self.stability_buffer[param].append(value)
            
            # Maintain buffer size
            if len(self.stability_buffer[param]) > self.config.trend_window:
                self.stability_buffer[param] = self.stability_buffer[param][
                    -self.config.trend_window:
                ]
        
        # Calculate adaptation metrics
        return AdaptationMetrics(
            adaptation_time=(datetime.now() - start_time).total_seconds(),
            improvement_ratio=improvement_ratio,
            stability_score=self._calculate_stability_score(),
            confidence_level=self._calculate_confidence(),
            change_magnitude=np.mean([abs(u) for u in updates.values()]),
            recovery_count=int(self.recovery_mode)
        )

    async def _calculate_parameter_updates(self) -> Dict[str, float]:
        """Calculate parameter updates based on performance model."""
        updates = {}
        current_params = self.tuner._get_current_parameters()
        
        for param, value in current_params.items():
            # Get parameter sensitivity
            sensitivity = await self._estimate_sensitivity(param)
            
            # Calculate update magnitude
            magnitude = sensitivity * self.config.learning_rate
            
            # Add noise for exploration
            noise = np.random.normal(0, 0.1 * magnitude)
            
            updates[param] = magnitude + noise
        
        return updates

    async def _estimate_sensitivity(self, param: str) -> float:
        """Estimate parameter sensitivity using performance model."""
        if not self.tuner.model or not self.tuner.scaler:
            return 0.0
        
        current_params = self.tuner._get_current_parameters()
        x_base = np.array(list(current_params.values()))
        
        # Calculate performance change for small parameter perturbation
        delta = 0.01 * current_params[param]
        x_perturbed = x_base.copy()
        param_idx = list(current_params.keys()).index(param)
        x_perturbed[param_idx] += delta
        
        # Predict performances
        x_base_scaled = self.tuner.scaler.transform(x_base.reshape(1, -1))
        x_pert_scaled = self.tuner.scaler.transform(x_perturbed.reshape(1, -1))
        
        y_base = self.tuner.model.predict(x_base_scaled)[0]
        y_pert = self.tuner.model.predict(x_pert_scaled)[0]
        
        # Calculate sensitivity
        sensitivity = (y_pert - y_base) / delta
        return float(sensitivity)

    def _calculate_stability_score(self) -> float:
        """Calculate overall stability score."""
        if not self.stability_buffer:
            return 1.0
        
        scores = []
        for values in self.stability_buffer.values():
            if len(values) < self.config.smoothing_window:
                continue
            
            recent_values = values[-self.config.smoothing_window:]
            std = np.std(recent_values)
            mean = np.mean(recent_values)
            cv = std / mean if mean != 0 else float('inf')
            
            scores.append(np.exp(-cv))
        
        return np.mean(scores) if scores else 1.0

    def _calculate_confidence(self) -> float:
        """Calculate confidence in current adaptation."""
        if not self.tuner.model:
            return 0.0
        
        # Get model uncertainty
        if hasattr(self.tuner.model, "predict_proba"):
            proba = self.tuner.model.predict_proba(
                self.tuner.scaler.transform(
                    np.array(list(self.tuner._get_current_parameters().values())).reshape(1, -1)
                )
            )
            return float(np.max(proba))
        
        return self.config.min_confidence

@pytest.fixture
def real_time_adapter(auto_tuner):
    """Create real-time adapter for testing."""
    config = AdaptationConfig()
    return RealTimeAdapter(auto_tuner, config)

@pytest.mark.asyncio
async def test_real_time_adaptation(real_time_adapter):
    """Test real-time adaptation process."""
    # Start adaptation
    await real_time_adapter.start_adaptation()
    
    # Generate test load
    for _ in range(10):
        await real_time_adapter.tuner._collect_metrics()
        await asyncio.sleep(0.1)
    
    # Allow time for adaptation
    await asyncio.sleep(1)
    
    # Verify adaptation occurred
    assert len(real_time_adapter.adaptation_history) > 0
    
    # Stop adaptation
    await real_time_adapter.stop_adaptation()

@pytest.mark.asyncio
async def test_stability_detection(real_time_adapter):
    """Test stability detection."""
    # Add stable data
    real_time_adapter.stability_buffer["test_param"] = [1.0] * 20
    assert real_time_adapter._is_stable()
    
    # Add unstable data
    real_time_adapter.stability_buffer["test_param"] = [
        1.0 + np.random.normal(0, 0.5) for _ in range(20)
    ]
    assert not real_time_adapter._is_stable()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
