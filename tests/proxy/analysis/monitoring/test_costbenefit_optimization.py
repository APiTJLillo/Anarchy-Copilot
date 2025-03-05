"""Performance optimization and error recovery for real-time visualizations."""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import logging
import traceback

import pytest
import numpy as np
from scipy import signal
from dataclasses_json import dataclass_json

from .test_costbenefit_realtime_viz import (
    RealtimeVisualizer,
    RealtimeConfig,
    TransitionConfig
)

@dataclass_json
@dataclass
class ErrorRecoveryConfig:
    """Configuration for error recovery."""
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    fallback_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    health_check_interval: float = 5.0
    recovery_log_path: Optional[str] = "recovery.log"
    error_notification_threshold: float = 0.1

@dataclass_json
@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    enable_caching: bool = True
    cache_size: int = 1000
    enable_compression: bool = True
    compression_ratio: float = 0.1
    downsampling_enabled: bool = True
    downsample_threshold: int = 1000
    enable_batching: bool = True
    batch_size: int = 50
    max_queue_size: int = 5000
    gc_interval: float = 300.0
    memory_limit_mb: float = 1024.0
    cpu_threshold: float = 0.8

class OptimizedVisualizer:
    """Add performance optimization and error recovery to visualizations."""

    def __init__(
        self,
        base_visualizer: RealtimeVisualizer,
        error_config: ErrorRecoveryConfig,
        optimization_config: OptimizationConfig
    ):
        self.base_visualizer = base_visualizer
        self.error_config = error_config
        self.optimization_config = optimization_config
        
        # Optimization state
        self.cache: Dict[str, Any] = {}
        self.compression_buffers: Dict[str, deque] = {}
        self.error_counts: Dict[str, int] = {}
        self.circuit_breaker_state: Dict[str, bool] = {}
        self.last_circuit_break: Dict[str, float] = {}
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Initialize logging
        self._setup_logging()
        
        # Start health checks
        self._start_health_checks()

    def _setup_logging(self) -> None:
        """Setup logging for error recovery."""
        if self.error_config.recovery_log_path:
            logging.basicConfig(
                filename=self.error_config.recovery_log_path,
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

    def _start_health_checks(self) -> None:
        """Start periodic health checks."""
        if not self.health_check_task:
            self.health_check_task = asyncio.create_task(self._run_health_checks())

    async def _run_health_checks(self) -> None:
        """Run periodic health checks."""
        while True:
            try:
                await self._check_memory_usage()
                await self._check_error_rates()
                await self._reset_circuit_breakers()
                await asyncio.sleep(self.error_config.health_check_interval)
            except Exception as e:
                logging.error(f"Health check error: {e}")

    async def optimize_update(
        self,
        data: Any,
        operation: str
    ) -> Optional[Any]:
        """Apply optimization to update operation."""
        cache_key = f"{operation}:{hash(str(data))}"
        
        # Check cache
        if (
            self.optimization_config.enable_caching and
            cache_key in self.cache
        ):
            return self.cache[cache_key]
        
        # Check circuit breaker
        if self._is_circuit_broken(operation):
            return self._get_fallback_value(operation)
        
        try:
            # Apply optimization
            result = await self._optimized_operation(data, operation)
            
            # Cache result
            if self.optimization_config.enable_caching:
                self.cache[cache_key] = result
                if len(self.cache) > self.optimization_config.cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
            
            return result
            
        except Exception as e:
            return await self._handle_error(e, operation, data)

    async def _optimized_operation(
        self,
        data: Any,
        operation: str
    ) -> Any:
        """Execute operation with optimizations."""
        # Apply compression if enabled
        if (
            self.optimization_config.enable_compression and
            isinstance(data, (list, np.ndarray))
        ):
            data = self._compress_data(data)
        
        # Apply downsampling if enabled
        if (
            self.optimization_config.downsampling_enabled and
            isinstance(data, (list, np.ndarray)) and
            len(data) > self.optimization_config.downsample_threshold
        ):
            data = self._downsample_data(data)
        
        # Execute operation
        method = getattr(self.base_visualizer, operation)
        return await method(data)

    def _compress_data(self, data: List[float]) -> List[float]:
        """Compress data to reduce size."""
        if len(data) < 2:
            return data
        
        # Use signal processing to compress
        target_size = int(len(data) * self.optimization_config.compression_ratio)
        if target_size < 2:
            target_size = 2
        
        return signal.resample(data, target_size).tolist()

    def _downsample_data(self, data: List[float]) -> List[float]:
        """Downsample data to reduce size."""
        if len(data) <= self.optimization_config.downsample_threshold:
            return data
        
        # Use window-based downsampling
        window_size = len(data) // self.optimization_config.downsample_threshold
        return signal.decimate(data, window_size).tolist()

    async def _handle_error(
        self,
        error: Exception,
        operation: str,
        data: Any
    ) -> Optional[Any]:
        """Handle operation errors with retry logic."""
        self.error_counts[operation] = self.error_counts.get(operation, 0) + 1
        
        # Log error
        logging.error(
            f"Error in {operation}: {error}\n{traceback.format_exc()}"
        )
        
        # Check if we should break circuit
        if self._should_break_circuit(operation):
            self._break_circuit(operation)
            return self._get_fallback_value(operation)
        
        # Attempt retry
        for attempt in range(self.error_config.max_retries):
            try:
                # Apply exponential backoff
                if self.error_config.exponential_backoff:
                    delay = self.error_config.retry_delay * (2 ** attempt)
                else:
                    delay = self.error_config.retry_delay
                
                await asyncio.sleep(delay)
                
                # Retry operation
                method = getattr(self.base_visualizer, operation)
                return await method(data)
                
            except Exception as retry_error:
                logging.error(
                    f"Retry {attempt + 1} failed for {operation}: {retry_error}"
                )
        
        # If all retries fail, return fallback
        return self._get_fallback_value(operation)

    def _should_break_circuit(self, operation: str) -> bool:
        """Determine if circuit breaker should be triggered."""
        return (
            self.error_counts.get(operation, 0) >=
            self.error_config.circuit_breaker_threshold
        )

    def _break_circuit(self, operation: str) -> None:
        """Break circuit for operation."""
        self.circuit_breaker_state[operation] = True
        self.last_circuit_break[operation] = time.time()
        logging.warning(f"Circuit breaker triggered for {operation}")

    def _is_circuit_broken(self, operation: str) -> bool:
        """Check if circuit is currently broken."""
        if not self.circuit_breaker_state.get(operation, False):
            return False
        
        # Check if timeout has elapsed
        if (
            time.time() - self.last_circuit_break.get(operation, 0) >
            self.error_config.circuit_breaker_timeout
        ):
            # Reset circuit
            self.circuit_breaker_state[operation] = False
            self.error_counts[operation] = 0
            return False
        
        return True

    def _get_fallback_value(self, operation: str) -> Any:
        """Get fallback value for failed operation."""
        if not self.error_config.fallback_enabled:
            return None
        
        # Return last successful value from cache if available
        cache_key = next(
            (k for k in self.cache if k.startswith(f"{operation}:")),
            None
        )
        if cache_key:
            return self.cache[cache_key]
        
        return None

    async def _check_memory_usage(self) -> None:
        """Check and optimize memory usage."""
        import psutil
        
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        
        if memory_usage > self.optimization_config.memory_limit_mb:
            # Clear caches
            self.cache.clear()
            self.compression_buffers.clear()
            
            # Force garbage collection
            import gc
            gc.collect()

    async def _check_error_rates(self) -> None:
        """Check error rates and notify if too high."""
        total_operations = sum(
            len(self.cache)
            for key in self.cache
            if key.split(":")[0] in self.error_counts
        )
        
        if total_operations == 0:
            return
        
        for operation, count in self.error_counts.items():
            error_rate = count / total_operations
            if error_rate > self.error_config.error_notification_threshold:
                logging.warning(
                    f"High error rate for {operation}: {error_rate:.2%}"
                )

    async def _reset_circuit_breakers(self) -> None:
        """Reset circuit breakers that have timed out."""
        current_time = time.time()
        for operation in list(self.circuit_breaker_state.keys()):
            if self.circuit_breaker_state[operation]:
                if (
                    current_time - self.last_circuit_break[operation] >
                    self.error_config.circuit_breaker_timeout
                ):
                    self.circuit_breaker_state[operation] = False
                    self.error_counts[operation] = 0
                    logging.info(f"Circuit breaker reset for {operation}")

@pytest.fixture
def optimized_visualizer(realtime_visualizer):
    """Create optimized visualizer for testing."""
    error_config = ErrorRecoveryConfig()
    optimization_config = OptimizationConfig()
    return OptimizedVisualizer(
        realtime_visualizer,
        error_config,
        optimization_config
    )

@pytest.mark.asyncio
async def test_optimization(optimized_visualizer):
    """Test optimization features."""
    # Test data compression
    data = list(range(1000))
    compressed = optimized_visualizer._compress_data(data)
    assert len(compressed) < len(data)
    
    # Test downsampling
    downsampled = optimized_visualizer._downsample_data(data)
    assert len(downsampled) <= optimized_visualizer.optimization_config.downsample_threshold
    
    # Test caching
    result1 = await optimized_visualizer.optimize_update(
        data,
        "update_feature_importance"
    )
    result2 = await optimized_visualizer.optimize_update(
        data,
        "update_feature_importance"
    )
    assert result1 == result2

@pytest.mark.asyncio
async def test_error_recovery(optimized_visualizer):
    """Test error recovery mechanisms."""
    # Test retry logic
    operation = "nonexistent_operation"
    result = await optimized_visualizer.optimize_update([], operation)
    assert result is None
    assert optimized_visualizer.error_counts[operation] > 0
    
    # Test circuit breaker
    for _ in range(optimized_visualizer.error_config.circuit_breaker_threshold + 1):
        await optimized_visualizer.optimize_update([], operation)
    
    assert optimized_visualizer._is_circuit_broken(operation)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
