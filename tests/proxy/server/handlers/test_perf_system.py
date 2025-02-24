"""System-level performance and stress tests."""
import pytest
import asyncio
import psutil
import gc
import os
import time
from pathlib import Path
import multiprocessing
from typing import Dict, Any, List
import logging
import signal

from .perf_config import PerformanceSettings
from .run_perf_tests import PerformanceTestRunner
from .mock_server import MockHttpsServer
from proxy.server.handlers.middleware import ProxyResponse

logger = logging.getLogger(__name__)

class SystemStressMiddleware:
    """Middleware that generates system stress."""
    def __init__(self, cpu_load: float = 0.0, memory_mb: int = 0):
        self.cpu_load = cpu_load
        self.memory_mb = memory_mb
        self._memory_blocks: List[bytearray] = []
    
    async def __call__(self, request: dict) -> ProxyResponse:
        # Generate CPU load
        if self.cpu_load > 0:
            end_time = time.time() + self.cpu_load
            while time.time() < end_time:
                _ = [i * i for i in range(1000)]
        
        # Allocate memory if requested
        if self.memory_mb > 0:
            block = bytearray(1024 * 1024)  # 1MB
            self._memory_blocks.append(block)
        
        return None
    
    def cleanup(self):
        """Release allocated memory."""
        self._memory_blocks.clear()
        gc.collect()

@pytest.fixture
def system_monitor():
    """Monitor system resources during tests."""
    class SystemMonitor:
        def __init__(self):
            self.start_time = time.time()
            self.process = psutil.Process()
            self.initial_memory = self.process.memory_info().rss
            self.samples: List[Dict[str, float]] = []
        
        def sample(self):
            """Take a resource sample."""
            current = {
                "time": time.time() - self.start_time,
                "cpu_percent": self.process.cpu_percent(),
                "memory_mb": self.process.memory_info().rss / (1024 * 1024),
                "threads": self.process.num_threads(),
                "fds": self.process.num_fds() if hasattr(self.process, "num_fds") else 0
            }
            self.samples.append(current)
            return current
        
        def get_stats(self) -> Dict[str, Any]:
            """Calculate resource usage statistics."""
            if not self.samples:
                return {}
            
            return {
                "duration": self.samples[-1]["time"] - self.samples[0]["time"],
                "peak_cpu": max(s["cpu_percent"] for s in self.samples),
                "peak_memory": max(s["memory_mb"] for s in self.samples),
                "max_threads": max(s["threads"] for s in self.samples),
                "max_fds": max(s["fds"] for s in self.samples)
            }
    
    monitor = SystemMonitor()
    yield monitor
    stats = monitor.get_stats()
    logger.info("System resource usage:\n%s", "\n".join(
        f"{k}: {v:.2f}" for k, v in stats.items()
    ))

async def run_parallel_tests(config_path: Path, count: int) -> List[Dict[str, Any]]:
    """Run multiple test instances in parallel."""
    async def run_instance(instance_id: int):
        runner = PerformanceTestRunner(config_path)
        runner.results_dir = runner.results_dir / f"instance_{instance_id}"
        runner.results_dir.mkdir(parents=True, exist_ok=True)
        return await runner.run_tests()
    
    tasks = [run_instance(i) for i in range(count)]
    return await asyncio.gather(*tasks)

@pytest.mark.system
@pytest.mark.asyncio
async def test_high_concurrent_connections(test_config, test_server, system_monitor):
    """Test handling of many concurrent connections."""
    server, _ = test_server
    
    # Configure for high concurrency
    with open(test_config) as f:
        config = PerformanceSettings(**yaml.safe_load(f))
    
    config.concurrent_connections = 1000
    config.request_count = 10000
    
    # Run test while monitoring
    runner = PerformanceTestRunner(test_config)
    monitoring_task = asyncio.create_task(
        monitor_resources(system_monitor, interval=0.1)
    )
    
    try:
        results = await runner.run_tests()
        await monitoring_task
        
        stats = system_monitor.get_stats()
        assert stats["peak_cpu"] < 90.0, "CPU usage too high"
        assert stats["max_threads"] < 2000, "Too many threads created"
        
    except Exception as e:
        logger.error("Test failed: %s", e)
        raise

@pytest.mark.system
@pytest.mark.asyncio
async def test_memory_pressure(test_config, test_server, system_monitor):
    """Test behavior under memory pressure."""
    server, _ = test_server
    
    # Add memory-intensive middleware
    memory_middleware = SystemStressMiddleware(memory_mb=100)  # Allocate 100MB per request
    server._routes["/test"].handler = memory_middleware
    
    try:
        runner = PerformanceTestRunner(test_config)
        monitoring_task = asyncio.create_task(
            monitor_resources(system_monitor, interval=0.1)
        )
        
        results = await runner.run_tests()
        await monitoring_task
        
        # Verify memory handling
        stats = system_monitor.get_stats()
        assert stats["peak_memory"] < 1024, "Memory usage exceeded 1GB"
        
    finally:
        memory_middleware.cleanup()

@pytest.mark.system
@pytest.mark.asyncio
async def test_cpu_intensive_load(test_config, test_server, system_monitor):
    """Test behavior under CPU pressure."""
    server, _ = test_server
    
    # Add CPU-intensive middleware
    server._routes["/test"].handler = SystemStressMiddleware(cpu_load=0.01)  # 10ms CPU work
    
    runner = PerformanceTestRunner(test_config)
    monitoring_task = asyncio.create_task(
        monitor_resources(system_monitor, interval=0.1)
    )
    
    results = await runner.run_tests()
    await monitoring_task
    
    # Verify CPU handling
    stats = system_monitor.get_stats()
    assert stats["peak_cpu"] < 95.0, "CPU usage too high"
    assert results["results"]["throughput"] > 100, "Throughput too low under CPU load"

@pytest.mark.system
@pytest.mark.asyncio
async def test_parallel_test_execution(test_config, tmp_path):
    """Test running multiple test instances in parallel."""
    # Run multiple test instances
    instance_count = 3
    results = await run_parallel_tests(test_config, instance_count)
    
    # Verify all instances completed
    assert len(results) == instance_count
    
    # Check resource isolation
    for instance_results in results:
        assert "results" in instance_results
        assert "timestamp" in instance_results
        assert instance_results["results"]["throughput"] > 0

@pytest.mark.system
@pytest.mark.asyncio
async def test_graceful_shutdown(test_config, test_server):
    """Test graceful shutdown during test execution."""
    server, _ = test_server
    
    async def delayed_shutdown():
        await asyncio.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)
    
    # Start shutdown timer
    shutdown_task = asyncio.create_task(delayed_shutdown())
    
    runner = PerformanceTestRunner(test_config)
    try:
        await runner.run_tests()
    except asyncio.CancelledError:
        # Verify cleanup
        assert server.is_running() is False
        
        # Check partial results
        results_file = list(Path(runner.results_dir).glob("results_*.json"))
        assert results_file, "No results saved during shutdown"
    finally:
        if not shutdown_task.done():
            shutdown_task.cancel()

async def monitor_resources(monitor, interval: float = 0.1):
    """Continuously monitor system resources."""
    while True:
        monitor.sample()
        await asyncio.sleep(interval)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])
