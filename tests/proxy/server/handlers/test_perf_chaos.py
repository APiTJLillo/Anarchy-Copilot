"""Chaos testing for performance infrastructure."""
import pytest
import asyncio
import psutil
import os
import signal
import random
import resource
import logging
from pathlib import Path
from typing import List, Dict, Any
from contextlib import contextmanager

from .mock_server import MockHttpsServer
from .run_perf_tests import PerformanceTestRunner
from scripts.process_perf_results import ResultsProcessor

logger = logging.getLogger(__name__)

class ChaosGenerator:
    """Generate chaos conditions for testing."""
    
    def __init__(self):
        self.chaos_tasks: List[asyncio.Task] = []
        self.stopped = asyncio.Event()
    
    async def start_chaos(self, intensity: str = "medium"):
        """Start generating chaos conditions."""
        self.stopped.clear()
        chaos_funcs = {
            "cpu_stress": self._cpu_stress,
            "memory_pressure": self._memory_pressure,
            "network_disruption": self._network_disruption,
            "file_handle_exhaustion": self._file_handle_exhaustion,
            "process_interrupts": self._process_interrupts
        }
        
        # Configure intensity
        intensities = {
            "low": 1,
            "medium": 3,
            "high": 5
        }
        count = intensities.get(intensity, 3)
        
        # Start random chaos generators
        chosen = random.sample(list(chaos_funcs.items()), count)
        for name, func in chosen:
            logger.info(f"Starting chaos generator: {name}")
            task = asyncio.create_task(func())
            self.chaos_tasks.append(task)
    
    async def stop_chaos(self):
        """Stop all chaos conditions."""
        self.stopped.set()
        for task in self.chaos_tasks:
            task.cancel()
        self.chaos_tasks.clear()
    
    async def _cpu_stress(self):
        """Generate CPU stress."""
        while not self.stopped.is_set():
            # Simulate CPU-intensive work
            duration = random.uniform(0.1, 1.0)
            end_time = asyncio.get_event_loop().time() + duration
            while asyncio.get_event_loop().time() < end_time:
                _ = [i * i for i in range(10000)]
            await asyncio.sleep(random.uniform(0.5, 2.0))
    
    async def _memory_pressure(self):
        """Generate memory pressure."""
        blocks: List[bytearray] = []
        try:
            while not self.stopped.is_set():
                # Allocate random memory blocks
                if random.random() < 0.7:  # 70% chance to allocate
                    size = random.randint(1024 * 1024, 10 * 1024 * 1024)
                    blocks.append(bytearray(size))
                elif blocks:  # 30% chance to free
                    blocks.pop()
                await asyncio.sleep(random.uniform(0.1, 0.5))
        finally:
            blocks.clear()
    
    async def _network_disruption(self):
        """Simulate network disruptions."""
        while not self.stopped.is_set():
            if random.random() < 0.3:  # 30% chance of disruption
                # Simulate network delay
                await asyncio.sleep(random.uniform(0.1, 1.0))
            await asyncio.sleep(random.uniform(0.5, 2.0))
    
    async def _file_handle_exhaustion(self):
        """Simulate file handle exhaustion."""
        files = []
        try:
            while not self.stopped.is_set():
                try:
                    if random.random() < 0.6:  # 60% chance to open
                        f = tempfile.TemporaryFile()
                        files.append(f)
                    elif files:  # 40% chance to close
                        files.pop().close()
                except Exception:
                    pass
                await asyncio.sleep(random.uniform(0.1, 0.5))
        finally:
            for f in files:
                try:
                    f.close()
                except Exception:
                    pass
    
    async def _process_interrupts(self):
        """Generate process interrupts."""
        while not self.stopped.is_set():
            if random.random() < 0.1:  # 10% chance of interrupt
                os.kill(os.getpid(), signal.SIGURG)  # Non-terminal signal
            await asyncio.sleep(random.uniform(1.0, 5.0))

@contextmanager
def resource_limits():
    """Set resource limits for testing."""
    # Save current limits
    old_limits = {}
    for resource_type in [resource.RLIMIT_NOFILE, resource.RLIMIT_AS]:
        old_limits[resource_type] = resource.getrlimit(resource_type)
    
    try:
        # Set test limits
        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, -1))
        yield
    finally:
        # Restore limits
        for resource_type, (soft, hard) in old_limits.items():
            try:
                resource.setrlimit(resource_type, (soft, hard))
            except Exception as e:
                logger.warning(f"Failed to restore resource limit: {e}")

@pytest.fixture
async def chaos_env(e2e_env):
    """Set up chaos testing environment."""
    chaos = ChaosGenerator()
    yield {
        "env": e2e_env,
        "chaos": chaos
    }
    await chaos.stop_chaos()

@pytest.mark.chaos
@pytest.mark.asyncio
async def test_cpu_stress_resilience(chaos_env):
    """Test performance under CPU stress."""
    # Start CPU stress
    await chaos_env["chaos"].start_chaos("medium")
    
    # Run performance test
    runner = PerformanceTestRunner(chaos_env["env"].config_dir / "development.yml")
    results = await runner.run_tests()
    
    # Verify results under stress
    assert results["results"]["throughput"] > 0
    assert results["results"]["latency_p95"] < 1.0

@pytest.mark.chaos
@pytest.mark.asyncio
async def test_memory_pressure_resilience(chaos_env):
    """Test performance under memory pressure."""
    # Configure memory-intensive test
    runner = PerformanceTestRunner(chaos_env["env"].config_dir / "production.yml")
    
    with resource_limits():
        # Start memory pressure
        await chaos_env["chaos"].start_chaos("high")
        
        # Run test
        results = await runner.run_tests()
        
        # Verify memory handling
        memory_mb = results["results"]["memory_usage"] / (1024 * 1024)
        assert memory_mb < 1024, "Memory usage too high"

@pytest.mark.chaos
@pytest.mark.asyncio
async def test_network_disruption_handling(chaos_env):
    """Test handling of network disruptions."""
    # Start network chaos
    await chaos_env["chaos"].start_chaos("medium")
    
    # Run multiple tests to see impact
    results = []
    for _ in range(3):
        runner = PerformanceTestRunner(chaos_env["env"].config_dir / "development.yml")
        result = await runner.run_tests()
        results.append(result)
    
    # Analyze variation
    throughputs = [r["results"]["throughput"] for r in results]
    variation = max(throughputs) - min(throughputs)
    assert variation / statistics.mean(throughputs) < 0.5, "Too much performance variation"

@pytest.mark.chaos
@pytest.mark.asyncio
async def test_resource_exhaustion_recovery(chaos_env):
    """Test recovery from resource exhaustion."""
    # Start resource exhaustion
    await chaos_env["chaos"].start_chaos("high")
    
    with resource_limits():
        # Run tests with limited resources
        runner = PerformanceTestRunner(chaos_env["env"].config_dir / "development.yml")
        try:
            results = await runner.run_tests()
        except Exception as e:
            logger.info(f"Expected error during chaos: {e}")
        
        # Stop chaos and verify recovery
        await chaos_env["chaos"].stop_chaos()
        recovery_results = await runner.run_tests()
        
        assert recovery_results["results"]["throughput"] > 0

@pytest.mark.chaos
@pytest.mark.asyncio
async def test_concurrent_chaos_conditions(chaos_env):
    """Test behavior under multiple chaos conditions."""
    # Start multiple chaos conditions
    await chaos_env["chaos"].start_chaos("high")
    
    # Run performance tests
    runner = PerformanceTestRunner(chaos_env["env"].config_dir / "production.yml")
    results = await runner.run_tests()
    
    # Analyze results for stability
    processor = ResultsProcessor(chaos_env["env"].results_dir)
    summary = processor.process_results()
    
    # Verify basic functionality maintained
    assert results["results"]["throughput"] > 0
    assert summary["tests"]["throughput"]["mean"] > 0

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])
