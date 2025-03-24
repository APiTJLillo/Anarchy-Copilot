"""End-to-end tests for performance testing infrastructure."""
import pytest
import asyncio
import os
import signal
import psutil
import tempfile
from pathlib import Path
import json
import yaml
import subprocess
from datetime import datetime, timedelta

from .mock_server import MockHttpsServer
from scripts.process_perf_results import ResultsProcessor
from .run_perf_tests import PerformanceTestRunner
from proxy.server.handlers.middleware import ProxyResponse

class LoadGenerator:
    """Generate test load patterns."""
    
    def __init__(self, concurrency: int, total_requests: int):
        self.concurrency = concurrency
        self.total_requests = total_requests
        self.completed = 0
        self.errors = 0
        
    async def run(self, url: str):
        """Run load generation."""
        tasks = []
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def make_request():
            async with semaphore:
                try:
                    async with ClientSession() as session:
                        async with session.get(url) as response:
                            await response.read()
                            self.completed += 1
                except Exception:
                    self.errors += 1
        
        for _ in range(self.total_requests):
            tasks.append(asyncio.create_task(make_request()))
        
        await asyncio.gather(*tasks)

class E2ETestEnvironment:
    """Manage end-to-end test environment."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / "results"
        self.config_dir = base_dir / "config"
        self.baseline_dir = base_dir / "baselines"
        
        for directory in [self.results_dir, self.config_dir, self.baseline_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def setup(self):
        """Set up test environment."""
        # Create test configuration
        self.create_test_configs()
        
        # Start mock server
        self.server = MockHttpsServer({"port": 0})
        await self.server.start()
        
        return self
    
    def create_test_configs(self):
        """Create test configuration files."""
        configs = {
            "ci": {
                "concurrent_connections": 10,
                "request_count": 100,
                "min_throughput": 500.0
            },
            "development": {
                "concurrent_connections": 50,
                "request_count": 500,
                "min_throughput": 1000.0
            },
            "production": {
                "concurrent_connections": 200,
                "request_count": 2000,
                "min_throughput": 2000.0
            }
        }
        
        for profile, settings in configs.items():
            settings.update({
                "output_dir": str(self.results_dir / profile),
                "baseline_dir": str(self.baseline_dir / profile)
            })
            
            config_path = self.config_dir / f"{profile}.yml"
            with open(config_path, "w") as f:
                yaml.dump(settings, f)
    
    async def cleanup(self):
        """Clean up test environment."""
        await self.server.stop()

@pytest.fixture
async def e2e_env():
    """Create end-to-end test environment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = E2ETestEnvironment(Path(tmpdir))
        env = await env.setup()
        yield env
        await env.cleanup()

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_complete_performance_workflow(e2e_env):
    """Test complete performance testing workflow."""
    # Run CI tests first
    ci_runner = PerformanceTestRunner(e2e_env.config_dir / "ci.yml")
    ci_results = await ci_runner.run_tests()
    
    # Verify CI results and create baseline
    assert ci_results["results"]["throughput"] > 0
    baseline_path = e2e_env.baseline_dir / "ci" / "baseline_initial.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(ci_results, f)
    
    # Run development tests
    dev_runner = PerformanceTestRunner(e2e_env.config_dir / "development.yml")
    dev_results = await dev_runner.run_tests()
    
    # Process and analyze all results
    processor = ResultsProcessor(e2e_env.results_dir)
    summary = processor.process_results()
    
    # Verify full pipeline results
    assert "tests" in summary
    assert "trends" in summary
    assert summary["tests"]["throughput"]["mean"] > 0

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_load_testing_scenarios(e2e_env):
    """Test different load testing scenarios."""
    scenarios = [
        ("light", 10, 100),
        ("medium", 50, 500),
        ("heavy", 100, 1000)
    ]
    
    results = {}
    for name, concurrency, requests in scenarios:
        # Configure load generator
        load_gen = LoadGenerator(concurrency, requests)
        
        # Run load test
        runner = PerformanceTestRunner(e2e_env.config_dir / "development.yml")
        test_start = datetime.now()
        
        # Generate load while running tests
        load_task = asyncio.create_task(
            load_gen.run(f"http://localhost:{e2e_env.server.port}/test")
        )
        results_task = asyncio.create_task(runner.run_tests())
        
        results[name] = await results_task
        await load_task
        
        # Verify test completion
        assert load_gen.completed == requests
        assert load_gen.errors == 0
        assert results[name]["results"]["throughput"] > 0
    
    # Verify load impact
    assert (results["heavy"]["results"]["latency_p95"] >
            results["light"]["results"]["latency_p95"])

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_system_stress_handling(e2e_env):
    """Test system behavior under stress."""
    # Monitor system resources
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Run stress test
    runner = PerformanceTestRunner(e2e_env.config_dir / "production.yml")
    results = await runner.run_tests()
    
    # Verify resource usage
    final_memory = process.memory_info().rss
    memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
    
    assert memory_increase < 1024, "Memory usage increased too much"
    assert results["results"]["throughput"] > 0

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_long_running_stability(e2e_env):
    """Test stability during long-running tests."""
    # Configure for longer run
    with open(e2e_env.config_dir / "stability.yml", "w") as f:
        yaml.dump({
            "concurrent_connections": 20,
            "request_count": 5000,
            "warmup_requests": 100,
            "output_dir": str(e2e_env.results_dir / "stability"),
            "baseline_dir": str(e2e_env.baseline_dir / "stability")
        }, f)
    
    # Run extended test
    runner = PerformanceTestRunner(e2e_env.config_dir / "stability.yml")
    start_time = datetime.now()
    results = await runner.run_tests()
    duration = datetime.now() - start_time
    
    # Verify stability
    assert duration > timedelta(seconds=30)
    assert results["results"]["throughput"] > 0
    
    # Check resource stability
    processor = ResultsProcessor(e2e_env.results_dir / "stability")
    summary = processor.process_results()
    
    assert summary["tests"]["throughput"]["mean"] > 0
    throughput_variance = (
        summary["tests"]["throughput"]["max"] -
        summary["tests"]["throughput"]["min"]
    ) / summary["tests"]["throughput"]["mean"]
    
    assert throughput_variance < 0.5, "Throughput variance too high"

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_error_recovery(e2e_env):
    """Test recovery from errors and interruptions."""
    # Set up error injection
    async def error_injector():
        await asyncio.sleep(2)
        # Simulate error conditions
        os.kill(os.getpid(), signal.SIGINT)
    
    # Run test with error injection
    runner = PerformanceTestRunner(e2e_env.config_dir / "development.yml")
    try:
        error_task = asyncio.create_task(error_injector())
        await runner.run_tests()
    except KeyboardInterrupt:
        pass
    finally:
        if not error_task.done():
            error_task.cancel()
    
    # Verify partial results were saved
    assert list(e2e_env.results_dir.glob("**/results_*.json"))
    
    # Verify recovery
    runner = PerformanceTestRunner(e2e_env.config_dir / "development.yml")
    results = await runner.run_tests()
    assert results["results"]["throughput"] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
