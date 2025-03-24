"""Integration tests for the complete performance testing pipeline."""
import pytest
import asyncio
import json
from pathlib import Path
import tempfile
import shutil
import yaml

from .perf_config import PerformanceSettings
from .run_perf_tests import PerformanceTestRunner
from .mock_server import MockHttpsServer
from proxy.server.handlers.http import HttpRequestHandler
from proxy.server.handlers.middleware import ProxyResponse

class TestMiddleware:
    """Test middleware that simulates varying performance characteristics."""
    def __init__(self, latency: float = 0.001):
        self.latency = latency
        self.request_count = 0
        
    async def __call__(self, request: dict) -> ProxyResponse:
        self.request_count += 1
        await asyncio.sleep(self.latency)
        return None

@pytest.fixture
def test_config(tmp_path):
    """Create test configuration."""
    config_path = tmp_path / "test_config.yml"
    config = {
        "concurrent_connections": 10,
        "request_count": 100,
        "warmup_requests": 10,
        "cooldown_time": 1,
        "min_throughput": 100.0,
        "max_latency_p95": 0.1,
        "max_memory_mb": 100.0,
        "data_sizes": [1024],
        "save_raw_data": True,
        "output_dir": str(tmp_path / "results"),
        "baseline_dir": str(tmp_path / "baselines")
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path

@pytest.fixture
async def test_server():
    """Create and run test server with middleware."""
    server = MockHttpsServer({"port": 0})
    middleware = TestMiddleware()
    
    @server.route("/test")
    async def test_handler(request):
        await middleware(request)
        return {
            "status": 200,
            "headers": {"Content-Type": "text/plain"},
            "body": b"test response"
        }
    
    await server.start()
    try:
        yield server, middleware
    finally:
        await server.stop()

@pytest.mark.asyncio
async def test_complete_performance_pipeline(test_config, test_server, tmp_path):
    """Test the entire performance testing pipeline."""
    server, middleware = test_server
    
    # Run performance tests
    runner = PerformanceTestRunner(test_config)
    results = await runner.run_tests()
    
    # Verify test execution
    assert middleware.request_count > 0
    assert "results" in results
    assert "timestamp" in results
    
    # Check result files
    results_dir = Path(runner.results_dir)
    assert results_dir.exists()
    
    results_files = list(results_dir.glob("results_*.json"))
    assert len(results_files) == 1
    
    summary_files = list(results_dir.glob("summary_*.txt"))
    assert len(summary_files) == 1
    
    # Verify results content
    with open(results_files[0]) as f:
        saved_results = json.load(f)
        assert "results" in saved_results
        assert "throughput" in saved_results["results"]
        assert "latency_p95" in saved_results["results"]
        assert "memory_usage" in saved_results["results"]

@pytest.mark.asyncio
async def test_baseline_comparison(test_config, test_server, tmp_path):
    """Test baseline comparison and regression detection."""
    server, middleware = test_server
    
    # Create baseline with better performance
    baseline_dir = Path(tmp_path) / "baselines"
    baseline_dir.mkdir()
    baseline_data = {
        "throughput": 5000.0,
        "latency_p95": 0.001,
        "memory_usage": 50 * 1024 * 1024,
        "timestamp": "2025-02-23T00:00:00"
    }
    
    with open(baseline_dir / "baseline_initial.json", "w") as f:
        json.dump(baseline_data, f)
    
    # Run tests with slower middleware
    middleware.latency = 0.01  # Increase latency
    runner = PerformanceTestRunner(test_config)
    results = await runner.run_tests()
    
    # Verify regression detection
    assert "regressions" in results
    assert len(results["regressions"]) > 0
    
    # Check regression reporting
    summary_files = list(Path(runner.results_dir).glob("summary_*.txt"))
    with open(summary_files[0]) as f:
        summary = f.read()
        assert "Performance Regressions" in summary
        assert "degradation" in summary

@pytest.mark.asyncio
async def test_different_load_profiles(test_config, test_server, tmp_path):
    """Test different load profiles and their impact."""
    server, middleware = test_server
    
    # Test configurations
    configs = {
        "light": {"concurrent_connections": 5, "request_count": 50},
        "medium": {"concurrent_connections": 20, "request_count": 200},
        "heavy": {"concurrent_connections": 50, "request_count": 500}
    }
    
    results = {}
    for profile, settings in configs.items():
        # Update config
        with open(test_config) as f:
            config = yaml.safe_load(f)
        config.update(settings)
        
        profile_config = Path(str(test_config).replace(".yml", f"_{profile}.yml"))
        with open(profile_config, "w") as f:
            yaml.dump(config, f)
        
        # Run tests
        runner = PerformanceTestRunner(profile_config)
        results[profile] = await runner.run_tests()
        
        # Verify increasing load impact
        if results:
            prev_profile = list(results.keys())[-2]
            assert results[profile]["results"]["latency_p95"] >= \
                   results[prev_profile]["results"]["latency_p95"]

@pytest.mark.asyncio
async def test_error_handling_and_recovery(test_config, test_server, tmp_path):
    """Test error handling and recovery during performance tests."""
    server, middleware = test_server
    
    class FailingMiddleware(TestMiddleware):
        def __init__(self, fail_after: int):
            super().__init__()
            self.fail_after = fail_after
        
        async def __call__(self, request: dict) -> ProxyResponse:
            self.request_count += 1
            if self.request_count == self.fail_after:
                raise Exception("Simulated failure")
            return await super().__call__(request)
    
    # Replace middleware with failing version
    server._routes["/test"].handler = FailingMiddleware(fail_after=50)
    
    runner = PerformanceTestRunner(test_config)
    with pytest.raises(Exception) as exc_info:
        await runner.run_tests()
    
    assert "Simulated failure" in str(exc_info.value)
    
    # Verify partial results were saved
    results_dir = Path(runner.results_dir)
    assert results_dir.exists()
    assert list(results_dir.glob("results_*.json"))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
