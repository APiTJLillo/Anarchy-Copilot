"""Integration tests for performance results processing pipeline."""
import pytest
import asyncio
from pathlib import Path
import json
import yaml
from datetime import datetime

from scripts.process_perf_results import ResultsProcessor
from .run_perf_tests import PerformanceTestRunner
from .mock_server import MockHttpsServer
from proxy.server.handlers.middleware import ProxyResponse

class MetricsMiddleware:
    """Middleware that generates predictable metrics."""
    def __init__(self, throughput: float = 1000.0, latency: float = 0.005):
        self.throughput = throughput
        self.latency = latency
        self.request_count = 0
    
    async def __call__(self, request: dict) -> ProxyResponse:
        self.request_count += 1
        await asyncio.sleep(self.latency)
        return None

@pytest.fixture
async def test_environment(tmp_path):
    """Set up integrated test environment."""
    # Create test directories
    results_dir = tmp_path / "results"
    config_dir = tmp_path / "config"
    results_dir.mkdir()
    config_dir.mkdir()
    
    # Create test configuration
    config = {
        "concurrent_connections": 10,
        "request_count": 100,
        "warmup_requests": 10,
        "min_throughput": 500.0,
        "max_latency_p95": 0.01,
        "output_dir": str(results_dir),
        "baseline_dir": str(tmp_path / "baselines")
    }
    
    config_path = config_dir / "test_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    # Set up test server
    server = MockHttpsServer({"port": 0})
    middleware = MetricsMiddleware()
    
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
        yield {
            "server": server,
            "middleware": middleware,
            "config_path": config_path,
            "results_dir": results_dir
        }
    finally:
        await server.stop()

@pytest.mark.asyncio
async def test_complete_pipeline(test_environment):
    """Test complete performance testing and results processing pipeline."""
    # Run performance tests
    runner = PerformanceTestRunner(test_environment["config_path"])
    test_results = await runner.run_tests()
    
    # Process results
    processor = ResultsProcessor(test_environment["results_dir"])
    summary = processor.process_results()
    
    # Verify test execution
    middleware = test_environment["middleware"]
    assert middleware.request_count > 0
    
    # Verify results processing
    assert "tests" in summary
    assert "system" in summary
    assert "trends" in summary
    assert summary["tests"]["throughput"]["mean"] > 0

@pytest.mark.asyncio
async def test_regression_detection(test_environment):
    """Test regression detection in the pipeline."""
    # Create baseline with better performance
    baseline_dir = Path(test_environment["results_dir"].parent) / "baselines"
    baseline_dir.mkdir()
    
    baseline_data = {
        "results": {
            "throughput": 2000.0,
            "latency_p95": 0.001,
            "memory_usage": 100 * 1024 * 1024
        },
        "timestamp": (datetime.now().isoformat())
    }
    
    with open(baseline_dir / "baseline_initial.json", "w") as f:
        json.dump(baseline_data, f)
    
    # Run tests with degraded performance
    test_environment["middleware"].latency = 0.01  # Increase latency
    runner = PerformanceTestRunner(test_environment["config_path"])
    await runner.run_tests()
    
    # Process results
    processor = ResultsProcessor(test_environment["results_dir"])
    summary = processor.process_results()
    
    # Verify regression detection
    assert summary["regressions"]["total_count"] > 0
    assert any(
        "latency" in metric 
        for metrics in summary["regressions"]["severity"].values()
        for metric in metrics
    )

@pytest.mark.asyncio
async def test_trend_analysis(test_environment):
    """Test performance trend analysis over multiple runs."""
    # Run multiple tests with varying performance
    latencies = [0.001, 0.002, 0.003]  # Degrading performance
    
    for latency in latencies:
        test_environment["middleware"].latency = latency
        runner = PerformanceTestRunner(test_environment["config_path"])
        await runner.run_tests()
    
    # Process results
    processor = ResultsProcessor(test_environment["results_dir"])
    summary = processor.process_results()
    
    # Verify trend analysis
    assert "trends" in summary
    assert "latency" in summary["trends"]
    assert summary["trends"]["latency"]["direction"] == "degrading"

@pytest.mark.asyncio
async def test_report_formats(test_environment):
    """Test different report formats in the pipeline."""
    # Run performance test
    runner = PerformanceTestRunner(test_environment["config_path"])
    await runner.run_tests()
    
    # Generate different report formats
    formats = ["text", "github"]
    reports = {}
    
    for fmt in formats:
        processor = ResultsProcessor(test_environment["results_dir"], fmt)
        summary = processor.process_results()
        reports[fmt] = processor.generate_report(summary)
    
    # Verify text format
    assert "Performance Test Results" in reports["text"]
    assert "=" * 20 in reports["text"]
    
    # Verify GitHub format
    assert "## Performance Test Results" in reports["github"]
    assert "|" in reports["github"]  # Table format

@pytest.mark.asyncio
async def test_system_metrics_integration(test_environment):
    """Test system metrics collection and processing."""
    # Run test with system monitoring
    runner = PerformanceTestRunner(test_environment["config_path"])
    await runner.run_tests()
    
    # Process results
    processor = ResultsProcessor(test_environment["results_dir"])
    summary = processor.process_results()
    
    # Verify system metrics
    assert "system" in summary
    system_metrics = summary["system"]
    assert "cpu_peak" in system_metrics
    assert "memory_peak" in system_metrics

@pytest.mark.asyncio
async def test_error_handling(test_environment):
    """Test error handling in the pipeline."""
    # Corrupt a results file
    with open(test_environment["results_dir"] / "corrupt_results.json", "w") as f:
        f.write("invalid json content")
    
    # Process results
    processor = ResultsProcessor(test_environment["results_dir"])
    summary = processor.process_results()
    
    # Verify graceful handling
    assert "tests" in summary
    assert "errors" not in summary  # Should skip invalid files without failing

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
