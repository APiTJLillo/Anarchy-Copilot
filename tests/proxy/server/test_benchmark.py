"""Unit tests for benchmark tools."""
import pytest
import asyncio
import tempfile
from pathlib import Path
import json
import time
from unittest.mock import Mock, patch, AsyncMock

from .benchmark import BenchmarkConfig, BenchmarkMetrics, BenchmarkReporter, run_benchmark

@pytest.fixture
def temp_report_dir():
    """Create a temporary directory for benchmark reports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_metrics():
    """Create sample benchmark metrics."""
    metrics = BenchmarkMetrics()
    
    # Add sample data
    for i in range(100):
        metrics.add_request(0.1 + (i * 0.001), 1024)  # Varying latencies
        metrics.sample_system_metrics()
    
    # Add some errors
    for _ in range(5):
        metrics.add_error()
        
    return metrics

@pytest.fixture
def benchmark_config():
    """Create a test benchmark configuration."""
    return BenchmarkConfig(
        concurrent_connections=10,
        request_count=100,
        data_sizes=[1024, 2048],
        duration=5,
        warmup_time=1,
        cooldown_time=1
    )

def test_benchmark_config_defaults():
    """Test benchmark configuration defaults."""
    config = BenchmarkConfig()
    
    assert config.concurrent_connections == 100
    assert config.duration == 60
    assert len(config.data_sizes) == 3
    assert config.warmup_time == 5
    assert config.cooldown_time == 5

def test_metrics_collection(sample_metrics):
    """Test metrics collection and analysis."""
    stats = sample_metrics.get_stats()
    
    assert stats["total_requests"] == 100
    assert stats["errors"] == 5
    assert 0.1 <= stats["latency"]["mean"] <= 0.2
    assert len(stats["system"]["cpu_usage"]) == 2
    assert stats["bytes_per_second"] > 0

def test_reporter_output(temp_report_dir, sample_metrics):
    """Test benchmark report generation."""
    config = BenchmarkConfig(report_dir=temp_report_dir)
    reporter = BenchmarkReporter(config)
    
    # Generate report
    reporter.generate_report(sample_metrics, "test_run")
    
    # Check output files
    report_dir = Path(temp_report_dir)
    assert (report_dir / "benchmark_test_run.json").exists()
    assert (report_dir / "benchmark_test_run.html").exists()
    assert (report_dir / "benchmark_test_run_latency.png").exists()
    assert (report_dir / "benchmark_test_run_throughput.png").exists()
    assert (report_dir / "benchmark_test_run_system.png").exists()

def test_metrics_calculations():
    """Test accuracy of metrics calculations."""
    metrics = BenchmarkMetrics()
    
    # Add known values
    test_latencies = [0.1, 0.2, 0.3, 0.4, 0.5]
    test_bytes = 1000
    
    for latency in test_latencies:
        metrics.add_request(latency, test_bytes)
    
    stats = metrics.get_stats()
    
    assert stats["latency"]["min"] == 0.1
    assert stats["latency"]["max"] == 0.5
    assert stats["latency"]["mean"] == 0.3
    assert stats["bytes_per_second"] == (test_bytes * len(test_latencies)) / (0.5 - 0.1)

@pytest.mark.asyncio
async def test_benchmark_run(benchmark_config):
    """Test benchmark execution."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.read = AsyncMock(return_value=b"test")
    mock_session.post = AsyncMock(return_value=mock_response)
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        metrics = await run_benchmark("http://test", benchmark_config)
        
        stats = metrics.get_stats()
        assert stats["total_requests"] > 0
        assert stats["errors"] == 0
        assert "latency" in stats
        assert "system" in stats

def test_error_handling(benchmark_config):
    """Test error handling in metrics collection."""
    metrics = BenchmarkMetrics()
    
    # Add successful requests
    metrics.add_request(0.1, 1000)
    metrics.add_request(0.2, 1000)
    
    # Add errors
    metrics.add_error()
    metrics.add_error()
    
    stats = metrics.get_stats()
    assert stats["total_requests"] == 2
    assert stats["errors"] == 2
    assert stats["latency"]["mean"] == 0.15

def test_system_metrics_sampling():
    """Test system metrics sampling."""
    metrics = BenchmarkMetrics()
    
    # Sample metrics multiple times
    for _ in range(5):
        metrics.sample_system_metrics()
        time.sleep(0.1)
    
    stats = metrics.get_stats()
    assert len(metrics.cpu_samples) == 5
    assert len(metrics.memory_samples) == 5
    assert all(0 <= cpu <= 100 for cpu in metrics.cpu_samples)
    assert all(memory > 0 for memory in metrics.memory_samples)

@pytest.mark.asyncio
async def test_concurrent_requests(benchmark_config):
    """Test handling of concurrent requests."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.read = AsyncMock(return_value=b"test")
    mock_session.post = AsyncMock(return_value=mock_response)
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        metrics = await run_benchmark("http://test", benchmark_config)
        
        # Verify concurrent execution
        call_count = mock_session.post.call_count
        assert call_count >= benchmark_config.concurrent_connections

def test_report_data_validity(temp_report_dir, sample_metrics):
    """Test validity of report data."""
    config = BenchmarkConfig(report_dir=temp_report_dir)
    reporter = BenchmarkReporter(config)
    
    reporter.generate_report(sample_metrics, "test_run")
    
    # Load and verify JSON data
    with open(Path(temp_report_dir) / "benchmark_test_run.json") as f:
        data = json.load(f)
        
        assert "total_requests" in data
        assert "latency" in data
        assert "system" in data
        assert data["total_requests"] == 100
        assert data["errors"] == 5

def test_benchmark_config_validation():
    """Test benchmark configuration validation."""
    # Invalid concurrent connections
    with pytest.raises(ValueError):
        BenchmarkConfig(concurrent_connections=0)
        
    # Invalid duration
    with pytest.raises(ValueError):
        BenchmarkConfig(duration=-1)
        
    # Invalid data sizes
    with pytest.raises(ValueError):
        BenchmarkConfig(data_sizes=[0])

def test_metrics_thread_safety():
    """Test thread safety of metrics collection."""
    metrics = BenchmarkMetrics()
    
    async def concurrent_updates():
        tasks = []
        for i in range(100):
            tasks.append(asyncio.create_task(
                asyncio.to_thread(metrics.add_request, 0.1, 1000)
            ))
        await asyncio.gather(*tasks)
    
    asyncio.run(concurrent_updates())
    
    stats = metrics.get_stats()
    assert stats["total_requests"] == 100
