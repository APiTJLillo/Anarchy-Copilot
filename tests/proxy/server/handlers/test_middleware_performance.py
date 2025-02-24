"""Performance tests for middleware chain execution."""
import pytest
import asyncio
import time
from typing import List, Dict, Any, Optional
import statistics
import logging
import contextlib
import psutil
import os
from pathlib import Path

from proxy.server.handlers.middleware import ProxyResponse, proxy_middleware
from proxy.server.handlers.http import HttpRequestHandler
from .test_middleware import sample_request
from .middleware_perf_visualizer import create_visualization

logger = logging.getLogger(__name__)

class TimingMiddleware:
    """Middleware that measures its execution time."""
    def __init__(self, name: str, delay: float = 0):
        self.name = name
        self.delay = delay
        self.execution_times: List[float] = []
    
    async def __call__(self, request: Dict[str, Any]) -> Optional[ProxyResponse]:
        start = time.perf_counter()
        if self.delay:
            await asyncio.sleep(self.delay)
        result = None
        end = time.perf_counter()
        self.execution_times.append(end - start)
        return result

    def get_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        if not self.execution_times:
            return {}
        return {
            "min": min(self.execution_times),
            "max": max(self.execution_times),
            "mean": statistics.mean(self.execution_times),
            "median": statistics.median(self.execution_times),
            "count": len(self.execution_times),
            "p95": statistics.quantiles(self.execution_times, n=20)[18],
            "p99": statistics.quantiles(self.execution_times, n=100)[98]
        }

@pytest.fixture
def perf_output_dir(tmp_path):
    """Create performance output directory."""
    output_dir = tmp_path / "perf_results"
    output_dir.mkdir(exist_ok=True)
    return output_dir

@contextlib.contextmanager
def measure_performance():
    """Context manager to measure system metrics."""
    process = psutil.Process()
    start_cpu = process.cpu_percent()
    start_mem = process.memory_info().rss
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        end_cpu = process.cpu_percent()
        end_mem = process.memory_info().rss
        
        metrics = {
            "duration": end_time - start_time,
            "cpu_delta": end_cpu - start_cpu,
            "memory_delta": end_mem - start_mem
        }
        return metrics

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_middleware_performance(perf_output_dir):
    """Run comprehensive middleware performance tests."""
    test_results = {
        "execution_times": {},
        "throughput": {},
        "memory_profile": [],
        "latency_matrix": []
    }
    
    # Single middleware test
    logger.info("Testing single middleware performance...")
    middleware = TimingMiddleware("single")
    handler = HttpRequestHandler("test-conn")
    handler.register_middleware(middleware)
    
    with measure_performance() as metrics:
        for _ in range(1000):
            await handler.handle_client_data(
                b"GET /test HTTP/1.1\r\nHost: example.com\r\n\r\n"
            )
    
    test_results["execution_times"]["single"] = middleware.execution_times
    test_results["throughput"]["single"] = 1000 / metrics["duration"]
    
    # Chain test
    logger.info("Testing middleware chain performance...")
    chain_middlewares = [
        TimingMiddleware(f"chain_{i}", delay=0.0001*i)
        for i in range(3)
    ]
    
    handler = HttpRequestHandler("test-conn")
    for mw in chain_middlewares:
        handler.register_middleware(mw)
    
    with measure_performance() as metrics:
        for _ in range(100):
            await handler.handle_client_data(
                b"GET /test HTTP/1.1\r\nHost: example.com\r\n\r\n"
            )
    
    for mw in chain_middlewares:
        test_results["execution_times"][mw.name] = mw.execution_times
    test_results["throughput"]["chain"] = 100 / metrics["duration"]
    
    # Concurrent test
    logger.info("Testing concurrent performance...")
    concurrent_mw = TimingMiddleware("concurrent")
    handler = HttpRequestHandler("test-conn")
    handler.register_middleware(concurrent_mw)
    
    latency_matrix = []
    concurrent_requests = 50
    iterations = 20
    
    async def make_requests():
        request_latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            await handler.handle_client_data(
                b"GET /test HTTP/1.1\r\nHost: example.com\r\n\r\n"
            )
            request_latencies.append(time.perf_counter() - start)
        return request_latencies
    
    with measure_performance() as metrics:
        tasks = [make_requests() for _ in range(concurrent_requests)]
        latencies = await asyncio.gather(*tasks)
        
    test_results["latency_matrix"] = latencies
    test_results["throughput"]["concurrent"] = (
        concurrent_requests * iterations / metrics["duration"]
    )
    
    # Memory profiling
    logger.info("Profiling memory usage...")
    process = psutil.Process()
    samples = 50
    
    for _ in range(samples):
        memory_info = process.memory_info()
        test_results["memory_profile"].append({
            "rss": memory_info.rss,
            "uss": getattr(memory_info, "uss", 0)  # Not available on all platforms
        })
        await asyncio.sleep(0.1)
    
    # Calculate overall metrics
    test_results.update({
        "avg_throughput": sum(test_results["throughput"].values()) / len(test_results["throughput"]),
        "mean_latency": statistics.mean(concurrent_mw.execution_times),
        "memory_usage": process.memory_info().rss
    })
    
    # Generate visualization report
    logger.info("Generating performance report...")
    report_path = create_visualization(test_results, str(perf_output_dir))
    
    logger.info(f"Performance report generated: {report_path}")
    
    # Performance assertions
    assert test_results["avg_throughput"] > 100, "Throughput below minimum threshold"
    assert test_results["mean_latency"] < 0.01, "Latency above maximum threshold"
    
    # Log summary
    logger.info(
        f"\nPerformance Summary:\n"
        f"Average Throughput: {test_results['avg_throughput']:.2f} req/s\n"
        f"Mean Latency: {test_results['mean_latency']*1000:.2f}ms\n"
        f"Memory Usage: {test_results['memory_usage']/1024/1024:.2f}MB"
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])
