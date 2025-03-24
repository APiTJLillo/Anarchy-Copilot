"""Performance and load tests for HTTPS proxy."""
import pytest
import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict
import logging
from concurrent.futures import ThreadPoolExecutor
import psutil
import resource

from proxy.server.https_intercept_protocol import HttpsInterceptProtocol
from proxy.server.tls.connection_manager import connection_mgr

logger = logging.getLogger(__name__)

@pytest.fixture
def performance_metrics():
    """Collect system performance metrics."""
    class Metrics:
        def __init__(self):
            self.start_time = time.time()
            self.start_cpu = psutil.cpu_percent(interval=None)
            self.start_memory = psutil.Process().memory_info().rss
            self.request_times: List[float] = []
            self.errors = 0
            self.completed = 0
            
        def add_request_time(self, duration: float):
            self.request_times.append(duration)
            self.completed += 1
            
        def add_error(self):
            self.errors += 1
            
        def get_stats(self) -> Dict:
            end_time = time.time()
            end_cpu = psutil.cpu_percent(interval=None)
            end_memory = psutil.Process().memory_info().rss
            
            if self.request_times:
                return {
                    "duration": end_time - self.start_time,
                    "requests_completed": self.completed,
                    "errors": self.errors,
                    "avg_request_time": statistics.mean(self.request_times),
                    "median_request_time": statistics.median(self.request_times),
                    "p95_request_time": statistics.quantiles(self.request_times, n=20)[18],
                    "min_request_time": min(self.request_times),
                    "max_request_time": max(self.request_times),
                    "memory_delta": end_memory - self.start_memory,
                    "cpu_usage": end_cpu - self.start_cpu,
                    "requests_per_second": self.completed / (end_time - self.start_time)
                }
            return {}
            
    return Metrics()

@pytest.mark.asyncio
async def test_concurrent_connection_limit(proxy_server, https_echo_server, 
                                        proxy_session, performance_metrics):
    """Test proxy behavior with many concurrent connections."""
    echo_host, echo_port = https_echo_server
    url = f"https://localhost:{echo_port}/test"
    concurrent_requests = 100
    
    async def make_request():
        try:
            start = time.time()
            async with proxy_session.get(url) as response:
                await response.read()
                duration = time.time() - start
                performance_metrics.add_request_time(duration)
                return response.status
        except Exception as e:
            logger.error(f"Request error: {e}")
            performance_metrics.add_error()
            return None
    
    # Send concurrent requests
    tasks = [make_request() for _ in range(concurrent_requests)]
    results = await asyncio.gather(*tasks)
    
    # Analyze results
    stats = performance_metrics.get_stats()
    logger.info(f"Performance stats: {stats}")
    
    # Verify success rate
    success_rate = len([r for r in results if r == 200]) / concurrent_requests
    assert success_rate > 0.95  # Allow 5% failure rate
    assert stats["requests_per_second"] > 10  # Minimum throughput

@pytest.mark.asyncio
async def test_large_data_throughput(proxy_server, https_echo_server, 
                                   proxy_session, performance_metrics):
    """Test proxy performance with large data transfers."""
    echo_host, echo_port = https_echo_server
    url = f"https://localhost:{echo_port}/test"
    data_size = 10 * 1024 * 1024  # 10MB
    test_data = b"X" * data_size
    
    async def transfer_data():
        try:
            start = time.time()
            async with proxy_session.post(url, data=test_data) as response:
                response_data = await response.read()
                duration = time.time() - start
                performance_metrics.add_request_time(duration)
                return len(response_data)
        except Exception as e:
            logger.error(f"Transfer error: {e}")
            performance_metrics.add_error()
            return 0
    
    # Perform multiple transfers
    tasks = [transfer_data() for _ in range(5)]
    results = await asyncio.gather(*tasks)
    
    # Analyze throughput
    stats = performance_metrics.get_stats()
    total_bytes = sum(results)
    duration = stats["duration"]
    mbps = (total_bytes / 1024 / 1024) / duration
    
    logger.info(f"Throughput: {mbps:.2f} MB/s")
    assert mbps > 1.0  # Minimum 1 MB/s throughput

@pytest.mark.asyncio
async def test_memory_usage(proxy_server, https_echo_server, 
                          proxy_session, performance_metrics):
    """Test memory usage under sustained load."""
    echo_host, echo_port = https_echo_server
    url = f"https://localhost:{echo_port}/test"
    
    async def sustained_load():
        for _ in range(100):
            try:
                start = time.time()
                async with proxy_session.get(url) as response:
                    await response.read()
                    duration = time.time() - start
                    performance_metrics.add_request_time(duration)
            except Exception as e:
                logger.error(f"Request error: {e}")
                performance_metrics.add_error()
            await asyncio.sleep(0.01)  # Small delay between requests
    
    # Run sustained load
    tasks = [sustained_load() for _ in range(5)]
    await asyncio.gather(*tasks)
    
    # Check memory usage
    stats = performance_metrics.get_stats()
    memory_mb = stats["memory_delta"] / 1024 / 1024
    logger.info(f"Memory increase: {memory_mb:.2f} MB")
    assert memory_mb < 100  # Less than 100MB increase

@pytest.mark.asyncio
async def test_connection_pool_reuse(proxy_server, https_echo_server, 
                                   proxy_session, performance_metrics):
    """Test connection pooling efficiency."""
    echo_host, echo_port = https_echo_server
    url = f"https://localhost:{echo_port}/test"
    
    # Make sequential requests
    for _ in range(20):
        try:
            start = time.time()
            async with proxy_session.get(url) as response:
                await response.read()
                duration = time.time() - start
                performance_metrics.add_request_time(duration)
        except Exception as e:
            logger.error(f"Request error: {e}")
            performance_metrics.add_error()
    
    # Analyze connection reuse
    stats = performance_metrics.get_stats()
    assert stats["avg_request_time"] < 0.1  # Average under 100ms
    assert connection_mgr.get_active_connection_count() <= 2  # Connection pooling working

@pytest.mark.asyncio
async def test_resource_limits(proxy_server, https_echo_server, 
                             proxy_session, performance_metrics):
    """Test proxy behavior near system resource limits."""
    echo_host, echo_port = https_echo_server
    url = f"https://localhost:{echo_port}/test"
    
    # Set artificial file descriptor limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(soft, 256)  # Limit to 256 file descriptors
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    
    async def make_request():
        try:
            start = time.time()
            async with proxy_session.get(url) as response:
                await response.read()
                duration = time.time() - start
                performance_metrics.add_request_time(duration)
                return True
        except Exception as e:
            logger.error(f"Request error: {e}")
            performance_metrics.add_error()
            return False
    
    # Send many requests
    tasks = [make_request() for _ in range(new_soft * 2)]
    results = await asyncio.gather(*tasks)
    
    # Verify graceful handling
    stats = performance_metrics.get_stats()
    success_count = len([r for r in results if r])
    assert success_count > 0  # Some requests should succeed
    assert stats["errors"] > 0  # Some should fail due to resource limits

@pytest.mark.asyncio
async def test_load_recovery(proxy_server, https_echo_server, 
                           proxy_session, performance_metrics):
    """Test proxy recovery after high load."""
    echo_host, echo_port = https_echo_server
    url = f"https://localhost:{echo_port}/test"
    
    # Generate high load
    high_load_tasks = [
        proxy_session.get(url)
        for _ in range(50)
    ]
    
    # Start all requests
    responses = await asyncio.gather(*high_load_tasks, return_exceptions=True)
    
    # Allow system to recover
    await asyncio.sleep(1)
    
    # Test normal operation
    async with proxy_session.get(url) as response:
        assert response.status == 200
        
    # Verify metrics
    stats = performance_metrics.get_stats()
    assert connection_mgr.get_active_connection_count() < 10  # Connections cleaned up
