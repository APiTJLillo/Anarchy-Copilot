"""Performance profiling tests for middleware and handlers."""
import pytest
import asyncio
import cProfile
import pstats
import io
import sys
import tracemalloc
import logging
import functools
from pathlib import Path
from typing import Dict, Any, Callable, List
from concurrent.futures import ThreadPoolExecutor

from .mock_server import MockHttpsServer
from .run_perf_tests import PerformanceTestRunner
from proxy.server.handlers.middleware import ProxyResponse

logger = logging.getLogger(__name__)

def profile_async(func):
    """Decorator to profile async functions."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            profiler.disable()
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
            stats.print_stats()
            logger.info(f"Profile for {func.__name__}:\n{s.getvalue()}")
    return wrapper

class MemoryProfiler:
    """Track memory usage during tests."""
    
    def __init__(self):
        self.snapshots: List[tracemalloc.Snapshot] = []
    
    def start(self):
        """Start memory profiling."""
        tracemalloc.start()
    
    def take_snapshot(self):
        """Take memory snapshot."""
        self.snapshots.append(tracemalloc.take_snapshot())
    
    def compare_snapshots(self, start_idx: int, end_idx: int) -> List[str]:
        """Compare two snapshots."""
        start = self.snapshots[start_idx]
        end = self.snapshots[end_idx]
        
        stats = end.compare_to(start, 'lineno')
        return [str(stat) for stat in stats[:10]]  # Top 10 differences
    
    def stop(self):
        """Stop memory profiling."""
        tracemalloc.stop()

class ProfilingMiddleware:
    """Middleware for detailed performance profiling."""
    
    def __init__(self):
        self.call_times: List[float] = []
        self.memory_usage: List[int] = []
        self.frame_times: List[float] = []
    
    async def __call__(self, request: dict) -> ProxyResponse:
        start_time = asyncio.get_event_loop().time()
        
        # Profile memory
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        
        try:
            # Simulate processing
            await asyncio.sleep(0.001)
            return None
            
        finally:
            # Record timing
            end_time = asyncio.get_event_loop().time()
            self.call_times.append(end_time - start_time)
            
            # Record memory
            end_snapshot = tracemalloc.take_snapshot()
            diff = end_snapshot.compare_to(start_snapshot, 'lineno')
            self.memory_usage.append(sum(stat.size_diff for stat in diff))
            
            tracemalloc.stop()
            
            # Record frame time
            self.frame_times.append(end_time - start_time)

@pytest.fixture
async def profiling_env(tmp_path):
    """Set up profiling environment."""
    # Create profiling directories
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    
    # Set up server with profiling middleware
    server = MockHttpsServer({"port": 0})
    middleware = ProfilingMiddleware()
    
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
            "profile_dir": profile_dir
        }
    finally:
        await server.stop()

@profile_async
@pytest.mark.profiling
@pytest.mark.asyncio
async def test_middleware_performance_profile(profiling_env):
    """Profile middleware performance."""
    # Configure test runner
    config = {
        "concurrent_connections": 100,
        "request_count": 1000,
        "warmup_requests": 50,
        "output_dir": str(profiling_env["profile_dir"])
    }
    
    runner = PerformanceTestRunner(config)
    results = await runner.run_tests()
    
    # Analyze middleware timing
    middleware = profiling_env["middleware"]
    avg_call_time = sum(middleware.call_times) / len(middleware.call_times)
    p95_call_time = sorted(middleware.call_times)[int(len(middleware.call_times) * 0.95)]
    
    logger.info(f"Average middleware call time: {avg_call_time:.6f}s")
    logger.info(f"P95 middleware call time: {p95_call_time:.6f}s")
    
    assert avg_call_time < 0.01, "Middleware too slow"

@pytest.mark.profiling
@pytest.mark.asyncio
async def test_memory_usage_profile(profiling_env):
    """Profile memory usage patterns."""
    memory_profiler = MemoryProfiler()
    memory_profiler.start()
    
    try:
        # Run test with memory tracking
        runner = PerformanceTestRunner({
            "concurrent_connections": 50,
            "request_count": 500,
            "output_dir": str(profiling_env["profile_dir"])
        })
        
        memory_profiler.take_snapshot()  # Before test
        results = await runner.run_tests()
        memory_profiler.take_snapshot()  # After test
        
        # Analyze memory usage
        diff_stats = memory_profiler.compare_snapshots(0, 1)
        logger.info("Memory usage changes:\n%s", "\n".join(diff_stats))
        
        # Verify no major leaks
        middleware = profiling_env["middleware"]
        final_memory = max(middleware.memory_usage)
        assert final_memory < 100 * 1024 * 1024, "Excessive memory usage"
        
    finally:
        memory_profiler.stop()

@pytest.mark.profiling
@pytest.mark.asyncio
async def test_cpu_profile_analysis(profiling_env):
    """Analyze CPU usage patterns."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        # Run CPU-intensive test
        runner = PerformanceTestRunner({
            "concurrent_connections": 200,
            "request_count": 2000,
            "output_dir": str(profiling_env["profile_dir"])
        })
        results = await runner.run_tests()
        
    finally:
        profiler.disable()
        
        # Analyze CPU profile
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumtime')
        
        # Save profile results
        profile_path = profiling_env["profile_dir"] / "cpu_profile.prof"
        stats.dump_stats(str(profile_path))
        
        # Log top time consumers
        with io.StringIO() as stream:
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats('cumtime')
            stats.print_stats(20)  # Top 20 time consumers
            logger.info("CPU Profile:\n%s", stream.getvalue())

@pytest.mark.profiling
@pytest.mark.asyncio
async def test_concurrency_profile(profiling_env):
    """Profile concurrent request handling."""
    async def measure_concurrency():
        # Track active requests
        active = 0
        max_active = 0
        active_times = []
        
        async def tracked_request():
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            start = asyncio.get_event_loop().time()
            
            try:
                await runner.run_tests()
            finally:
                active -= 1
                duration = asyncio.get_event_loop().time() - start
                active_times.append(duration)
        
        # Run concurrent requests
        runner = PerformanceTestRunner({
            "concurrent_connections": 10,
            "request_count": 100,
            "output_dir": str(profiling_env["profile_dir"])
        })
        
        tasks = [tracked_request() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        return {
            "max_concurrent": max_active,
            "avg_duration": sum(active_times) / len(active_times)
        }
    
    # Run concurrency test
    concurrency_stats = await measure_concurrency()
    logger.info("Concurrency profile: %s", concurrency_stats)
    
    assert concurrency_stats["max_concurrent"] <= 5, "Unexpected concurrency level"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])
