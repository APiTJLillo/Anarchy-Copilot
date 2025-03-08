"""Benchmarking utilities for HTTPS proxy performance testing."""
import asyncio
import aiohttp
import time
import json
from datetime import datetime
import statistics
from pathlib import Path
import logging
import psutil
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    concurrent_connections: int = 100
    request_count: int = 1000
    data_sizes: List[int] = None  # In bytes
    duration: int = 60  # Seconds
    warmup_time: int = 5  # Seconds
    cooldown_time: int = 5  # Seconds
    report_dir: str = "benchmark_reports"

    def __post_init__(self):
        if self.data_sizes is None:
            self.data_sizes = [
                1024,        # 1KB
                64 * 1024,   # 64KB
                1024 * 1024  # 1MB
            ]

class BenchmarkMetrics:
    """Collects and analyzes benchmark metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_times: List[float] = []
        self.error_count = 0
        self.bytes_transferred = 0
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        
    def add_request(self, duration: float, bytes_count: int):
        """Record a completed request."""
        self.request_times.append(duration)
        self.bytes_transferred += bytes_count
        
    def add_error(self):
        """Record an error."""
        self.error_count += 1
        
    def sample_system_metrics(self):
        """Sample current system metrics."""
        self.cpu_samples.append(psutil.cpu_percent())
        self.memory_samples.append(psutil.Process().memory_info().rss)
        
    def get_stats(self) -> Dict:
        """Get statistical analysis of metrics."""
        duration = time.time() - self.start_time
        
        if not self.request_times:
            return {}
            
        return {
            "duration_seconds": duration,
            "total_requests": len(self.request_times),
            "errors": self.error_count,
            "requests_per_second": len(self.request_times) / duration,
            "bytes_per_second": self.bytes_transferred / duration,
            "latency": {
                "min": min(self.request_times),
                "max": max(self.request_times),
                "mean": statistics.mean(self.request_times),
                "median": statistics.median(self.request_times),
                "p95": statistics.quantiles(self.request_times, n=20)[18],
                "p99": statistics.quantiles(self.request_times, n=100)[98]
            },
            "system": {
                "cpu_usage": {
                    "mean": statistics.mean(self.cpu_samples),
                    "max": max(self.cpu_samples)
                },
                "memory_mb": {
                    "mean": statistics.mean(self.memory_samples) / 1024 / 1024,
                    "max": max(self.memory_samples) / 1024 / 1024
                }
            }
        }

class BenchmarkReporter:
    """Generates benchmark reports."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.report_dir = Path(config.report_dir)
        self.report_dir.mkdir(exist_ok=True)
        
    def generate_report(self, metrics: BenchmarkMetrics, test_name: str):
        """Generate a report from benchmark metrics."""
        stats = metrics.get_stats()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"benchmark_{test_name}_{timestamp}"
        
        # Save raw data
        with open(f"{report_path}.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        # Generate plots
        self._plot_latency_distribution(metrics.request_times, report_path)
        self._plot_throughput(metrics.request_times, report_path)
        self._plot_system_metrics(metrics, report_path)
        
        # Generate HTML report
        self._generate_html_report(stats, test_name, report_path)
        
    def _plot_latency_distribution(self, request_times: List[float], report_path: Path):
        """Plot latency distribution."""
        plt.figure(figsize=(10, 6))
        plt.hist(request_times, bins=50)
        plt.title("Request Latency Distribution")
        plt.xlabel("Latency (seconds)")
        plt.ylabel("Count")
        plt.savefig(f"{report_path}_latency.png")
        plt.close()
        
    def _plot_throughput(self, request_times: List[float], report_path: Path):
        """Plot throughput over time."""
        plt.figure(figsize=(10, 6))
        cumulative_times = np.cumsum(request_times)
        request_counts = np.arange(1, len(request_times) + 1)
        throughput = request_counts / cumulative_times
        plt.plot(cumulative_times, throughput)
        plt.title("Throughput Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Requests per Second")
        plt.savefig(f"{report_path}_throughput.png")
        plt.close()
        
    def _plot_system_metrics(self, metrics: BenchmarkMetrics, report_path: Path):
        """Plot system metrics over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        timestamps = np.linspace(0, metrics.get_stats()["duration_seconds"], 
                               len(metrics.cpu_samples))
        
        ax1.plot(timestamps, metrics.cpu_samples)
        ax1.set_title("CPU Usage Over Time")
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("CPU %")
        
        memory_mb = [m / 1024 / 1024 for m in metrics.memory_samples]
        ax2.plot(timestamps, memory_mb)
        ax2.set_title("Memory Usage Over Time")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Memory (MB)")
        
        plt.tight_layout()
        plt.savefig(f"{report_path}_system.png")
        plt.close()
        
    def _generate_html_report(self, stats: Dict, test_name: str, report_path: Path):
        """Generate HTML report."""
        html = f"""
        <html>
        <head>
            <title>Benchmark Report - {test_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Benchmark Report - {test_name}</h1>
            <div class="metric">
                <h2>Summary</h2>
                <p>Total Requests: {stats['total_requests']}</p>
                <p>Duration: {stats['duration_seconds']:.2f} seconds</p>
                <p>Requests/second: {stats['requests_per_second']:.2f}</p>
                <p>Error rate: {(stats['errors'] / stats['total_requests'] * 100):.2f}%</p>
            </div>
            
            <div class="metric">
                <h2>Latency (seconds)</h2>
                <p>Min: {stats['latency']['min']:.3f}</p>
                <p>Max: {stats['latency']['max']:.3f}</p>
                <p>Mean: {stats['latency']['mean']:.3f}</p>
                <p>P95: {stats['latency']['p95']:.3f}</p>
                <p>P99: {stats['latency']['p99']:.3f}</p>
            </div>
            
            <div class="metric">
                <h2>System Metrics</h2>
                <p>CPU Usage (mean): {stats['system']['cpu_usage']['mean']:.1f}%</p>
                <p>Memory Usage (max): {stats['system']['memory_mb']['max']:.1f} MB</p>
            </div>
            
            <div class="chart">
                <h2>Charts</h2>
                <img src="{report_path}_latency.png" alt="Latency Distribution">
                <img src="{report_path}_throughput.png" alt="Throughput">
                <img src="{report_path}_system.png" alt="System Metrics">
            </div>
        </body>
        </html>
        """
        
        with open(f"{report_path}.html", "w") as f:
            f.write(html)

async def run_benchmark(proxy_url: str, config: BenchmarkConfig) -> BenchmarkMetrics:
    """Run a benchmark against the proxy."""
    metrics = BenchmarkMetrics()
    
    # Setup client session
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=False),
        proxy=proxy_url
    ) as session:
        # Warmup
        logger.info("Starting warmup...")
        await asyncio.sleep(config.warmup_time)
        
        # Run benchmark
        logger.info("Starting benchmark...")
        start_time = time.time()
        
        async def make_request(size: int):
            try:
                data = b"X" * size
                req_start = time.time()
                async with session.post(
                    "https://example.com/test",
                    data=data
                ) as response:
                    await response.read()
                    duration = time.time() - req_start
                    metrics.add_request(duration, size)
            except Exception as e:
                logger.error(f"Request error: {e}")
                metrics.add_error()
        
        # Create request batches
        while time.time() - start_time < config.duration:
            # Sample system metrics
            metrics.sample_system_metrics()
            
            # Create batch of requests
            tasks = []
            for size in config.data_sizes:
                tasks.extend([
                    make_request(size)
                    for _ in range(config.concurrent_connections // len(config.data_sizes))
                ])
            
            await asyncio.gather(*tasks)
            
        # Cooldown
        logger.info("Starting cooldown...")
        await asyncio.sleep(config.cooldown_time)
    
    return metrics

if __name__ == "__main__":
    # Example usage
    async def main():
        config = BenchmarkConfig()
        metrics = await run_benchmark("http://localhost:8083", config)
        reporter = BenchmarkReporter(config)
        reporter.generate_report(metrics, "baseline")
        
    asyncio.run(main())
