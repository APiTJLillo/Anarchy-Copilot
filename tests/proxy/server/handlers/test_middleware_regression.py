"""Regression tests for middleware performance."""
import pytest
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import statistics
import logging
import datetime
from dataclasses import dataclass
import numpy as np

from proxy.server.handlers.middleware import ProxyResponse
from proxy.server.handlers.http import HttpRequestHandler
from .test_middleware_performance import TimingMiddleware, measure_performance
from .middleware_perf_visualizer import PerformanceVisualizer

logger = logging.getLogger(__name__)

@dataclass
class PerformanceBaseline:
    """Performance baseline metrics."""
    throughput: float
    latency_p95: float
    memory_usage: int
    timestamp: str
    git_commit: Optional[str] = None
    
    @classmethod
    def load(cls, path: Path) -> Optional['PerformanceBaseline']:
        """Load baseline from file."""
        try:
            with open(path) as f:
                data = json.load(f)
                return cls(**data)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def save(self, path: Path):
        """Save baseline to file."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

class RegressionTest:
    """Regression test runner."""
    
    def __init__(self, 
                 baseline_dir: Path,
                 threshold: float = 0.15,  # 15% degradation threshold
                 window_size: int = 5):    # Number of samples for moving average
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(exist_ok=True)
        self.threshold = threshold
        self.window_size = window_size
        self.visualizer = PerformanceVisualizer(str(baseline_dir))
    
    def get_baseline(self) -> Optional[PerformanceBaseline]:
        """Get latest baseline if available."""
        baselines = sorted(self.baseline_dir.glob("baseline_*.json"))
        if not baselines:
            return None
        return PerformanceBaseline.load(baselines[-1])
    
    def save_baseline(self, baseline: PerformanceBaseline):
        """Save new baseline."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.baseline_dir / f"baseline_{timestamp}.json"
        baseline.save(path)
    
    def check_regression(self, current: Dict[str, float], 
                        baseline: PerformanceBaseline) -> Dict[str, Any]:
        """Check for performance regression."""
        regressions = {}
        
        # Check throughput
        if current["throughput"] < baseline.throughput * (1 - self.threshold):
            regressions["throughput"] = {
                "baseline": baseline.throughput,
                "current": current["throughput"],
                "degradation": (
                    (baseline.throughput - current["throughput"]) / 
                    baseline.throughput * 100
                )
            }
        
        # Check latency
        if current["latency_p95"] > baseline.latency_p95 * (1 + self.threshold):
            regressions["latency"] = {
                "baseline": baseline.latency_p95,
                "current": current["latency_p95"],
                "degradation": (
                    (current["latency_p95"] - baseline.latency_p95) / 
                    baseline.latency_p95 * 100
                )
            }
        
        # Check memory
        if current["memory_usage"] > baseline.memory_usage * (1 + self.threshold):
            regressions["memory"] = {
                "baseline": baseline.memory_usage,
                "current": current["memory_usage"],
                "degradation": (
                    (current["memory_usage"] - baseline.memory_usage) / 
                    baseline.memory_usage * 100
                )
            }
        
        return regressions

async def run_performance_test() -> Dict[str, float]:
    """Run standard performance test suite."""
    handler = HttpRequestHandler("test-conn")
    middleware = TimingMiddleware("regression-test")
    handler.register_middleware(middleware)
    
    # Run basic performance test
    with measure_performance() as metrics:
        for _ in range(1000):
            await handler.handle_client_data(
                b"GET /test HTTP/1.1\r\nHost: example.com\r\n\r\n"
            )
    
    stats = middleware.get_stats()
    return {
        "throughput": 1000 / metrics["duration"],
        "latency_p95": stats["p95"],
        "memory_usage": metrics["memory_delta"]
    }

@pytest.mark.regression
@pytest.mark.asyncio
async def test_performance_regression(tmp_path):
    """Test for performance regression against baseline."""
    regression_tester = RegressionTest(tmp_path / "regression_baselines")
    
    # Run current performance test
    current_metrics = await run_performance_test()
    
    # Get baseline
    baseline = regression_tester.get_baseline()
    if baseline is None:
        # No baseline exists, create one
        logger.info("No baseline found, creating new baseline")
        baseline = PerformanceBaseline(
            throughput=current_metrics["throughput"],
            latency_p95=current_metrics["latency_p95"],
            memory_usage=current_metrics["memory_usage"],
            timestamp=datetime.datetime.now().isoformat()
        )
        regression_tester.save_baseline(baseline)
        return
    
    # Check for regressions
    regressions = regression_tester.check_regression(current_metrics, baseline)
    
    # Generate comparison visualization
    comparison_data = {
        "metrics": {
            "baseline": {
                "throughput": baseline.throughput,
                "latency": baseline.latency_p95,
                "memory": baseline.memory_usage
            },
            "current": current_metrics
        },
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    regression_tester.visualizer.plot_comparison(comparison_data)
    
    # Handle regressions
    if regressions:
        regression_report = (
            "\nPerformance Regression Detected:\n" +
            "\n".join(
                f"{metric}:\n"
                f"  Baseline: {details['baseline']:.2f}\n"
                f"  Current:  {details['current']:.2f}\n"
                f"  Degradation: {details['degradation']:.1f}%"
                for metric, details in regressions.items()
            )
        )
        logger.error(regression_report)
        assert False, regression_report
    else:
        logger.info("No performance regression detected")
        
        # If performance improved significantly, update baseline
        if (current_metrics["throughput"] > baseline.throughput * 1.1 and
            current_metrics["latency_p95"] < baseline.latency_p95 * 0.9):
            logger.info("Performance improved significantly, updating baseline")
            new_baseline = PerformanceBaseline(
                throughput=current_metrics["throughput"],
                latency_p95=current_metrics["latency_p95"],
                memory_usage=current_metrics["memory_usage"],
                timestamp=datetime.datetime.now().isoformat()
            )
            regression_tester.save_baseline(new_baseline)

@pytest.mark.regression
def test_baseline_stability(tmp_path):
    """Test stability of performance measurements."""
    # Run multiple tests to check variance
    iterations = 5
    metrics_list = []
    
    async def collect_metrics():
        return await run_performance_test()
    
    # Collect metrics
    for _ in range(iterations):
        metrics = asyncio.run(collect_metrics())
        metrics_list.append(metrics)
    
    # Calculate variance
    variances = {
        metric: statistics.variance([m[metric] for m in metrics_list])
        for metric in metrics_list[0].keys()
    }
    
    # Check stability
    max_variance_pct = 10  # Maximum allowed variance percentage
    for metric, variance in variances.items():
        mean = statistics.mean([m[metric] for m in metrics_list])
        variance_pct = (np.sqrt(variance) / mean) * 100
        assert variance_pct < max_variance_pct, (
            f"Unstable {metric} measurements: {variance_pct:.1f}% variance"
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])
