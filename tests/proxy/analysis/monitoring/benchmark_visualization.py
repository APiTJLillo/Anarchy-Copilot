"""Performance benchmarks for visualization components."""

import pytest
import asyncio
from datetime import datetime, timedelta
import time
import numpy as np
from typing import List, Dict
import json
import cProfile
import pstats
import tempfile
from pathlib import Path
import memory_profiler

from proxy.analysis.monitoring.metrics import MetricValue, TimeseriesMetric
from proxy.analysis.monitoring.alerts import (
    Alert,
    AlertSeverity,
    AlertManager,
    AlertRule
)
from proxy.analysis.monitoring.storage import (
    MetricStore,
    TimeseriesStore
)
from proxy.analysis.monitoring.visualization import MonitoringDashboard

def generate_test_data(
    num_metrics: int,
    num_points: int,
    num_alerts: int
) -> tuple[List[TimeseriesMetric], List[Alert]]:
    """Generate test data for benchmarking."""
    # Generate metrics
    now = datetime.now()
    metrics = []
    for i in range(num_metrics):
        values = np.random.normal(100, 10, num_points).tolist()
        timestamps = [
            now - timedelta(minutes=j)
            for j in range(num_points)
        ]
        metrics.append(TimeseriesMetric(
            name=f"test.metric.{i}",
            values=values,
            timestamps=timestamps,
            tags={"test": str(i)}
        ))
    
    # Generate alerts
    alerts = []
    for i in range(num_alerts):
        alerts.append(Alert(
            id=f"test_{i}",
            rule_name=f"rule_{i}",
            metric_name=f"test.metric.{i % num_metrics}",
            value=110.0,
            threshold=100.0,
            severity=AlertSeverity.WARNING,
            timestamp=now - timedelta(minutes=i),
            description=f"Test alert {i}"
        ))
    
    return metrics, alerts

class MockStore:
    """Mock store for benchmarking."""
    
    def __init__(self, metrics: List[TimeseriesMetric]):
        self.metrics = {m.name: m for m in metrics}
    
    async def get_metric(self, *args, **kwargs):
        return self.metrics[args[0]]
    
    async def get_timeseries(self, *args, **kwargs):
        return self.metrics[args[0]]

class BenchmarkConfig:
    """Benchmark configuration."""
    SMALL_LOAD = {"metrics": 10, "points": 100, "alerts": 5}
    MEDIUM_LOAD = {"metrics": 50, "points": 1000, "alerts": 20}
    LARGE_LOAD = {"metrics": 200, "points": 5000, "alerts": 100}
    STRESS_LOAD = {"metrics": 1000, "points": 10000, "alerts": 500}

@pytest.mark.benchmark
class TestVisualizationPerformance:
    """Benchmark visualization performance."""
    
    @pytest.mark.parametrize("load", [
        BenchmarkConfig.SMALL_LOAD,
        BenchmarkConfig.MEDIUM_LOAD,
        BenchmarkConfig.LARGE_LOAD
    ])
    def test_dashboard_initialization(self, load, benchmark):
        """Benchmark dashboard initialization."""
        metrics, alerts = generate_test_data(
            load["metrics"],
            load["points"],
            load["alerts"]
        )
        
        def create_dashboard():
            store = MockStore(metrics)
            dashboard = MonitoringDashboard(
                alert_manager=None,
                metric_store=store,
                timeseries_store=store
            )
            return dashboard
        
        result = benchmark(create_dashboard)
        assert result is not None
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("load", [
        BenchmarkConfig.SMALL_LOAD,
        BenchmarkConfig.MEDIUM_LOAD,
        BenchmarkConfig.LARGE_LOAD
    ])
    async def test_update_performance(self, load, benchmark):
        """Benchmark dashboard update performance."""
        metrics, alerts = generate_test_data(
            load["metrics"],
            load["points"],
            load["alerts"]
        )
        store = MockStore(metrics)
        dashboard = MonitoringDashboard(
            alert_manager=None,
            metric_store=store,
            timeseries_store=store
        )
        
        # Get update callback
        update_callback = None
        for callback in dashboard.app.callback_map.values():
            if "update_dashboard" in str(callback.callback):
                update_callback = callback.callback
                break
        
        assert update_callback is not None
        
        async def run_update():
            return await update_callback(1, 1, "1h")
        
        result = await benchmark(run_update)
        assert result is not None
    
    @pytest.mark.parametrize("load", [
        BenchmarkConfig.SMALL_LOAD,
        BenchmarkConfig.MEDIUM_LOAD,
        BenchmarkConfig.LARGE_LOAD
    ])
    def test_memory_usage(self, load):
        """Test memory usage under different loads."""
        metrics, alerts = generate_test_data(
            load["metrics"],
            load["points"],
            load["alerts"]
        )
        store = MockStore(metrics)
        
        @memory_profiler.profile
        def create_and_update():
            dashboard = MonitoringDashboard(
                alert_manager=None,
                metric_store=store,
                timeseries_store=store
            )
            # Force layout creation
            _ = dashboard.app.layout
            return dashboard
        
        dashboard = create_and_update()
        assert dashboard is not None

    def test_component_profiling(self):
        """Profile individual component rendering."""
        metrics, alerts = generate_test_data(
            BenchmarkConfig.MEDIUM_LOAD["metrics"],
            BenchmarkConfig.MEDIUM_LOAD["points"],
            BenchmarkConfig.MEDIUM_LOAD["alerts"]
        )
        store = MockStore(metrics)
        dashboard = MonitoringDashboard(
            alert_manager=None,
            metric_store=store,
            timeseries_store=store
        )
        
        with tempfile.NamedTemporaryFile(suffix='.prof') as tmp:
            profiler = cProfile.Profile()
            profiler.enable()
            
            # Profile layout creation
            _ = dashboard.app.layout
            
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.dump_stats(tmp.name)
            
            # Load and check profiling data
            loaded_stats = pstats.Stats(tmp.name)
            
            # Verify profiling data exists
            assert loaded_stats.total_calls > 0
            top_functions = loaded_stats.stats.items()
            assert len(top_functions) > 0

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_stress_load(self):
        """Test behavior under stress load."""
        metrics, alerts = generate_test_data(
            BenchmarkConfig.STRESS_LOAD["metrics"],
            BenchmarkConfig.STRESS_LOAD["points"],
            BenchmarkConfig.STRESS_LOAD["alerts"]
        )
        store = MockStore(metrics)
        dashboard = MonitoringDashboard(
            alert_manager=None,
            metric_store=store,
            timeseries_store=store
        )
        
        # Get update callback
        update_callback = None
        for callback in dashboard.app.callback_map.values():
            if "update_dashboard" in str(callback.callback):
                update_callback = callback.callback
                break
        
        assert update_callback is not None
        
        # Measure response times
        start_time = time.time()
        response_times = []
        
        for _ in range(10):  # Run 10 updates
            iteration_start = time.time()
            result = await update_callback(1, 1, "1h")
            response_times.append(time.time() - iteration_start)
            
            assert result is not None  # Verify update completed
        
        total_time = time.time() - start_time
        avg_response = sum(response_times) / len(response_times)
        max_response = max(response_times)
        
        # Log performance metrics
        print(f"\nStress Test Results:")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average Response: {avg_response:.2f}s")
        print(f"Max Response: {max_response:.2f}s")
        print(f"Updates/Second: {10/total_time:.2f}")
        
        # Assert performance requirements
        assert avg_response < 5.0  # Average response under 5 seconds
        assert max_response < 10.0  # Max response under 10 seconds

def run_benchmarks():
    """Run all benchmarks and generate report."""
    import pytest
    import json
    from datetime import datetime
    
    # Run benchmarks
    results = pytest.main([
        __file__,
        "-v",
        "--benchmark-only",
        "--benchmark-json=benchmark_results.json"
    ])
    
    # Load results
    with open("benchmark_results.json") as f:
        data = json.load(f)
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": len(data["benchmarks"]),
            "total_time": sum(b["stats"]["total"] for b in data["benchmarks"]),
            "fastest_test": min(b["stats"]["mean"] for b in data["benchmarks"]),
            "slowest_test": max(b["stats"]["mean"] for b in data["benchmarks"])
        },
        "tests": [{
            "name": b["name"],
            "mean": b["stats"]["mean"],
            "std_dev": b["stats"]["stddev"],
            "rounds": b["stats"]["rounds"],
            "median": b["stats"]["median"],
            "iterations": b["stats"]["iterations"]
        } for b in data["benchmarks"]]
    }
    
    # Save report
    with open("visualization_benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    return results

if __name__ == "__main__":
    run_benchmarks()
