#!/usr/bin/env python3
"""Performance tests for alert throttling system."""

import pytest
import time
import random
import statistics
from pathlib import Path
from typing import Dict, Any, List, Generator, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import psutil
import os
from contextlib import contextmanager

from scripts.alert_throttling import (
    AlertThrottler,
    ThrottlingConfig,
    AlertKey
)

@contextmanager
def measure_time_and_memory() -> Generator[None, None, Tuple[float, float]]:
    """Measure execution time and memory usage."""
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.perf_counter()
    
    yield
    
    end_time = time.perf_counter()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return end_time - start_time, end_memory - start_memory

@pytest.fixture
def performance_config() -> ThrottlingConfig:
    """Configuration for performance testing."""
    return ThrottlingConfig(
        cooldown_minutes=1,
        max_alerts_per_hour=1000,
        min_change_threshold=1.0,
        reset_after_hours=24
    )

@pytest.fixture
def temp_storage(tmp_path: Path) -> Path:
    """Temporary storage path."""
    return tmp_path / "perf_test_history.json"

def generate_test_alerts(count: int) -> List[Dict[str, Any]]:
    """Generate test alerts with some duplicates and variations."""
    alerts = []
    metrics = ["cpu", "memory", "latency", "errors", "throughput"]
    severities = ["info", "warning", "critical"]
    
    for i in range(count):
        # Create some patterns to test deduplication
        if i % 3 == 0:  # Duplicate every third alert
            if alerts:
                alerts.append(alerts[-1].copy())
                continue
        
        alert = {
            "title": f"Alert {i // 10}",  # Create some duplicates
            "severity": random.choice(severities),
            "metric": random.choice(metrics),
            "value": random.uniform(0, 1000)
        }
        alerts.append(alert)
    
    return alerts

def test_throttling_performance(performance_config: ThrottlingConfig, temp_storage: Path) -> None:
    """Test throttling performance with different load sizes."""
    throttler = AlertThrottler(performance_config, temp_storage)
    
    load_sizes = [100, 1000, 10000]
    results: Dict[int, Dict[str, float]] = {}
    
    for size in load_sizes:
        alerts = generate_test_alerts(size)
        timings: List[float] = []
        memory_usage: List[float] = []
        
        # Warm up
        throttler = AlertThrottler(performance_config, temp_storage)
        
        # Measure processing time
        with measure_time_and_memory() as measurements:
            for alert in alerts:
                if not throttler.should_throttle(**alert):
                    throttler.record_alert(**alert)
        
        duration, memory = measurements
        
        results[size] = {
            "alerts_per_second": size / duration,
            "average_memory_mb": memory,
            "total_duration": duration
        }
    
    # Save performance results
    perf_file = temp_storage.parent / "throttling_performance.json"
    with open(perf_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Verify performance meets minimal requirements
    for size, metrics in results.items():
        # Should process at least 100 alerts per second for any size
        assert metrics["alerts_per_second"] >= 100, f"Poor performance for size {size}"
        # Memory usage should scale sub-linearly
        assert metrics["average_memory_mb"] <= size * 0.001, f"High memory usage for size {size}"

def test_concurrent_performance(performance_config: ThrottlingConfig, temp_storage: Path) -> None:
    """Test performance under concurrent load."""
    throttler = AlertThrottler(performance_config, temp_storage)
    
    num_threads = 4
    alerts_per_thread = 1000
    alerts = generate_test_alerts(alerts_per_thread * num_threads)
    
    results: List[Tuple[float, float]] = []
    
    def process_alerts(thread_alerts: List[Dict[str, Any]]) -> Tuple[float, float]:
        with measure_time_and_memory() as measurements:
            for alert in thread_alerts:
                if not throttler.should_throttle(**alert):
                    throttler.record_alert(**alert)
        return measurements
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks
        futures = []
        for i in range(num_threads):
            start_idx = i * alerts_per_thread
            end_idx = start_idx + alerts_per_thread
            futures.append(executor.submit(process_alerts, alerts[start_idx:end_idx]))
        
        # Collect results
        for future in as_completed(futures):
            results.append(future.result())
    
    # Analyze results
    durations = [r[0] for r in results]
    memory_deltas = [r[1] for r in results]
    
    metrics = {
        "avg_duration": statistics.mean(durations),
        "max_duration": max(durations),
        "total_memory_mb": sum(memory_deltas),
        "alerts_per_second": (alerts_per_thread * num_threads) / max(durations)
    }
    
    # Save concurrent performance results
    perf_file = temp_storage.parent / "concurrent_performance.json"
    with open(perf_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Verify concurrent performance
    assert metrics["alerts_per_second"] >= 500, "Poor concurrent performance"
    assert metrics["total_memory_mb"] <= 100, "High memory usage under concurrency"

def test_storage_performance(performance_config: ThrottlingConfig, temp_storage: Path) -> None:
    """Test storage read/write performance."""
    throttler = AlertThrottler(performance_config, temp_storage)
    alerts = generate_test_alerts(1000)
    
    # Measure write performance
    write_times: List[float] = []
    for alert in alerts[:100]:  # Sample first 100 alerts
        start = time.perf_counter()
        throttler.record_alert(**alert)
        write_times.append(time.perf_counter() - start)
    
    # Measure read performance
    read_times: List[float] = []
    for _ in range(100):
        start = time.perf_counter()
        throttler._load_history()
        read_times.append(time.perf_counter() - start)
    
    storage_metrics = {
        "avg_write_ms": statistics.mean(write_times) * 1000,
        "max_write_ms": max(write_times) * 1000,
        "avg_read_ms": statistics.mean(read_times) * 1000,
        "max_read_ms": max(read_times) * 1000,
        "file_size_kb": os.path.getsize(temp_storage) / 1024
    }
    
    # Save storage performance results
    perf_file = temp_storage.parent / "storage_performance.json"
    with open(perf_file, 'w') as f:
        json.dump(storage_metrics, f, indent=2)
    
    # Verify storage performance
    assert storage_metrics["avg_write_ms"] <= 1, "Slow write performance"
    assert storage_metrics["avg_read_ms"] <= 1, "Slow read performance"
    assert storage_metrics["file_size_kb"] <= 1000, "Large storage file size"

def test_cleanup_performance(performance_config: ThrottlingConfig, temp_storage: Path) -> None:
    """Test performance of cleanup operations."""
    throttler = AlertThrottler(performance_config, temp_storage)
    alerts = generate_test_alerts(10000)
    
    # Fill history
    for alert in alerts:
        if not throttler.should_throttle(**alert):
            throttler.record_alert(**alert)
    
    # Measure cleanup performance
    cleanup_times: List[float] = []
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024
    
    for _ in range(10):
        start = time.perf_counter()
        throttler._cleanup_old_records()
        cleanup_times.append(time.perf_counter() - start)
    
    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
    
    cleanup_metrics = {
        "avg_cleanup_ms": statistics.mean(cleanup_times) * 1000,
        "max_cleanup_ms": max(cleanup_times) * 1000,
        "memory_impact_mb": memory_after - memory_before
    }
    
    # Save cleanup performance results
    perf_file = temp_storage.parent / "cleanup_performance.json"
    with open(perf_file, 'w') as f:
        json.dump(cleanup_metrics, f, indent=2)
    
    # Verify cleanup performance
    assert cleanup_metrics["avg_cleanup_ms"] <= 100, "Slow cleanup performance"
    assert cleanup_metrics["memory_impact_mb"] <= 10, "High cleanup memory impact"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
