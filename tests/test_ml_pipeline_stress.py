#!/usr/bin/env python3
"""Stress testing for ML pipeline components."""

import pytest
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Generator, List, Dict, Any
import multiprocessing as mp
import threading
import time
import resource
import psutil
import os
import signal
from contextlib import contextmanager

from scripts.alert_throttling import AlertThrottler, ThrottlingConfig
from scripts.predict_throttling_performance import PerformancePredictor
from scripts.validate_throttling_models import ModelValidator
from scripts.multi_variant_test import MultiVariantTester
from scripts.track_experiments import ExperimentTracker, ExperimentResult
from scripts.analyze_experiments import ExperimentAnalyzer

@contextmanager
def limit_resources(memory_mb: int = 100, cpu_seconds: int = 30) -> Generator:
    """Temporarily limit process resources."""
    # Set memory limit
    gb_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (gb_bytes, gb_bytes))
    
    # Set CPU time limit
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    
    try:
        yield
    finally:
        # Reset limits
        resource.setrlimit(resource.RLIMIT_AS, (-1, -1))
        resource.setrlimit(resource.RLIMIT_CPU, (-1, -1))

@contextmanager
def measure_resources() -> Generator[Dict[str, Any], None, None]:
    """Measure resource usage."""
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss
    start_cpu = process.cpu_percent()
    
    stats = {
        "start_memory": start_memory,
        "start_cpu": start_cpu,
        "peak_memory": start_memory,
        "max_cpu": start_cpu
    }
    
    def update_stats():
        while True:
            try:
                current_memory = process.memory_info().rss
                current_cpu = process.cpu_percent()
                stats["peak_memory"] = max(stats["peak_memory"], current_memory)
                stats["max_cpu"] = max(stats["max_cpu"], current_cpu)
                time.sleep(0.1)
            except:
                break
    
    monitor = threading.Thread(target=update_stats)
    monitor.daemon = True
    monitor.start()
    
    try:
        yield stats
    finally:
        stats["duration"] = time.time() - start_time
        monitor.join(timeout=1)

def generate_large_history(size: int = 1000000) -> List[Dict[str, Any]]:
    """Generate large performance history."""
    history = []
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(size):
        timestamp = start_date + timedelta(minutes=i)
        history.append({
            "timestamp": timestamp.isoformat(),
            "throughput": np.random.normal(100, 10),
            "memory_usage": np.random.normal(50, 5),
            "storage_size": np.random.normal(200, 20),
            "cleanup_time": np.random.normal(10, 1),
            "alerts_per_second": np.random.normal(500, 50)
        })
    
    return history

def test_large_data_handling(test_data_dir: Path) -> None:
    """Test handling of large datasets."""
    # Generate large history
    history = generate_large_history()
    history_file = test_data_dir / "large_history.json"
    history_file.write_text(json.dumps(history))
    
    with measure_resources() as stats:
        # Initialize components
        config = ThrottlingConfig(
            cooldown_minutes=5,
            max_alerts_per_hour=1000,
            min_change_threshold=1.0,
            reset_after_hours=24
        )
        throttler = AlertThrottler(config, history_file)
        predictor = PerformancePredictor(history_file)
        
        # Process large dataset
        for i in range(1000):
            alert = {
                "title": f"Test Alert {i}",
                "severity": "critical",
                "metric": "latency",
                "value": np.random.normal(100, 10)
            }
            if not throttler.should_throttle(**alert):
                throttler.record_alert(**alert)
    
    # Verify resource usage
    assert stats["peak_memory"] < 1024 * 1024 * 1024  # 1GB limit
    assert stats["duration"] < 60  # Should complete within 60 seconds

def test_concurrent_model_training(test_data_dir: Path) -> None:
    """Test concurrent model training under load."""
    history_file = test_data_dir / "concurrent_history.json"
    history_file.write_text(json.dumps(generate_large_history(10000)))
    
    def train_models() -> None:
        predictor = PerformancePredictor(history_file)
        predictor.train_models()
    
    # Start multiple training processes
    processes = []
    for _ in range(4):
        p = mp.Process(target=train_models)
        p.start()
        processes.append(p)
    
    # Monitor resources
    with measure_resources() as stats:
        for p in processes:
            p.join()
    
    # Verify all processes completed
    assert all(not p.is_alive() for p in processes)
    assert stats["max_cpu"] <= 100 * mp.cpu_count()  # Should not exceed available CPUs

def test_memory_constrained_operation(test_data_dir: Path) -> None:
    """Test operation under memory constraints."""
    history_file = test_data_dir / "memory_test_history.json"
    history_file.write_text(json.dumps(generate_large_history(50000)))
    
    with limit_resources(memory_mb=100):  # Limit to 100MB
        try:
            analyzer = ExperimentAnalyzer(test_data_dir)
            analyzer.run_analysis()
        except MemoryError:
            pytest.fail("Failed to handle memory constraints")

def test_continuous_operation(test_data_dir: Path) -> None:
    """Test continuous operation over extended period."""
    history_file = test_data_dir / "continuous_history.json"
    history_file.write_text(json.dumps(generate_large_history(1000)))
    
    start_time = time.time()
    end_time = start_time + 300  # Run for 5 minutes
    
    with measure_resources() as stats:
        while time.time() < end_time:
            try:
                # Simulate continuous pipeline operation
                predictor = PerformancePredictor(history_file)
                predictor.train_models()
                
                validator = ModelValidator(test_data_dir, history_file)
                validator.validate_all_models()
                
                tester = MultiVariantTester(test_data_dir, history_file)
                tester.compare_variants("throughput")
                
                # Short sleep to prevent CPU overload
                time.sleep(0.1)
                
            except Exception as e:
                pytest.fail(f"Failed during continuous operation: {e}")
    
    # Verify stable operation
    assert stats["duration"] >= 300  # Ran for full duration
    assert stats["max_cpu"] <= 90  # Didn't max out CPU

def test_rapid_model_updates(test_data_dir: Path) -> None:
    """Test handling of rapid model updates."""
    history_file = test_data_dir / "rapid_update_history.json"
    history_file.write_text(json.dumps(generate_large_history(1000)))
    
    predictor = PerformancePredictor(history_file)
    updates = []
    
    # Perform rapid model updates
    with measure_resources() as stats:
        for i in range(100):
            try:
                predictor.train_models()
                updates.append(time.time())
                
                # Ensure minimum 10ms between updates
                if len(updates) > 1:
                    assert updates[-1] - updates[-2] >= 0.01
                
            except Exception as e:
                pytest.fail(f"Failed during rapid updates: {e}")
    
    # Verify update rate
    update_rate = len(updates) / stats["duration"]
    assert update_rate <= 100  # Max 100 updates per second

def test_interrupt_handling(test_data_dir: Path) -> None:
    """Test handling of interrupts during processing."""
    history_file = test_data_dir / "interrupt_history.json"
    history_file.write_text(json.dumps(generate_large_history(10000)))
    
    def interrupt_handler(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt()
    
    # Set up interrupt handler
    signal.signal(signal.SIGALRM, interrupt_handler)
    signal.alarm(2)  # Interrupt after 2 seconds
    
    try:
        predictor = PerformancePredictor(history_file)
        predictor.train_models()
    except KeyboardInterrupt:
        # Verify cleanup
        assert not any(f.endswith('.tmp') for f in os.listdir(test_data_dir))
    finally:
        signal.alarm(0)

def test_recovery_speed(test_data_dir: Path) -> None:
    """Test speed of recovery after failures."""
    history_file = test_data_dir / "recovery_history.json"
    history_file.write_text(json.dumps(generate_large_history(5000)))
    
    predictor = PerformancePredictor(history_file)
    
    # Measure normal operation time
    with measure_resources() as normal_stats:
        predictor.train_models()
    normal_duration = normal_stats["duration"]
    
    # Force failure and measure recovery
    try:
        raise RuntimeError("Simulated failure")
    except RuntimeError:
        with measure_resources() as recovery_stats:
            predictor.train_models()
    
    recovery_duration = recovery_stats["duration"]
    
    # Recovery should not be significantly slower
    assert recovery_duration <= normal_duration * 1.5

def test_resource_leaks(test_data_dir: Path) -> None:
    """Test for resource leaks during repeated operations."""
    history_file = test_data_dir / "leak_test_history.json"
    history_file.write_text(json.dumps(generate_large_history(1000)))
    
    initial_resources = psutil.Process().memory_info().rss
    file_descriptors_start = len(psutil.Process().open_files())
    
    for _ in range(100):
        predictor = PerformancePredictor(history_file)
        predictor.train_models()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    final_resources = psutil.Process().memory_info().rss
    file_descriptors_end = len(psutil.Process().open_files())
    
    # Check for leaks
    memory_growth = final_resources - initial_resources
    fd_growth = file_descriptors_end - file_descriptors_start
    
    assert memory_growth < 10 * 1024 * 1024  # Less than 10MB growth
    assert fd_growth <= 5  # Minimal file descriptor growth

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
