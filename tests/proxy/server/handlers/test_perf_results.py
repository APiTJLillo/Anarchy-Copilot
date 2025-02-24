"""Tests for performance results processing."""
import pytest
from pathlib import Path
import json
import os
from datetime import datetime
import tempfile

from scripts.process_perf_results import ResultsProcessor

@pytest.fixture
def sample_test_results():
    """Create sample test results."""
    return {
        "results": {
            "throughput": 1500.0,
            "latency_p95": 0.005,
            "memory_usage": 512 * 1024 * 1024,  # 512MB
            "execution_times": {
                "middleware1": [0.001, 0.002, 0.003],
                "middleware2": [0.002, 0.003, 0.004]
            }
        },
        "regressions": {
            "throughput": {
                "baseline": 2000.0,
                "current": 1500.0,
                "degradation": 25.0
            }
        },
        "timestamp": datetime.now().isoformat()
    }

@pytest.fixture
def sample_system_metrics():
    """Create sample system metrics."""
    return """System Performance Summary
==========================
Peak CPU usage: 75.5%
Peak Memory: 512MB
IO Wait Average: 2.5%
"""

@pytest.fixture
def results_dir(tmp_path, sample_test_results, sample_system_metrics):
    """Create test results directory with sample data."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Save multiple test results
    for i in range(3):
        with open(results_dir / f"results_{i}.json", "w") as f:
            json.dump(sample_test_results, f)
    
    # Save system metrics
    with open(results_dir / "metrics_summary.txt", "w") as f:
        f.write(sample_system_metrics)
    
    return results_dir

def test_load_test_results(results_dir):
    """Test loading and aggregating test results."""
    processor = ResultsProcessor(results_dir)
    results = processor._load_test_results()
    
    assert "throughput" in results
    assert "latency" in results
    assert "memory" in results
    assert "execution_times" in results
    
    # Check statistics
    assert results["throughput"]["mean"] == 1500.0
    assert results["latency"]["mean"] == 0.005
    assert results["memory"]["peak"] == 512 * 1024 * 1024

def test_load_system_metrics(results_dir):
    """Test loading system metrics."""
    processor = ResultsProcessor(results_dir)
    metrics = processor._load_system_metrics()
    
    assert metrics["cpu_peak"] == 75.5
    assert metrics["memory_peak"] == 512.0
    assert metrics["io_wait_avg"] == 2.5

def test_regression_analysis(results_dir):
    """Test analyzing performance regressions."""
    processor = ResultsProcessor(results_dir)
    regressions = processor._analyze_regressions()
    
    assert regressions["total_count"] == 1
    assert "critical" in regressions["severity"]
    assert "throughput" in regressions["details"]
    assert regressions["details"]["throughput"]["degradation"] == 25.0

def test_trend_analysis(results_dir, sample_test_results):
    """Test analyzing performance trends."""
    # Create results with varying performance
    for i in range(3):
        results = sample_test_results.copy()
        results["results"]["throughput"] = 1000.0 + (i * 250)  # Improving trend
        with open(results_dir / f"trend_results_{i}.json", "w") as f:
            json.dump(results, f)
    
    processor = ResultsProcessor(results_dir)
    trends = processor._analyze_trends()
    
    assert "throughput" in trends
    assert trends["throughput"]["direction"] == "improving"
    assert trends["throughput"]["change_pct"] > 0

def test_github_report_generation(results_dir):
    """Test generating GitHub-flavored markdown report."""
    processor = ResultsProcessor(results_dir, output_format="github")
    summary = processor.process_results()
    report = processor.generate_report(summary)
    
    assert "## Performance Test Results" in report
    assert "### Summary" in report
    assert "| Metric | Value |" in report
    assert "### Performance Trends" in report

def test_text_report_generation(results_dir):
    """Test generating plain text report."""
    processor = ResultsProcessor(results_dir, output_format="text")
    summary = processor.process_results()
    report = processor.generate_report(summary)
    
    assert "Performance Test Results" in report
    assert "Test Results:" in report
    assert "Throughput:" in report
    assert "Performance Trends:" in report

def test_missing_results_handling(tmp_path):
    """Test handling of missing results."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    processor = ResultsProcessor(empty_dir)
    summary = processor.process_results()
    
    assert not summary["tests"]
    assert not summary["system"]
    assert not summary["trends"]

def test_invalid_results_handling(results_dir):
    """Test handling of invalid results data."""
    # Create invalid JSON file
    with open(results_dir / "invalid_results.json", "w") as f:
        f.write("invalid json content")
    
    processor = ResultsProcessor(results_dir)
    summary = processor.process_results()
    
    # Should still process valid files
    assert summary["tests"]
    assert summary["system"]

def test_trend_stability(results_dir, sample_test_results):
    """Test trend analysis stability with fluctuating data."""
    # Create results with fluctuating values
    values = [1000, 1200, 800, 1100, 900]  # Fluctuating throughput
    for i, value in enumerate(values):
        results = sample_test_results.copy()
        results["results"]["throughput"] = value
        with open(results_dir / f"fluctuating_{i}.json", "w") as f:
            json.dump(results, f)
    
    processor = ResultsProcessor(results_dir)
    trends = processor._analyze_trends()
    
    # Should identify overall trend despite fluctuations
    assert "throughput" in trends
    assert abs(trends["throughput"]["change_pct"]) < 20  # Not a strong trend

def test_large_dataset_handling(results_dir, sample_test_results):
    """Test handling of large datasets."""
    # Create many result files
    for i in range(100):
        results = sample_test_results.copy()
        results["results"]["throughput"] = 1000 + (i % 10) * 100
        with open(results_dir / f"large_dataset_{i}.json", "w") as f:
            json.dump(results, f)
    
    processor = ResultsProcessor(results_dir)
    summary = processor.process_results()
    
    assert "tests" in summary
    assert "trends" in summary
    assert len(summary["trends"]) > 0

def test_results_caching(results_dir):
    """Test results caching functionality."""
    processor = ResultsProcessor(results_dir)
    
    # First call should process results
    initial_summary = processor.process_results()
    
    # Second call should use cache
    cached_summary = processor.process_results()
    
    assert processor.results_cache == initial_summary
    assert cached_summary == initial_summary

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
