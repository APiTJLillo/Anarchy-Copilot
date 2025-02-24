"""Tests for performance threshold checking."""
import pytest
from pathlib import Path
import json
import yaml
import tempfile
from datetime import datetime

from scripts.check_perf_thresholds import ThresholdChecker

@pytest.fixture
def test_results():
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
        "regressions": {},
        "timestamp": datetime.now().isoformat()
    }

@pytest.fixture
def test_thresholds():
    """Create sample thresholds configuration."""
    return {
        "min_throughput": 1000.0,
        "max_latency_p95": 0.01,
        "max_memory_mb": 1024.0,
        "max_cpu_percent": 80.0,
        "throughput_regression_threshold": 0.15
    }

@pytest.fixture
def results_dir(tmp_path, test_results):
    """Create test results directory with sample data."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Save test results
    with open(results_dir / "results_20250223.json", "w") as f:
        json.dump(test_results, f)
    
    # Create metrics summary
    with open(results_dir / "metrics_summary.txt", "w") as f:
        f.write(
            "System Performance Summary\n"
            "=========================\n"
            "Peak CPU usage: 75.5%\n"
            "Peak Memory: 512MB\n"
        )
    
    return results_dir

@pytest.fixture
def config_dir(tmp_path, test_thresholds):
    """Create test configuration directory."""
    config_dir = tmp_path / "config_examples"
    config_dir.mkdir(parents=True)
    
    # Create test profiles
    profiles = {
        "ci": {**test_thresholds, "min_throughput": 500.0},
        "development": test_thresholds,
        "production": {**test_thresholds, "min_throughput": 2000.0},
        "stress": {**test_thresholds, "min_throughput": 5000.0}
    }
    
    for profile, thresholds in profiles.items():
        with open(config_dir / f"{profile}.yml", "w") as f:
            yaml.dump(thresholds, f)
    
    return config_dir

def test_threshold_loading(config_dir, monkeypatch):
    """Test loading threshold configurations."""
    monkeypatch.setattr(
        "scripts.check_perf_thresholds.Path.__truediv__",
        lambda self, other: config_dir / other.split("/")[-1]
    )
    
    checker = ThresholdChecker("development")
    assert checker.thresholds["throughput"] == 1000.0
    assert checker.thresholds["latency_p95"] == 0.01
    assert checker.thresholds["memory_mb"] == 1024.0

def test_successful_check(results_dir):
    """Test checking results against thresholds - success case."""
    checker = ThresholdChecker("development")
    violations = checker.check_results(results_dir)
    
    assert not violations, "No violations should be found"
    assert (results_dir / "threshold_report.txt").exists()

def test_threshold_violations(results_dir, test_results):
    """Test detection of threshold violations."""
    # Modify results to trigger violations
    test_results["results"].update({
        "throughput": 500.0,  # Below minimum
        "latency_p95": 0.02,  # Above maximum
        "memory_usage": 2 * 1024 * 1024 * 1024  # 2GB, above maximum
    })
    
    with open(results_dir / "results_20250223.json", "w") as f:
        json.dump(test_results, f)
    
    checker = ThresholdChecker("development")
    violations = checker.check_results(results_dir)
    
    assert len(violations) == 3
    assert any("throughput" in v.lower() for v in violations)
    assert any("latency" in v.lower() for v in violations)
    assert any("memory" in v.lower() for v in violations)

def test_regression_detection(results_dir, test_results):
    """Test detection of performance regressions."""
    # Add regression data
    test_results["regressions"] = {
        "throughput": {
            "baseline": 2000.0,
            "current": 1500.0,
            "degradation": 25.0  # 25% degradation
        }
    }
    
    with open(results_dir / "results_20250223.json", "w") as f:
        json.dump(test_results, f)
    
    checker = ThresholdChecker("development")
    violations = checker.check_results(results_dir)
    
    assert len(violations) == 1
    assert "regression" in violations[0].lower()
    assert "25.0%" in violations[0]

def test_invalid_results_handling(tmp_path):
    """Test handling of invalid or missing results."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    checker = ThresholdChecker("development")
    violations = checker.check_results(empty_dir)
    
    assert len(violations) == 1
    assert "error" in violations[0].lower()

def test_report_generation(results_dir):
    """Test generation of threshold report."""
    checker = ThresholdChecker("development")
    checker.check_results(results_dir)
    
    report_path = results_dir / "threshold_report.txt"
    assert report_path.exists()
    
    with open(report_path) as f:
        report = f.read()
        assert "Thresholds:" in report
        assert "Results:" in report
        assert "development" in report

def test_different_profiles(config_dir, results_dir, monkeypatch):
    """Test checking against different profiles."""
    monkeypatch.setattr(
        "scripts.check_perf_thresholds.Path.__truediv__",
        lambda self, other: config_dir / other.split("/")[-1]
    )
    
    # Test CI profile (more lenient)
    ci_checker = ThresholdChecker("ci")
    ci_violations = ci_checker.check_results(results_dir)
    assert not ci_violations
    
    # Test production profile (stricter)
    prod_checker = ThresholdChecker("production")
    prod_violations = prod_checker.check_results(results_dir)
    assert len(prod_violations) == 1
    assert "throughput" in prod_violations[0].lower()

def test_system_metrics_check(results_dir):
    """Test checking of system metrics."""
    # Update metrics summary with high CPU usage
    with open(results_dir / "metrics_summary.txt", "w") as f:
        f.write(
            "System Performance Summary\n"
            "=========================\n"
            "Peak CPU usage: 95.5%\n"  # Above threshold
            "Peak Memory: 512MB\n"
        )
    
    checker = ThresholdChecker("development")
    violations = checker.check_results(results_dir)
    
    assert len(violations) == 1
    assert "cpu usage" in violations[0].lower()
    assert "95.5%" in violations[0]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
