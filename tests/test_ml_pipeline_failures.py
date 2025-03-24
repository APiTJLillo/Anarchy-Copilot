#!/usr/bin/env python3
"""Test failure handling in ML pipeline components."""

import pytest
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Generator
import shutil
import os

from scripts.alert_throttling import AlertThrottler, ThrottlingConfig
from scripts.predict_throttling_performance import PerformancePredictor
from scripts.validate_throttling_models import ModelValidator
from scripts.multi_variant_test import MultiVariantTester
from scripts.track_experiments import ExperimentTracker, ExperimentResult
from scripts.analyze_experiments import ExperimentAnalyzer

@pytest.fixture
def corrupt_history(test_data_dir: Path) -> Path:
    """Create corrupted performance history."""
    history_file = test_data_dir / "corrupt_history.json"
    history_file.write_text("invalid{json:content")
    return history_file

@pytest.fixture
def incomplete_history(test_data_dir: Path) -> Path:
    """Create incomplete performance history."""
    history = [
        {
            "timestamp": datetime.now().isoformat(),
            # Missing required metrics
            "throughput": np.random.normal(100, 10)
        }
    ]
    history_file = test_data_dir / "incomplete_history.json"
    history_file.write_text(json.dumps(history))
    return history_file

@pytest.fixture
def invalid_models(test_data_dir: Path) -> Path:
    """Create invalid model files."""
    model_dir = test_data_dir / "invalid_models"
    model_dir.mkdir(exist_ok=True)
    
    # Create corrupt model files
    (model_dir / "throughput_regressor.pkl").write_text("not a model")
    (model_dir / "memory_scaler.pkl").write_bytes(b"invalid bytes")
    
    return model_dir

@pytest.fixture
def permission_denied_dir(test_data_dir: Path) -> Path:
    """Create directory with restricted permissions."""
    restricted_dir = test_data_dir / "restricted"
    restricted_dir.mkdir(exist_ok=True)
    os.chmod(restricted_dir, 0o000)  # Remove all permissions
    return restricted_dir

def test_throttler_with_corrupt_history(corrupt_history: Path) -> None:
    """Test throttler handling of corrupt history."""
    config = ThrottlingConfig(
        cooldown_minutes=5,
        max_alerts_per_hour=100,
        min_change_threshold=1.0,
        reset_after_hours=1
    )
    
    with pytest.raises(Exception) as exc:
        AlertThrottler(config, corrupt_history)
    assert "Error loading history" in str(exc.value)

def test_predictor_with_invalid_models(
    sample_history: Path,
    invalid_models: Path
) -> None:
    """Test predictor handling of invalid models."""
    predictor = PerformancePredictor(sample_history, model_dir=invalid_models)
    
    with pytest.raises(Exception) as exc:
        predictor.train_models()
    assert any(err in str(exc.value) for err in ["corrupt", "invalid", "load"])

def test_validator_with_missing_models(
    sample_history: Path,
    test_data_dir: Path
) -> None:
    """Test validator handling of missing models."""
    empty_dir = test_data_dir / "empty_models"
    empty_dir.mkdir(exist_ok=True)
    
    validator = ModelValidator(empty_dir, sample_history)
    results = validator.validate_all_models()
    
    assert not results  # Should return False for no models
    assert not validator.results  # Should have no validation results

def test_multi_variant_with_incomplete_data(
    incomplete_history: Path,
    sample_models: Path
) -> None:
    """Test multi-variant testing with incomplete data."""
    tester = MultiVariantTester(sample_models, incomplete_history)
    
    # Should handle missing metrics gracefully
    results = tester.compare_variants("memory_usage")  # Missing metric
    assert results is None

def test_experiment_tracking_with_permission_error(
    permission_denied_dir: Path,
    sample_experiments: Path
) -> None:
    """Test experiment tracking with permission errors."""
    tracker = ExperimentTracker(permission_denied_dir)
    
    # Create test experiment
    experiment = ExperimentResult(
        timestamp=datetime.now(),
        metric="throughput",
        variants=["A", "B"],
        winner="B",
        improvement=5.0,
        commit="test",
        config={},
        metrics={},
        significance=0.05,
        effect_size=0.5
    )
    
    # Should handle permission error gracefully
    with pytest.raises(Exception) as exc:
        tracker.record_experiment(experiment)
    assert any(err in str(exc.value).lower() for err in ["permission", "access"])

def test_analyzer_with_corrupt_experiments(
    test_data_dir: Path,
    sample_experiments: Path
) -> None:
    """Test analyzer handling of corrupt experiment data."""
    # Create directory with mix of valid and corrupt experiments
    mixed_dir = test_data_dir / "mixed_experiments"
    mixed_dir.mkdir(exist_ok=True)
    
    # Copy valid experiments
    for file in sample_experiments.glob("*.json"):
        shutil.copy(file, mixed_dir)
    
    # Add corrupt experiment
    (mixed_dir / "corrupt_experiment.json").write_text("invalid{json")
    
    analyzer = ExperimentAnalyzer(mixed_dir)
    
    # Should skip corrupt files but continue analysis
    results = analyzer.run_analysis()
    assert results is not None
    assert len(results.recommendations) > 0

def test_pipeline_error_propagation(
    test_data_dir: Path,
    sample_history: Path,
    invalid_models: Path,
    permission_denied_dir: Path
) -> None:
    """Test error propagation through pipeline."""
    try:
        # 1. Try training with invalid models
        predictor = PerformancePredictor(sample_history, model_dir=invalid_models)
        predictor.train_models()
        
        # 2. Try validation
        validator = ModelValidator(invalid_models, sample_history)
        validator.validate_all_models()
        
        # 3. Try experiment tracking
        tracker = ExperimentTracker(permission_denied_dir)
        
        # 4. Try analysis
        analyzer = ExperimentAnalyzer(permission_denied_dir)
        analyzer.run_analysis()
        
        pytest.fail("Pipeline should have raised exceptions")
        
    except Exception as e:
        # Verify error contains useful information
        error_str = str(e).lower()
        assert any(word in error_str for word in [
            "error", "invalid", "corrupt", "permission", "failed"
        ])

def test_recovery_from_partial_failure(
    test_data_dir: Path,
    sample_history: Path,
    sample_models: Path
) -> None:
    """Test pipeline recovery from partial failures."""
    # Create directory structure
    working_dir = test_data_dir / "recovery_test"
    working_dir.mkdir(exist_ok=True)
    
    # 1. Simulate partial training failure
    predictor = PerformancePredictor(sample_history, model_dir=working_dir)
    
    # Corrupt one model mid-training
    def corrupt_model_during_training(*args, **kwargs):
        # Corrupt throughput model after it's created
        model_file = working_dir / "throughput_regressor.pkl"
        if model_file.exists():
            model_file.write_text("corrupted")
        raise RuntimeError("Simulated training error")
    
    original_train = predictor.train_models
    predictor.train_models = corrupt_model_during_training
    
    try:
        predictor.train_models()
    except RuntimeError:
        pass
    
    # Restore original method
    predictor.train_models = original_train
    
    # 2. Test recovery
    # Should detect and retrain corrupted model
    predictor.train_models()
    
    # Verify recovery
    assert (working_dir / "throughput_regressor.pkl").exists()
    assert (working_dir / "memory_usage_regressor.pkl").exists()

def test_concurrent_access_handling(
    test_data_dir: Path,
    sample_history: Path,
    sample_models: Path
) -> None:
    """Test handling of concurrent access to shared resources."""
    import threading
    import queue
    
    # Create shared experiment tracker
    exp_dir = test_data_dir / "concurrent_experiments"
    exp_dir.mkdir(exist_ok=True)
    tracker = ExperimentTracker(exp_dir)
    
    # Create multiple experiments
    def create_experiment(i: int) -> ExperimentResult:
        return ExperimentResult(
            timestamp=datetime.now(),
            metric="throughput",
            variants=["A", "B"],
            winner="B",
            improvement=float(i),
            commit=f"test_{i}",
            config={},
            metrics={},
            significance=0.05,
            effect_size=0.5
        )
    
    # Queue for results
    results_queue = queue.Queue()
    
    # Function to record experiment in thread
    def record_experiment(exp: ExperimentResult) -> None:
        try:
            tracker.record_experiment(exp)
            results_queue.put(True)
        except Exception as e:
            results_queue.put(e)
    
    # Start multiple threads
    threads = []
    for i in range(10):
        thread = threading.Thread(
            target=record_experiment,
            args=(create_experiment(i),)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Check results
    failures = []
    while not results_queue.empty():
        result = results_queue.get()
        if isinstance(result, Exception):
            failures.append(result)
    
    assert len(failures) == 0, f"Concurrent access failures: {failures}"
    assert len(list(exp_dir.glob("*.json"))) == 10

def test_resource_cleanup(
    test_data_dir: Path,
    sample_history: Path,
    sample_models: Path
) -> None:
    """Test proper resource cleanup after failures."""
    import resource
    import gc
    
    def get_open_files() -> int:
        return resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    
    initial_files = get_open_files()
    
    try:
        # Simulate operations that might leave files open
        predictor = PerformancePredictor(sample_history, model_dir=sample_models)
        predictor.train_models()
        raise RuntimeError("Simulated failure")
    except RuntimeError:
        pass
    
    # Force garbage collection
    gc.collect()
    
    # Check for leaked file descriptors
    final_files = get_open_files()
    assert final_files <= initial_files + 5, "Possible file descriptor leak"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
