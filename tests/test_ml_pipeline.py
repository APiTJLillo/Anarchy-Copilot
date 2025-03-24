#!/usr/bin/env python3
"""Test suite for ML pipeline components."""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Generator

from scripts.alert_throttling import AlertThrottler, ThrottlingConfig
from scripts.predict_throttling_performance import PerformancePredictor
from scripts.validate_throttling_models import ModelValidator
from scripts.multi_variant_test import MultiVariantTester
from scripts.track_experiments import ExperimentTracker
from scripts.analyze_experiments import ExperimentAnalyzer

@pytest.fixture
def test_data_dir() -> Generator[Path, None, None]:
    """Provide temporary test data directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_history(test_data_dir: Path) -> Path:
    """Create sample performance history."""
    history = []
    start_date = datetime.now() - timedelta(days=30)
    
    for day in range(30):
        timestamp = start_date + timedelta(days=day)
        history.append({
            "timestamp": timestamp.isoformat(),
            "throughput": np.random.normal(100, 10),
            "memory_usage": np.random.normal(50, 5),
            "storage_size": np.random.normal(200, 20),
            "cleanup_time": np.random.normal(10, 1),
            "alerts_per_second": np.random.normal(500, 50)
        })
    
    history_file = test_data_dir / "performance_history.json"
    history_file.write_text(json.dumps(history))
    return history_file

@pytest.fixture
def sample_models(test_data_dir: Path) -> Path:
    """Create sample model files."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    model_dir = test_data_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    metrics = ["throughput", "memory_usage", "storage_size", "cleanup_time", "alerts_per_second"]
    variants = ["current", "previous", "experimental"]
    
    for metric in metrics:
        # Create scaler
        scaler = StandardScaler()
        scaler.fit(np.random.normal(0, 1, (100, 5)).reshape(-1, 5))
        joblib.dump(scaler, model_dir / f"{metric}_scaler.pkl")
        
        # Create model variants
        for variant in variants:
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(
                np.random.normal(0, 1, (100, 5)),
                np.random.normal(0, 1, 100)
            )
            joblib.dump(model, model_dir / f"{metric}_{variant}_regressor.pkl")
    
    return model_dir

@pytest.fixture
def sample_experiments(test_data_dir: Path) -> Path:
    """Create sample experiment results."""
    exp_dir = test_data_dir / "experiments"
    exp_dir.mkdir(exist_ok=True)
    
    metrics = ["throughput", "memory_usage", "storage_size", "cleanup_time", "alerts_per_second"]
    start_date = datetime.now() - timedelta(days=30)
    
    for day in range(30):
        for metric in metrics:
            timestamp = start_date + timedelta(days=day)
            experiment = {
                "timestamp": timestamp.isoformat(),
                "metric": metric,
                "variants": ["current", "previous", "experimental"],
                "winner": "experimental" if np.random.random() > 0.5 else None,
                "improvement": np.random.normal(5, 2),
                "commit": f"commit_{day}_{metric}",
                "config": {
                    "n_estimators": 100,
                    "max_depth": 10
                },
                "metrics": {
                    "current": {"r2": 0.8, "mse": 0.2},
                    "experimental": {"r2": 0.85, "mse": 0.15}
                },
                "significance": np.random.random(),
                "effect_size": np.random.normal(0.5, 0.1)
            }
            
            file_path = exp_dir / f"experiment_{timestamp.strftime('%Y%m%d_%H%M%S')}_{metric}.json"
            file_path.write_text(json.dumps(experiment))
    
    return exp_dir

def test_throttler_initialization(sample_history: Path) -> None:
    """Test throttler initialization."""
    config = ThrottlingConfig(
        cooldown_minutes=5,
        max_alerts_per_hour=100,
        min_change_threshold=1.0,
        reset_after_hours=1
    )
    
    throttler = AlertThrottler(config, sample_history)
    assert throttler.data is not None
    assert len(throttler.alert_history) == 0

def test_predictor_training(sample_history: Path, sample_models: Path) -> None:
    """Test model predictor training."""
    predictor = PerformancePredictor(sample_history, model_dir=sample_models)
    predictor.train_models()
    
    # Verify models were created
    assert (sample_models / "throughput_current_regressor.pkl").exists()
    assert (sample_models / "throughput_scaler.pkl").exists()

def test_model_validation(sample_history: Path, sample_models: Path) -> None:
    """Test model validation."""
    validator = ModelValidator(sample_models, sample_history)
    passed = validator.validate_all_models()
    
    assert isinstance(passed, bool)
    assert len(validator.results) > 0

def test_multi_variant_testing(sample_history: Path, sample_models: Path) -> None:
    """Test multi-variant testing."""
    tester = MultiVariantTester(sample_models, sample_history)
    
    # Test specific metric
    results = tester.compare_variants("throughput")
    assert results is not None
    assert results.variants == ["current", "experimental", "previous"]
    assert results.best_variant in results.variants

def test_experiment_tracking(sample_experiments: Path) -> None:
    """Test experiment tracking."""
    tracker = ExperimentTracker(sample_experiments)
    
    # Verify history loading
    assert len(tracker.history) > 0
    
    # Test new experiment recording
    from scripts.track_experiments import ExperimentResult
    new_result = ExperimentResult(
        timestamp=datetime.now(),
        metric="throughput",
        variants=["A", "B"],
        winner="B",
        improvement=5.0,
        commit="test_commit",
        config={"param": "value"},
        metrics={"A": {"r2": 0.8}, "B": {"r2": 0.85}},
        significance=0.01,
        effect_size=0.5
    )
    
    tracker.record_experiment(new_result)
    assert len(tracker.history["throughput"]) > 0

def test_experiment_analysis(sample_experiments: Path) -> None:
    """Test experiment analysis."""
    analyzer = ExperimentAnalyzer(sample_experiments)
    
    # Run analysis
    results = analyzer.run_analysis()
    assert results is not None
    
    # Check patterns
    patterns = analyzer.analyze_improvement_patterns()
    assert "success_rate_by_metric" in patterns
    assert "avg_improvement_by_metric" in patterns
    
    # Test recommendations
    recommendations = analyzer.generate_recommendations()
    assert len(recommendations) > 0

def test_end_to_end_pipeline(
    test_data_dir: Path,
    sample_history: Path,
    sample_models: Path,
    sample_experiments: Path
) -> None:
    """Test complete ML pipeline."""
    # 1. Train and validate models
    predictor = PerformancePredictor(sample_history, model_dir=sample_models)
    predictor.train_models()
    
    validator = ModelValidator(sample_models, sample_history)
    validation_passed = validator.validate_all_models()
    
    # 2. Run multi-variant tests
    tester = MultiVariantTester(sample_models, sample_history)
    test_results = {}
    for metric in ["throughput", "memory_usage"]:
        results = tester.compare_variants(metric)
        if results:
            test_results[metric] = results
    
    # 3. Track experiments
    tracker = ExperimentTracker(sample_experiments)
    for metric, results in test_results.items():
        if results.best_variant:
            from scripts.track_experiments import ExperimentResult
            tracker.record_experiment(ExperimentResult(
                timestamp=datetime.now(),
                metric=metric,
                variants=results.variants,
                winner=results.best_variant,
                improvement=results.improvements.get(results.best_variant, 0.0),
                commit="pipeline_test",
                config={"test": True},
                metrics=results.metrics,
                significance=results.significances.get(
                    (results.variants[0], results.best_variant),
                    1.0
                ),
                effect_size=results.effects.get(
                    (results.variants[0], results.best_variant),
                    0.0
                )
            ))
    
    # 4. Analyze results
    analyzer = ExperimentAnalyzer(sample_experiments)
    analysis = analyzer.run_analysis()
    
    # Verify pipeline outputs
    assert validation_passed is not None
    assert len(test_results) > 0
    assert len(tracker.history) > 0
    assert analysis is not None
    assert len(analysis.recommendations) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
