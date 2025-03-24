"""Tests for ML components of the proxy analysis module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from datetime import datetime

from proxy.analysis.ml.models import (
    ThrottlingModel,
    PerformancePredictor,
    ExperimentTracker,
    ExperimentConfig
)
from proxy.analysis.ml.training import (
    train_model,
    validate_model,
    update_model,
    cross_validate_model
)
from proxy.analysis.ml.evaluation import (
    ModelEvaluator,
    analyze_results
)

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.normal(0, 1, (1000, 3))
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, 1000)
    return X, y

@pytest.fixture
def trained_model(sample_data):
    """Create a trained model for testing."""
    X, y = sample_data
    model = ThrottlingModel()
    model.fit(X, y)
    return model

@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

class TestThrottlingModel:
    """Test ThrottlingModel functionality."""

    def test_model_initialization(self):
        """Test model initialization."""
        model = ThrottlingModel()
        assert model.window_size == 100
        assert model.threshold == 0.8
        assert "request_rate" in model.features
    
    def test_model_fit_predict(self, sample_data):
        """Test model fitting and prediction."""
        X, y = sample_data
        model = ThrottlingModel()
        
        # Test fit
        fitted_model = model.fit(X, y)
        assert fitted_model is model  # Should return self
        
        # Test predict
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)
    
    def test_model_save_load(self, trained_model, temp_output_dir):
        """Test model serialization."""
        model_path = temp_output_dir / "model.joblib"
        
        # Save model
        trained_model.save(model_path)
        assert model_path.exists()
        
        # Load model
        loaded_model = ThrottlingModel.load(model_path)
        assert isinstance(loaded_model, ThrottlingModel)
        assert loaded_model.window_size == trained_model.window_size

class TestPerformancePredictor:
    """Test PerformancePredictor functionality."""

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        metrics = ["latency", "throughput"]
        predictor = PerformancePredictor(metrics)
        
        assert predictor.metrics == metrics
        assert all(m in predictor.history for m in metrics)
    
    def test_predictor_update(self):
        """Test metric updates."""
        predictor = PerformancePredictor(["latency"])
        
        # Update with new values
        predictor.update({"latency": 100.0})
        assert len(predictor.history["latency"]) == 1
        assert predictor.history["latency"][0] == 100.0
        
        # Test history window limit
        for i in range(2000):
            predictor.update({"latency": float(i)})
        assert len(predictor.history["latency"]) <= predictor.history_window

class TestExperimentTracker:
    """Test ExperimentTracker functionality."""

    def test_experiment_lifecycle(self, temp_output_dir):
        """Test full experiment lifecycle."""
        tracker = ExperimentTracker(str(temp_output_dir))
        
        # Start experiment
        exp = tracker.start_experiment(
            "test_exp",
            {"param1": 1.0},
            {"description": "Test experiment"}
        )
        assert isinstance(exp, ExperimentConfig)
        assert exp.name == "test_exp"
        
        # Record metrics
        tracker.record_metric("test_exp", "accuracy", 0.95)
        assert "accuracy" in tracker.active_experiments["test_exp"].metrics
        
        # End experiment
        completed_exp = tracker.end_experiment("test_exp")
        assert completed_exp.end_time is not None
        assert "test_exp" not in tracker.active_experiments

class TestModelTraining:
    """Test model training functionality."""

    def test_train_model(self, sample_data):
        """Test model training."""
        X, y = sample_data
        model, metrics = train_model(X, y)
        
        assert isinstance(model, ThrottlingModel)
        assert "mse" in metrics
        assert "r2" in metrics
    
    def test_validate_model(self, trained_model, sample_data):
        """Test model validation."""
        X, y = sample_data
        is_valid, metrics = validate_model(trained_model, X, y)
        
        assert isinstance(is_valid, bool)
        assert "accuracy" in metrics
    
    def test_update_model(self, trained_model, sample_data):
        """Test model updating."""
        X, y = sample_data
        updated_model, metrics = update_model(trained_model, X, y)
        
        assert isinstance(updated_model, ThrottlingModel)
        assert hasattr(updated_model, "previous_data")
        assert "mse" in metrics

class TestModelEvaluation:
    """Test model evaluation functionality."""

    def test_evaluator_initialization(self, temp_output_dir):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator(temp_output_dir)
        assert "mse" in evaluator.metrics
    
    def test_performance_evaluation(self, trained_model, sample_data):
        """Test performance evaluation."""
        X, y = sample_data
        evaluator = ModelEvaluator(Path("/tmp"))
        
        metrics = evaluator.evaluate_performance(trained_model, X, y)
        assert "mse" in metrics
        assert "r2" in metrics
    
    def test_error_analysis(self, trained_model, sample_data):
        """Test error analysis."""
        X, y = sample_data
        evaluator = ModelEvaluator(Path("/tmp"))
        
        predictions = trained_model.predict(X)
        analysis = evaluator.analyze_errors(predictions, y)
        
        assert "error_mean" in analysis
        assert "error_std" in analysis
        assert "outliers" in analysis
    
    def test_report_generation(self, trained_model, sample_data, temp_output_dir):
        """Test report generation."""
        X, y = sample_data
        evaluator = ModelEvaluator(temp_output_dir)
        
        report = evaluator.generate_report(trained_model, X, y)
        assert isinstance(report, dict)
        assert "metrics" in report
        assert "error_analysis" in report
        assert "data_stats" in report

def test_end_to_end(sample_data, temp_output_dir):
    """Test complete ML pipeline."""
    X, y = sample_data
    
    # Train model
    model, train_metrics = train_model(X, y)
    
    # Validate model
    is_valid, val_metrics = validate_model(model, X, y)
    
    # Update model
    updated_model, update_metrics = update_model(model, X[:100], y[:100])
    
    # Evaluate model
    evaluator = ModelEvaluator(temp_output_dir)
    results = analyze_results(evaluator, updated_model, X, y)
    
    assert "metrics" in results
    assert "error_analysis" in results
    assert "report" in results

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
