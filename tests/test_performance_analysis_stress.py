#!/usr/bin/env python3
"""Stress tests for performance analysis components."""

import pytest
import numpy as np
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime, timedelta
import gc

from scripts.analyze_mixtures import MixtureAnalyzer
from scripts.analyze_distributions import DistributionAnalyzer
from scripts.analyze_sensitivity import SensitivityAnalyzer
from scripts.analyze_uncertainty import UncertaintyAnalyzer

@pytest.fixture
def large_sample_data():
    """Generate large test datasets."""
    np.random.seed(42)
    return {
        "medium": np.random.normal(0, 1, 10**4),
        "large": np.random.normal(0, 1, 10**5),
        "xlarge": np.random.normal(0, 1, 10**6),
        "mixed": np.concatenate([
            np.random.normal(0, 1, 10**4),
            np.random.normal(5, 2, 10**4)
        ])
    }

@pytest.fixture
def concurrent_data():
    """Generate multiple datasets for concurrent testing."""
    np.random.seed(42)
    return [
        np.random.normal(i, 1, 10**4)
        for i in range(10)
    ]

class PerformanceMetrics:
    """Track performance metrics during tests."""
    
    def __init__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
        self.peak_memory = self.start_memory
        self.duration = 0
        self.cpu_percent = 0
        
    def stop(self):
        """Stop tracking and calculate metrics."""
        self.duration = time.time() - self.start_time
        current_memory = psutil.Process().memory_info().rss
        self.peak_memory = max(current_memory, self.peak_memory)
        self.cpu_percent = psutil.Process().cpu_percent()
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "duration_seconds": self.duration,
            "peak_memory_mb": self.peak_memory / (1024 * 1024),
            "memory_increase_mb": (self.peak_memory - self.start_memory) / (1024 * 1024),
            "cpu_percent": self.cpu_percent
        }

class TestMixtureAnalysisStress:
    """Stress test mixture model analysis."""
    
    @pytest.mark.parametrize("size_key", ["medium", "large", "xlarge"])
    def test_mixture_scaling(self, large_sample_data, size_key):
        """Test mixture analysis scaling with data size."""
        analyzer = MixtureAnalyzer(Path("test_dir"))
        data = large_sample_data[size_key]
        
        metrics = PerformanceMetrics()
        model, score = analyzer.fit_gaussian_mixture(data)
        metrics.stop()
        
        # Verify performance bounds
        assert metrics.duration < 60  # Should complete within 60 seconds
        assert metrics.peak_memory < 4 * 1024 * 1024 * 1024  # Should use less than 4GB
        
        # Verify result quality
        mixture_fit = analyzer.analyze_components(model, data)
        assert 1 <= mixture_fit.n_components <= analyzer.config["mixture"]["max_components"]
    
    def test_concurrent_mixture_analysis(self, concurrent_data):
        """Test concurrent mixture analysis."""
        analyzer = MixtureAnalyzer(Path("test_dir"))
        
        def analyze_data(data: np.ndarray) -> Tuple[int, float]:
            model, score = analyzer.fit_gaussian_mixture(data)
            mixture_fit = analyzer.analyze_components(model, data)
            return mixture_fit.n_components, score
        
        metrics = PerformanceMetrics()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(analyze_data, concurrent_data))
        metrics.stop()
        
        # Verify all analyses completed
        assert len(results) == len(concurrent_data)
        
        # Verify reasonable resource usage
        assert metrics.duration < 120  # Should complete within 2 minutes
        assert metrics.cpu_percent < 400  # Should not exceed 400% CPU (4 cores)

class TestDistributionAnalysisStress:
    """Stress test distribution fitting analysis."""
    
    @pytest.mark.parametrize("dist_name", ["norm", "gamma", "lognorm"])
    def test_distribution_fitting_stress(self, large_sample_data, dist_name):
        """Test distribution fitting with large datasets."""
        analyzer = DistributionAnalyzer(Path("test_dir"))
        data = large_sample_data["large"]
        
        metrics = PerformanceMetrics()
        fit = analyzer.fit_distribution(data, dist_name)
        metrics.stop()
        
        assert metrics.duration < 30  # Should fit within 30 seconds
        assert fit.p_value is not None
    
    def test_multiple_distribution_comparison(self, large_sample_data):
        """Test comparing multiple distributions simultaneously."""
        analyzer = DistributionAnalyzer(Path("test_dir"))
        data = large_sample_data["mixed"]
        
        metrics = PerformanceMetrics()
        fits = []
        for dist_name in ["norm", "gamma", "lognorm", "uniform", "expon"]:
            fit = analyzer.fit_distribution(data, dist_name)
            fits.append(fit)
        metrics.stop()
        
        assert len(fits) == 5
        assert metrics.duration < 60  # Should complete within 60 seconds

class TestSensitivityAnalysisStress:
    """Stress test sensitivity analysis."""
    
    def test_large_parameter_space(self, large_sample_data):
        """Test sensitivity analysis with large parameter space."""
        analyzer = SensitivityAnalyzer(Path("test_dir"))
        data = large_sample_data["medium"]
        
        # Create large parameter grid
        param_values = np.linspace(0.1, 1.0, 100)
        
        metrics = PerformanceMetrics()
        result = analyzer.analyze_sensitivity(param_values, data)
        metrics.stop()
        
        assert metrics.duration < 120  # Should complete within 2 minutes
        assert len(result.values) == len(param_values)
    
    def test_parallel_sensitivity_analysis(self, large_sample_data):
        """Test parallel sensitivity analysis."""
        analyzer = SensitivityAnalyzer(Path("test_dir"))
        data = large_sample_data["medium"]
        
        def analyze_subset(params: np.ndarray) -> Dict[str, Any]:
            return analyzer.analyze_sensitivity(params, data)
        
        # Split parameter space
        param_sets = np.array_split(np.linspace(0.1, 1.0, 100), 4)
        
        metrics = PerformanceMetrics()
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(analyze_subset, param_sets))
        metrics.stop()
        
        assert len(results) == 4
        assert metrics.cpu_percent < 400  # Should not exceed 400% CPU

class TestUncertaintyAnalysisStress:
    """Stress test uncertainty analysis."""
    
    @pytest.mark.parametrize("n_samples", [100, 1000, 10000])
    def test_bootstrap_scaling(self, n_samples):
        """Test bootstrap analysis scaling."""
        analyzer = UncertaintyAnalyzer(Path("test_dir"))
        data = np.random.normal(0, 1, n_samples)
        
        metrics = PerformanceMetrics()
        bounds = analyzer.calculate_uncertainty_bounds(
            data,
            confidence_level=0.95
        )
        metrics.stop()
        
        # Verify scaling is roughly linear
        assert metrics.duration < (n_samples * 0.001)  # 1ms per sample max
    
    def test_memory_cleanup(self, large_sample_data):
        """Test memory cleanup during analysis."""
        analyzer = UncertaintyAnalyzer(Path("test_dir"))
        data = large_sample_data["xlarge"]
        
        initial_memory = psutil.Process().memory_info().rss
        metrics = PerformanceMetrics()
        
        for _ in range(5):
            bounds = analyzer.calculate_uncertainty_bounds(
                data,
                confidence_level=0.95
            )
            gc.collect()  # Force garbage collection
        
        metrics.stop()
        final_memory = psutil.Process().memory_info().rss
        
        # Verify no significant memory leak
        assert (final_memory - initial_memory) < 100 * 1024 * 1024  # Less than 100MB growth

def test_full_pipeline_stress(large_sample_data, tmp_path):
    """Test full analysis pipeline under stress."""
    history_dir = tmp_path / "performance_history"
    history_dir.mkdir()
    
    # Initialize analyzers
    mixture_analyzer = MixtureAnalyzer(history_dir)
    dist_analyzer = DistributionAnalyzer(history_dir)
    sensitivity_analyzer = SensitivityAnalyzer(history_dir)
    uncertainty_analyzer = UncertaintyAnalyzer(history_dir)
    
    data = large_sample_data["large"]
    parameter_values = np.linspace(0.1, 1.0, 50)
    
    metrics = PerformanceMetrics()
    
    try:
        # Run full pipeline
        model, score = mixture_analyzer.fit_gaussian_mixture(data)
        mixture_fit = mixture_analyzer.analyze_components(model, data)
        
        fits = dist_analyzer.analyze_distributions(data)
        
        sensitivity_result = sensitivity_analyzer.analyze_sensitivity(
            parameter_values,
            data
        )
        
        bounds = uncertainty_analyzer.calculate_uncertainty_bounds(
            data,
            confidence_level=0.95
        )
        
    finally:
        metrics.stop()
        
        # Save performance metrics
        metrics_file = history_dir / "stress_test_metrics.json"
        metrics_file.write_text(json.dumps(metrics.to_dict(), indent=2))
    
    # Verify overall performance
    assert metrics.duration < 300  # Full pipeline within 5 minutes
    assert metrics.peak_memory < 8 * 1024 * 1024 * 1024  # Less than 8GB peak memory

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
