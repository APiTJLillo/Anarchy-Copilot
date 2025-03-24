#!/usr/bin/env python3
"""Profile resource usage of performance analysis components."""

import pytest
import numpy as np
import cProfile
import pstats
import io
import time
import psutil
import tracemalloc
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import json
from datetime import datetime
from functools import wraps
from contextlib import contextmanager

from scripts.analyze_mixtures import MixtureAnalyzer
from scripts.analyze_distributions import DistributionAnalyzer
from scripts.analyze_sensitivity import SensitivityAnalyzer
from scripts.analyze_uncertainty import UncertaintyAnalyzer

class ResourceProfile:
    """Detailed resource usage profiling."""
    
    def __init__(self):
        self.cpu_stats: Optional[pstats.Stats] = None
        self.memory_peak = 0
        self.memory_blocks: List[tracemalloc.Snapshot] = []
        self.io_counters_start = None
        self.io_counters_end = None
        self.duration = 0
        self.start_time = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        result = {
            "duration_seconds": self.duration,
            "peak_memory_mb": self.memory_peak / (1024 * 1024),
            "top_functions": []
        }
        
        if self.cpu_stats:
            # Get top 10 time-consuming functions
            stream = io.StringIO()
            self.cpu_stats.sort_stats("cumulative").print_stats(10)
            result["top_functions"] = stream.getvalue().split("\n")[3:13]
        
        if self.io_counters_end and self.io_counters_start:
            result["io_operations"] = {
                "read_bytes": self.io_counters_end.read_bytes - self.io_counters_start.read_bytes,
                "write_bytes": self.io_counters_end.write_bytes - self.io_counters_start.write_bytes
            }
        
        return result

@contextmanager
def profile_resources() -> ResourceProfile:
    """Context manager for resource profiling."""
    profile = ResourceProfile()
    
    # Start profilers
    cpu_profiler = cProfile.Profile()
    cpu_profiler.enable()
    
    tracemalloc.start()
    profile.io_counters_start = psutil.Process().io_counters()
    profile.start_time = time.time()
    
    try:
        yield profile
    finally:
        # Stop profilers and collect stats
        cpu_profiler.disable()
        profile.duration = time.time() - profile.start_time
        
        # CPU stats
        stats_stream = io.StringIO()
        stats = pstats.Stats(cpu_profiler, stream=stats_stream)
        stats.sort_stats("cumulative")
        profile.cpu_stats = stats
        
        # Memory stats
        profile.memory_peak = tracemalloc.get_traced_memory()[1]
        profile.memory_blocks = tracemalloc.take_snapshot()
        
        # IO stats
        profile.io_counters_end = psutil.Process().io_counters()
        
        tracemalloc.stop()

def profile_function(func: Callable) -> Callable:
    """Decorator to profile a function."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        with profile_resources() as profile:
            result = func(*args, **kwargs)
            
        # Save profile data
        profile_dir = Path("profiling_results")
        profile_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_file = profile_dir / f"{func.__name__}_{timestamp}.json"
        profile_file.write_text(json.dumps(profile.to_dict(), indent=2))
        
        return result
    return wrapper

@pytest.fixture
def profiling_data():
    """Generate test data for profiling."""
    np.random.seed(42)
    sizes = [100, 1000, 10000, 100000]
    return {
        f"size_{size}": np.random.normal(0, 1, size)
        for size in sizes
    }

class TestMixtureAnalysisProfile:
    """Profile mixture model analysis."""
    
    @pytest.mark.parametrize("size_key", ["size_1000", "size_10000", "size_100000"])
    def test_mixture_profile(self, profiling_data, size_key):
        """Profile mixture analysis with different data sizes."""
        analyzer = MixtureAnalyzer(Path("test_dir"))
        data = profiling_data[size_key]
        
        with profile_resources() as profile:
            model, score = analyzer.fit_gaussian_mixture(data)
            mixture_fit = analyzer.analyze_components(model, data)
        
        # Analyze performance characteristics
        stats = profile.to_dict()
        
        # Memory should scale roughly linearly
        expected_memory = len(data) * 8 * 3  # Rough estimate: 3x data size
        assert stats["peak_memory_mb"] < expected_memory / 1024 / 1024 * 2
        
        # Time should scale no worse than O(n log n)
        n = len(data)
        expected_time = n * np.log2(n) * 1e-5  # Rough baseline in seconds
        assert stats["duration_seconds"] < expected_time * 3

class TestDistributionAnalysisProfile:
    """Profile distribution fitting analysis."""
    
    @profile_function
    def fit_distributions(self, data: np.ndarray) -> List[Any]:
        """Profile multiple distribution fits."""
        analyzer = DistributionAnalyzer(Path("test_dir"))
        distributions = ["norm", "gamma", "lognorm", "uniform", "expon"]
        
        return [
            analyzer.fit_distribution(data, dist)
            for dist in distributions
        ]
    
    def test_distribution_profile(self, profiling_data):
        """Profile distribution fitting performance."""
        results = {}
        
        for size_key, data in profiling_data.items():
            results[size_key] = self.fit_distributions(data)
            
            # Verify results
            assert len(results[size_key]) == 5
            assert all(fit is not None for fit in results[size_key])

class TestSensitivityAnalysisProfile:
    """Profile sensitivity analysis."""
    
    @profile_function
    def analyze_parameter_sensitivity(
        self,
        data: np.ndarray,
        n_params: int
    ) -> Any:
        """Profile sensitivity analysis with varying parameter counts."""
        analyzer = SensitivityAnalyzer(Path("test_dir"))
        params = np.linspace(0.1, 1.0, n_params)
        
        return analyzer.analyze_sensitivity(params, data)
    
    @pytest.mark.parametrize("n_params", [10, 50, 100])
    def test_sensitivity_profile(self, profiling_data, n_params):
        """Profile sensitivity analysis scaling."""
        data = profiling_data["size_10000"]
        result = self.analyze_parameter_sensitivity(data, n_params)
        
        # Verify results scale reasonably with parameter count
        assert len(result.values) == n_params

class TestUncertaintyAnalysisProfile:
    """Profile uncertainty analysis."""
    
    @profile_function
    def analyze_uncertainty(
        self,
        data: np.ndarray,
        n_bootstrap: int
    ) -> Any:
        """Profile uncertainty analysis with varying bootstrap samples."""
        analyzer = UncertaintyAnalyzer(Path("test_dir"))
        analyzer.config["bootstrap"]["n_iterations"] = n_bootstrap
        
        return analyzer.calculate_uncertainty_bounds(data, confidence_level=0.95)
    
    @pytest.mark.parametrize("n_bootstrap", [100, 500, 1000])
    def test_uncertainty_profile(self, profiling_data, n_bootstrap):
        """Profile uncertainty analysis with different bootstrap sizes."""
        data = profiling_data["size_10000"]
        bounds = self.analyze_uncertainty(data, n_bootstrap)
        
        # Verify memory scales linearly with bootstrap samples
        profile_file = list(Path("profiling_results").glob("analyze_uncertainty_*.json"))[-1]
        profile = json.loads(profile_file.read_text())
        
        expected_memory = len(data) * n_bootstrap * 8  # Rough estimate
        assert profile["peak_memory_mb"] < expected_memory / 1024 / 1024 * 2

def test_full_pipeline_profile(profiling_data, tmp_path):
    """Profile full analysis pipeline."""
    history_dir = tmp_path / "performance_history"
    history_dir.mkdir()
    
    data = profiling_data["size_10000"]
    
    with profile_resources() as profile:
        # Run full pipeline
        mixture_analyzer = MixtureAnalyzer(history_dir)
        model, score = mixture_analyzer.fit_gaussian_mixture(data)
        mixture_fit = mixture_analyzer.analyze_components(model, data)
        
        dist_analyzer = DistributionAnalyzer(history_dir)
        fits = dist_analyzer.analyze_distributions(data)
        
        sensitivity_analyzer = SensitivityAnalyzer(history_dir)
        params = np.linspace(0.1, 1.0, 50)
        sensitivity_result = sensitivity_analyzer.analyze_sensitivity(params, data)
        
        uncertainty_analyzer = UncertaintyAnalyzer(history_dir)
        bounds = uncertainty_analyzer.calculate_uncertainty_bounds(
            data,
            confidence_level=0.95
        )
    
    # Save pipeline profile
    stats = profile.to_dict()
    pipeline_profile = history_dir / "pipeline_profile.json"
    pipeline_profile.write_text(json.dumps(stats, indent=2))
    
    # Verify overall resource usage
    assert stats["duration_seconds"] < 300  # Should complete within 5 minutes
    assert stats["peak_memory_mb"] < 4 * 1024  # Should use less than 4GB
    
    # Verify component contributions
    if stats["top_functions"]:
        component_times = {
            "mixture": sum(t for t in stats["top_functions"] if "mixture" in t.lower()),
            "distribution": sum(t for t in stats["top_functions"] if "distribution" in t.lower()),
            "sensitivity": sum(t for t in stats["top_functions"] if "sensitivity" in t.lower()),
            "uncertainty": sum(t for t in stats["top_functions"] if "uncertainty" in t.lower())
        }
        
        # No single component should dominate
        for component, time in component_times.items():
            assert time < stats["duration_seconds"] * 0.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
