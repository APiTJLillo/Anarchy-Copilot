#!/usr/bin/env python3
"""Test failure cases for performance analysis components."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import asdict

from scripts.analyze_mixtures import (
    MixtureAnalyzer,
    MixtureComponent,
    MixtureFit,
    ClusterAssignment
)
from scripts.analyze_distributions import (
    DistributionAnalyzer,
    FittedDistribution,
    DistributionFitResult
)
from scripts.analyze_sensitivity import (
    SensitivityAnalyzer,
    SensitivityResult
)
from scripts.analyze_uncertainty import (
    UncertaintyAnalyzer,
    UncertaintyBounds,
    UncertaintyResult
)

@pytest.fixture
def invalid_data():
    """Generate invalid test data scenarios."""
    return {
        "empty": np.array([]),
        "single": np.array([1.0]),
        "constant": np.array([1.0] * 100),
        "nans": np.array([np.nan] * 10 + [1.0] * 90),
        "infs": np.array([np.inf] * 5 + [1.0] * 95),
        "mixed_types": np.array([1, "2", 3.0]),
        "huge": np.random.normal(0, 1, 1000000),  # Memory/performance test
        "tiny": np.random.normal(0, 1e-10, 100),  # Numerical stability test
        "outliers": np.array([0] * 99 + [1e6]),  # Extreme outlier
    }

@pytest.fixture
def corrupt_files(tmp_path):
    """Create corrupt file scenarios."""
    files = {}
    
    # Invalid JSON
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{not valid json")
    files["invalid_json"] = invalid_json
    
    # Empty file
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("")
    files["empty"] = empty_file
    
    # Missing required fields
    missing_fields = tmp_path / "missing_fields.json"
    missing_fields.write_text('{"some": "data"}')
    files["missing_fields"] = missing_fields
    
    # Wrong data types
    wrong_types = tmp_path / "wrong_types.json"
    wrong_types.write_text('{"values": "not a list", "timestamp": 123}')
    files["wrong_types"] = wrong_types
    
    return files

class TestMixtureAnalysisFailures:
    """Test mixture model analysis failure cases."""
    
    @pytest.mark.parametrize("data_key", [
        "empty", "single", "constant", "nans", "infs"
    ])
    def test_invalid_data(self, invalid_data, data_key):
        """Test handling of invalid input data."""
        analyzer = MixtureAnalyzer(Path("test_dir"))
        data = invalid_data[data_key]
        
        with pytest.raises(Exception) as exc_info:
            model, score = analyzer.fit_gaussian_mixture(data)
        
        assert "Invalid data" in str(exc_info.value)
    
    def test_component_limit_exceeded(self, sample_data):
        """Test handling of excessive components."""
        analyzer = MixtureAnalyzer(Path("test_dir"))
        analyzer.config["mixture"]["max_components"] = 100
        
        with pytest.raises(ValueError) as exc_info:
            model, score = analyzer.fit_gaussian_mixture(sample_data)
        
        assert "components" in str(exc_info.value)
    
    def test_convergence_failure(self, invalid_data):
        """Test handling of convergence failures."""
        analyzer = MixtureAnalyzer(Path("test_dir"))
        analyzer.config["mixture"]["convergence_tol"] = 1e-10
        analyzer.config["mixture"]["max_iter"] = 1
        
        with pytest.raises(Exception) as exc_info:
            model, score = analyzer.fit_gaussian_mixture(invalid_data["tiny"])
        
        assert "convergence" in str(exc_info.value).lower()

class TestDistributionAnalysisFailures:
    """Test distribution fitting failure cases."""
    
    @pytest.mark.parametrize("dist_name", [
        "not_a_dist",
        "",
        None,
        123
    ])
    def test_invalid_distribution(self, sample_data, dist_name):
        """Test handling of invalid distribution names."""
        analyzer = DistributionAnalyzer(Path("test_dir"))
        
        with pytest.raises(Exception):
            analyzer.fit_distribution(sample_data, dist_name)
    
    def test_distribution_fit_failures(self, invalid_data):
        """Test handling of distribution fitting failures."""
        analyzer = DistributionAnalyzer(Path("test_dir"))
        
        for key, data in invalid_data.items():
            if key in ["empty", "single", "nans", "infs"]:
                with pytest.raises(Exception):
                    analyzer.fit_distribution(data, "norm")

class TestSensitivityAnalysisFailures:
    """Test sensitivity analysis failure cases."""
    
    def test_invalid_parameter_values(self, sample_data):
        """Test handling of invalid parameter values."""
        analyzer = SensitivityAnalyzer(Path("test_dir"))
        
        invalid_params = [
            np.array([]),  # Empty
            np.array([1]),  # Single value
            np.array([1, 1, 1]),  # Constant values
            np.array([np.nan, 1, 2]),  # Contains NaN
            np.array([np.inf, 1, 2]),  # Contains inf
        ]
        
        for params in invalid_params:
            with pytest.raises(Exception):
                analyzer.analyze_sensitivity(params, sample_data)
    
    def test_incompatible_dimensions(self, sample_data):
        """Test handling of dimension mismatches."""
        analyzer = SensitivityAnalyzer(Path("test_dir"))
        
        # More parameter values than data points
        params = np.linspace(0, 1, len(sample_data) + 1)
        
        with pytest.raises(ValueError) as exc_info:
            analyzer.analyze_sensitivity(params, sample_data)
        
        assert "dimension" in str(exc_info.value).lower()

class TestUncertaintyAnalysisFailures:
    """Test uncertainty analysis failure cases."""
    
    def test_invalid_confidence_levels(self, sample_data):
        """Test handling of invalid confidence levels."""
        analyzer = UncertaintyAnalyzer(Path("test_dir"))
        
        invalid_levels = [-0.1, 0, 1.1, np.nan, np.inf]
        
        for level in invalid_levels:
            with pytest.raises(ValueError) as exc_info:
                analyzer.calculate_uncertainty_bounds(sample_data, level)
            
            assert "confidence" in str(exc_info.value).lower()
    
    def test_insufficient_data(self, invalid_data):
        """Test handling of insufficient data."""
        analyzer = UncertaintyAnalyzer(Path("test_dir"))
        
        with pytest.raises(ValueError) as exc_info:
            analyzer.calculate_uncertainty_bounds(invalid_data["single"])
        
        assert "insufficient" in str(exc_info.value).lower()

class TestFileHandlingFailures:
    """Test file handling failure cases."""
    
    def test_corrupt_files(self, corrupt_files):
        """Test handling of corrupt input files."""
        analyzer = MixtureAnalyzer(Path("test_dir"))
        
        for key, file_path in corrupt_files.items():
            with pytest.raises(Exception):
                with open(file_path) as f:
                    data = json.load(f)
                analyzer.fit_gaussian_mixture(np.array(data.get("values", [])))
    
    def test_missing_directory(self):
        """Test handling of missing directories."""
        non_existent = Path("does_not_exist")
        
        with pytest.raises(FileNotFoundError):
            MixtureAnalyzer(non_existent)
    
    def test_permission_errors(self, tmp_path):
        """Test handling of permission errors."""
        restricted_dir = tmp_path / "restricted"
        restricted_dir.mkdir()
        restricted_dir.chmod(0o000)  # Remove all permissions
        
        try:
            with pytest.raises(PermissionError):
                analyzer = MixtureAnalyzer(restricted_dir)
                analyzer.fit_gaussian_mixture(np.array([1, 2, 3]))
        finally:
            restricted_dir.chmod(0o755)  # Restore permissions

class TestMemoryAndPerformanceFailures:
    """Test memory and performance-related failure cases."""
    
    def test_memory_limits(self, invalid_data):
        """Test handling of memory-intensive operations."""
        analyzer = MixtureAnalyzer(Path("test_dir"))
        
        with pytest.raises(MemoryError):
            # Create data too large to fit in memory
            huge_data = np.random.normal(0, 1, 10**8)
            analyzer.fit_gaussian_mixture(huge_data)
    
    def test_computation_timeout(self, sample_data):
        """Test handling of computation timeouts."""
        analyzer = MixtureAnalyzer(Path("test_dir"))
        analyzer.config["mixture"]["max_iter"] = 1  # Force quick timeout
        
        with pytest.raises(Exception) as exc_info:
            model, score = analyzer.fit_gaussian_mixture(sample_data)
            
        assert "timeout" in str(exc_info.value).lower()

def test_end_to_end_failures(invalid_data, corrupt_files, tmp_path):
    """Test end-to-end failure scenarios."""
    history_dir = tmp_path / "performance_history"
    history_dir.mkdir()
    
    # Test with various invalid inputs
    for data_key, data in invalid_data.items():
        with pytest.raises(Exception):
            # Run full pipeline with invalid data
            mixture_analyzer = MixtureAnalyzer(history_dir)
            dist_analyzer = DistributionAnalyzer(history_dir)
            sensitivity_analyzer = SensitivityAnalyzer(history_dir)
            uncertainty_analyzer = UncertaintyAnalyzer(history_dir)
            
            try:
                # Attempt full analysis pipeline
                model, score = mixture_analyzer.fit_gaussian_mixture(data)
                mixture_fit = mixture_analyzer.analyze_components(model, data)
                
                fits = dist_analyzer.analyze_distributions(data)
                
                parameter_values = np.linspace(0.1, 1.0, 10)
                sensitivity_result = sensitivity_analyzer.analyze_sensitivity(
                    parameter_values,
                    data
                )
                
                bounds = uncertainty_analyzer.calculate_uncertainty_bounds(
                    data,
                    confidence_level=0.95
                )
            except Exception as e:
                assert "Invalid data" in str(e)
                raise

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
