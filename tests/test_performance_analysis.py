#!/usr/bin/env python3
"""Test suite for performance analysis components."""

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
def sample_data():
    """Generate sample test data."""
    # Create mixture of two normal distributions
    np.random.seed(42)
    n_samples = 1000
    
    # First component
    samples1 = np.random.normal(0, 1, n_samples // 2)
    # Second component
    samples2 = np.random.normal(3, 0.5, n_samples // 2)
    
    return np.concatenate([samples1, samples2])

@pytest.fixture
def sample_mixture_fit():
    """Create sample mixture fit."""
    return MixtureFit(
        n_components=2,
        components=[
            MixtureComponent(
                weight=0.5,
                params={"mean": 0.0, "std": 1.0},
                distribution="normal"
            ),
            MixtureComponent(
                weight=0.5,
                params={"mean": 3.0, "std": 0.5},
                distribution="normal"
            )
        ],
        bic=2000.0,
        aic=1950.0,
        likelihood=-970.0,
        entropy=0.693
    )

@pytest.fixture
def sample_time_series():
    """Generate time series test data."""
    np.random.seed(42)
    times = pd.date_range(
        start=datetime.now(),
        periods=100,
        freq="1H"
    )
    values = np.random.normal(10, 2, 100)
    
    # Add trend
    values += np.linspace(0, 5, 100)
    
    return pd.Series(values, index=times)

class TestMixtureAnalysis:
    """Test mixture model analysis functionality."""
    
    def test_component_similarity(self, sample_mixture_fit):
        """Test component similarity calculation."""
        analyzer = MixtureAnalyzer(Path("test_dir"))
        
        comp1, comp2 = sample_mixture_fit.components
        similarity = analyzer.calculate_component_similarity(comp1, comp2)
        
        assert 0 <= similarity <= 1
        # Components are different, so similarity should be low
        assert similarity < 0.5
    
    def test_track_components(self, sample_data):
        """Test component tracking over time."""
        analyzer = MixtureAnalyzer(Path("test_dir"))
        
        # Create sequence of mixture fits
        fits = []
        times = [
            datetime.now() + timedelta(hours=i)
            for i in range(5)
        ]
        
        for t in times:
            # Add some noise to components
            noise = np.random.normal(0, 0.1)
            fit = MixtureFit(
                n_components=2,
                components=[
                    MixtureComponent(
                        weight=0.5,
                        params={"mean": 0.0 + noise, "std": 1.0},
                        distribution="normal"
                    ),
                    MixtureComponent(
                        weight=0.5,
                        params={"mean": 3.0 + noise, "std": 0.5},
                        distribution="normal"
                    )
                ],
                bic=2000.0,
                aic=1950.0,
                likelihood=-970.0,
                entropy=0.693
            )
            fits.append((t, fit))
        
        tracks = analyzer.track_components(fits)
        
        assert len(tracks) == 2  # Should track two components
        assert all(len(t.timestamps) > 1 for t in tracks)  # Each track should have multiple points
    
    def test_detect_transitions(self, sample_mixture_fit):
        """Test transition detection between components."""
        analyzer = MixtureAnalyzer(Path("test_dir"))
        
        # Create tracks with clear transition
        track1 = ComponentTrack(
            track_id=1,
            timestamps=[datetime.now() + timedelta(hours=i) for i in range(3)],
            weights=[0.5, 0.4, 0.3],
            means=[0.0, 0.5, 1.0],
            stds=[1.0, 1.0, 1.0],
            stability=0.8
        )
        
        track2 = ComponentTrack(
            track_id=2,
            timestamps=[datetime.now() + timedelta(hours=i) for i in range(3)],
            weights=[0.3, 0.4, 0.5],
            means=[3.0, 2.5, 2.0],
            stds=[0.5, 0.5, 0.5],
            stability=0.7
        )
        
        transitions = analyzer.detect_transitions([track1, track2])
        
        assert len(transitions) > 0
        assert all(hasattr(t, "type") for t in transitions)
        assert all(t.similarity > 0 for t in transitions)

class TestDistributionAnalysis:
    """Test distribution fitting and analysis."""
    
    def test_distribution_fitting(self, sample_data):
        """Test fitting distributions to data."""
        analyzer = DistributionAnalyzer(Path("test_dir"))
        
        fit = analyzer.fit_distribution(sample_data, "norm")
        
        assert isinstance(fit, FittedDistribution)
        assert "mean" in fit.params
        assert "std" in fit.params
        assert fit.aic is not None
        assert fit.bic is not None
    
    def test_qq_plot_data(self, sample_data):
        """Test Q-Q plot data generation."""
        analyzer = DistributionAnalyzer(Path("test_dir"))
        
        fit = analyzer.fit_distribution(sample_data, "norm")
        theoretical, empirical = analyzer.create_qq_plot_data(sample_data, fit)
        
        assert len(theoretical) == len(empirical)
        assert len(theoretical) > 0

class TestSensitivityAnalysis:
    """Test sensitivity analysis functionality."""
    
    def test_elasticity_calculation(self, sample_time_series):
        """Test elasticity calculation."""
        analyzer = SensitivityAnalyzer(Path("test_dir"))
        
        parameter_values = np.linspace(0.1, 1.0, 10)
        impact_values = parameter_values ** 2  # Quadratic relationship
        
        elasticity = analyzer._calculate_elasticity(
            parameter_values,
            impact_values
        )
        
        assert elasticity > 0  # Should be positive for increasing function
        assert abs(elasticity - 2.0) < 0.1  # Should be close to 2 for quadratic

class TestUncertaintyAnalysis:
    """Test uncertainty quantification."""
    
    def test_confidence_bounds(self, sample_data):
        """Test confidence bound calculation."""
        analyzer = UncertaintyAnalyzer(Path("test_dir"))
        
        bounds = analyzer.calculate_uncertainty_bounds(
            sample_data,
            confidence_level=0.95
        )
        
        assert isinstance(bounds, UncertaintyBounds)
        assert all(bounds.lower <= bounds.upper)
        assert len(bounds.lower) == len(sample_data)

def test_end_to_end(sample_data, tmp_path):
    """Test full analysis pipeline."""
    # Create test directory structure
    history_dir = tmp_path / "performance_history"
    history_dir.mkdir()
    
    # Save sample data
    data_file = history_dir / "test_data.json"
    data_file.write_text(json.dumps({
        "values": sample_data.tolist(),
        "timestamp": datetime.now().isoformat()
    }))
    
    # Run mixture analysis
    mixture_analyzer = MixtureAnalyzer(history_dir)
    model, score = mixture_analyzer.fit_gaussian_mixture(sample_data)
    mixture_fit = mixture_analyzer.analyze_components(model, sample_data)
    
    assert mixture_fit.n_components >= 2  # Should detect at least 2 components
    
    # Run distribution analysis
    dist_analyzer = DistributionAnalyzer(history_dir)
    fits = dist_analyzer.analyze_distributions(sample_data)
    
    assert len(fits) > 0
    assert any(f.p_value > 0.05 for f in fits)  # At least one good fit
    
    # Run sensitivity analysis
    sensitivity_analyzer = SensitivityAnalyzer(history_dir)
    parameter_values = np.linspace(0.1, 1.0, 10)
    result = sensitivity_analyzer.analyze_sensitivity(
        parameter_values,
        sample_data
    )
    
    assert isinstance(result, SensitivityResult)
    
    # Run uncertainty analysis
    uncertainty_analyzer = UncertaintyAnalyzer(history_dir)
    bounds = uncertainty_analyzer.calculate_uncertainty_bounds(
        sample_data,
        confidence_level=0.95
    )
    
    assert isinstance(bounds, UncertaintyBounds)
    assert all(bounds.lower <= bounds.upper)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
