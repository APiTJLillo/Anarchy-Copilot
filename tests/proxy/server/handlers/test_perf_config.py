"""Tests for performance test configuration."""
import pytest
import os
import tempfile
from pathlib import Path
import yaml
from typing import Dict, Any

from .perf_config import (
    PerformanceSettings,
    load_config,
    PERFORMANCE_PROFILES
)

@pytest.fixture
def temp_config_file():
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump({
            "concurrent_connections": 50,
            "request_count": 500,
            "min_throughput": 2000.0,
            "data_sizes": [512, 2048, 4096]
        }, f)
    yield Path(f.name)
    os.unlink(f.name)

@pytest.fixture
def env_vars():
    """Set up test environment variables."""
    original = dict(os.environ)
    os.environ.update({
        "PERF_CONCURRENT_CONNECTIONS": "75",
        "PERF_MIN_THROUGHPUT": "1500.0",
        "PERF_DISABLE_GC": "true"
    })
    yield
    os.environ.clear()
    os.environ.update(original)

def test_default_settings():
    """Test default configuration values."""
    config = PerformanceSettings()
    
    assert config.concurrent_connections == 100
    assert config.request_count == 1000
    assert config.min_throughput == 1000.0
    assert len(config.data_sizes) == 3
    assert not config.disable_gc
    assert config.report_format == "html"

def test_load_from_file(temp_config_file):
    """Test loading configuration from file."""
    config = PerformanceSettings.load_from_file(temp_config_file)
    
    assert config.concurrent_connections == 50
    assert config.request_count == 500
    assert config.min_throughput == 2000.0
    assert config.data_sizes == [512, 2048, 4096]

def test_save_to_file(tmp_path):
    """Test saving configuration to file."""
    config = PerformanceSettings(
        concurrent_connections=150,
        request_count=1500,
        min_throughput=3000.0
    )
    
    save_path = tmp_path / "test_config.yml"
    config.save_to_file(save_path)
    
    # Load and verify
    loaded = PerformanceSettings.load_from_file(save_path)
    assert loaded.concurrent_connections == 150
    assert loaded.request_count == 1500
    assert loaded.min_throughput == 3000.0

def test_environment_variables(env_vars):
    """Test loading settings from environment variables."""
    config = PerformanceSettings()
    
    assert config.concurrent_connections == 75
    assert config.min_throughput == 1500.0
    assert config.disable_gc is True

def test_data_distribution_validation():
    """Test data distribution validation."""
    with pytest.raises(ValueError):
        PerformanceSettings(
            data_distributions={
                "small": 0.5,
                "large": 0.2
                # Sum < 1.0, should fail
            }
        )

def test_test_profiles():
    """Test predefined test profiles."""
    config = PerformanceSettings()
    
    # Quick profile
    quick = config.get_test_profile("quick")
    assert quick["concurrent_connections"] == 10
    assert quick["request_count"] == 100
    assert len(quick["data_sizes"]) == 1
    
    # Stress profile
    stress = config.get_test_profile("stress")
    assert stress["concurrent_connections"] == config.concurrent_connections * 2
    assert stress["request_count"] == config.request_count * 2
    assert len(stress["data_sizes"]) > len(config.data_sizes)

def test_invalid_config_file():
    """Test handling of invalid configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("invalid: yaml: content")
    
    try:
        config = PerformanceSettings.load_from_file(Path(f.name))
        # Should return default config on error
        assert config.concurrent_connections == 100
    finally:
        os.unlink(f.name)

def test_performance_profiles():
    """Test performance profile definitions."""
    assert "ci" in PERFORMANCE_PROFILES
    assert "baseline" in PERFORMANCE_PROFILES
    assert "stress" in PERFORMANCE_PROFILES
    
    # Check CI profile
    ci_profile = PERFORMANCE_PROFILES["ci"]
    assert "description" in ci_profile
    assert "settings" in ci_profile
    assert ci_profile["settings"]["concurrent_connections"] == 50

def test_config_validation():
    """Test configuration validation."""
    # Invalid connections
    with pytest.raises(ValueError):
        PerformanceSettings(concurrent_connections=-1)
    
    # Invalid latency threshold
    with pytest.raises(ValueError):
        PerformanceSettings(max_latency_p95=-0.1)
    
    # Invalid regression threshold
    with pytest.raises(ValueError):
        PerformanceSettings(throughput_regression_threshold=2.0)

def test_config_inheritance():
    """Test configuration inheritance and overrides."""
    base_config = PerformanceSettings()
    
    # Create derived config
    derived = PerformanceSettings(
        **base_config.dict(),
        concurrent_connections=200,
        request_count=2000
    )
    
    assert derived.concurrent_connections == 200
    assert derived.request_count == 2000
    assert derived.warmup_requests == base_config.warmup_requests
    assert derived.data_sizes == base_config.data_sizes

@pytest.mark.parametrize("profile_name,expected_connections", [
    ("ci", 50),
    ("baseline", 100),
    ("stress", 200)
])
def test_profile_settings(profile_name, expected_connections):
    """Test different profile settings."""
    profile = PERFORMANCE_PROFILES[profile_name]
    assert profile["settings"]["concurrent_connections"] == expected_connections

def test_custom_profile():
    """Test creating and using custom profile."""
    config = PerformanceSettings()
    
    custom_profile = {
        "concurrent_connections": 25,
        "request_count": 250,
        "warmup_requests": 25,
        "data_sizes": [512]
    }
    
    # Add custom profile
    setattr(config, "custom_profile", custom_profile)
    
    # Use custom profile
    test_settings = config.get_test_profile("custom")
    assert test_settings == config.get_test_profile("standard")  # Falls back to standard

def test_output_paths(tmp_path):
    """Test output path handling."""
    config = PerformanceSettings(
        output_dir=tmp_path / "results",
        baseline_dir=tmp_path / "baselines"
    )
    
    assert config.output_dir.parent == tmp_path
    assert config.baseline_dir.parent == tmp_path
    
    # Test path creation
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.baseline_dir.mkdir(parents=True, exist_ok=True)
    
    assert config.output_dir.exists()
    assert config.baseline_dir.exists()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
