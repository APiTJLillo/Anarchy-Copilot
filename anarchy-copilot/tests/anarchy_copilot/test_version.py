"""Tests for version information."""

import pytest
import re
from unittest.mock import patch, MagicMock
from packaging.version import Version

from anarchy_copilot.version import (
    __version__,
    get_version,
    get_version_info,
    check_compatibility,
    VERSION_MAJOR,
    VERSION_MINOR,
    VERSION_PATCH,
    BUILD_INFO,
    COMPONENT_VERSIONS,
    DEPENDENCY_VERSIONS
)

def test_version_format():
    """Test version string format."""
    # Version should match semantic versioning
    pattern = r'^\d+\.\d+\.\d+$'
    assert re.match(pattern, __version__), "Version should follow semantic versioning"
    
    # Version components should match major.minor.patch
    version = Version(__version__)
    assert version.major == VERSION_MAJOR
    assert version.minor == VERSION_MINOR
    assert version.micro == VERSION_PATCH

def test_get_version():
    """Test get_version function."""
    version = get_version()
    assert version == __version__
    assert isinstance(version, str)

def test_version_info_structure():
    """Test version info dictionary structure."""
    info = get_version_info()
    
    # Check required keys
    assert "version" in info
    assert "build" in info
    assert "components" in info
    assert "dependencies" in info
    assert "metadata" in info
    
    # Check build info
    assert "timestamp" in info["build"]
    assert "git_commit" in info["build"]
    assert "git_branch" in info["build"]
    assert "build_number" in info["build"]
    assert "build_type" in info["build"]
    
    # Check components
    for component in ["api", "recon_module", "vuln_module", "core"]:
        assert component in info["components"]
    
    # Check metadata
    assert "author" in info["metadata"]
    assert "license" in info["metadata"]

def test_component_versions_consistency():
    """Test component versions are consistent."""
    # All components should have the same version in development
    assert len(set(COMPONENT_VERSIONS.values())) == 1
    assert all(v == __version__ for v in COMPONENT_VERSIONS.values())

@pytest.mark.parametrize("python_version,expected", [
    ((3, 8, 0), True),    # Minimum supported
    ((3, 9, 0), True),    # Supported
    ((3, 10, 0), True),   # Supported
    ((3, 7, 0), False),   # Too old
    ((2, 7, 0), False),   # Python 2
])
def test_python_version_compatibility(python_version, expected):
    """Test Python version compatibility check."""
    with patch('sys.version_info', python_version):
        with patch('pkg_resources.require') as mock_require:
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                mock_require.return_value = True
                assert check_compatibility() == expected

def test_dependency_version_check():
    """Test dependency version checking."""
    # Mock successful dependency checks
    with patch('pkg_resources.require') as mock_require:
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            mock_require.return_value = True
            assert check_compatibility()
        
        # Test dependency version conflict
        mock_require.side_effect = Exception("Version conflict")
        assert not check_compatibility()

def test_nuclei_compatibility():
    """Test Nuclei version compatibility check."""
    with patch('pkg_resources.require') as mock_require:
        mock_require.return_value = True
        
        # Test when nuclei is available
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert check_compatibility()
            
            # Test when nuclei version check fails
            mock_run.return_value = MagicMock(returncode=1)
            assert not check_compatibility()
        
        # Test when nuclei is not installed
        with patch('subprocess.run', side_effect=FileNotFoundError):
            assert not check_compatibility()

def test_build_info_format():
    """Test build information format."""
    # Check timestamp format
    timestamp_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$'
    assert re.match(timestamp_pattern, BUILD_INFO["timestamp"])
    
    # Check git commit format
    assert isinstance(BUILD_INFO["git_commit"], str)
    assert isinstance(BUILD_INFO["git_branch"], str)
    
    # Check build number and type
    assert isinstance(BUILD_INFO["build_number"], str)
    assert isinstance(BUILD_INFO["build_type"], str)

def test_dependency_versions_format():
    """Test dependency versions specification format."""
    version_pattern = r'^[>=<]+=\d+\.\d+\.\d+$'
    for dep_version in DEPENDENCY_VERSIONS.values():
        assert re.match(version_pattern, dep_version), f"Invalid version spec: {dep_version}"
