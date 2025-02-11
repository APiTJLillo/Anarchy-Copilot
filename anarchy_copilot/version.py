"""Version information for Anarchy Copilot."""

__version__ = "0.1.0"
__author__ = "Anarchy Copilot Team"
__license__ = "MIT"

# Semantic versioning
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 0

# Build information
BUILD_INFO = {
    "timestamp": "2025-02-10T12:00:00Z",
    "git_commit": "development",
    "git_branch": "main",
    "build_number": "dev",
    "build_type": "development"
}

# Component versions
COMPONENT_VERSIONS = {
    "api": __version__,
    "recon_module": __version__,
    "vuln_module": __version__,
    "core": __version__
}

# Dependency requirements
DEPENDENCY_VERSIONS = {
    "nuclei": ">=2.8.0",
    "python": ">=3.8.0",
    "fastapi": ">=0.68.0",
    "sqlalchemy": ">=1.4.0"
}

def get_version() -> str:
    """Get full version string."""
    return __version__

def get_version_info() -> dict:
    """Get complete version information."""
    return {
        "version": __version__,
        "build": BUILD_INFO,
        "components": COMPONENT_VERSIONS,
        "dependencies": DEPENDENCY_VERSIONS,
        "metadata": {
            "author": __author__,
            "license": __license__
        }
    }

def check_compatibility() -> bool:
    """Check if current environment meets version requirements."""
    import sys
    import pkg_resources

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        return False
    
    try:
        # Check fastapi version
        pkg_resources.require(f"fastapi{DEPENDENCY_VERSIONS['fastapi']}")
        pkg_resources.require(f"sqlalchemy{DEPENDENCY_VERSIONS['sqlalchemy']}")
    except pkg_resources.VersionConflict:
        return False

    # Check nuclei version if available
    try:
        import subprocess
        result = subprocess.run(
            ["nuclei", "-version"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return False
        # Version parsing could be added here
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

    return True
