"""Version information for Anarchy Copilot."""

import sys
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any
import pkg_resources

# Version components
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 0

# Build version string
__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"

# Component versions - all components share the same version in development
COMPONENT_VERSIONS: Dict[str, str] = {
    "api": __version__,
    "recon_module": __version__,
    "vuln_module": __version__,
    "core": __version__
}

# Minimum required versions of dependencies
DEPENDENCY_VERSIONS: Dict[str, str] = {
    "fastapi": ">=0.68.0",
    "sqlalchemy": ">=1.4.0",
    "uvicorn": ">=0.15.0",
    "aiohttp": ">=3.8.0",
    "asyncio": ">=3.4.3",
    "pydantic": ">=1.8.0",
    "pyppeteer": ">=1.0.2",
    "mitmproxy": ">=9.0.0",
}

# Build information
BUILD_INFO: Dict[str, str] = {
    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "git_commit": "development",  # Placeholder for actual git commit
    "git_branch": "main",        # Placeholder for actual git branch
    "build_number": "0",         # Placeholder for CI build number
    "build_type": "development"  # development, release, etc.
}

def get_version() -> str:
    """Return the current version string."""
    return __version__

def get_version_info() -> Dict[str, Any]:
    """Return detailed version information."""
    return {
        "version": __version__,
        "build": BUILD_INFO,
        "components": COMPONENT_VERSIONS,
        "dependencies": DEPENDENCY_VERSIONS,
        "metadata": {
            "author": "Anarchy Copilot Team",
            "license": "MIT"
        }
    }

def check_compatibility() -> bool:
    """Check if the current environment is compatible."""
    # Check Python version
    if sys.version_info < (3, 8):
        return False

    # Check package dependencies
    try:
        for dep, ver in DEPENDENCY_VERSIONS.items():
            pkg_resources.require(f"{dep}{ver}")
    except Exception:
        return False

    # Check Nuclei availability
    try:
        result = subprocess.run(["nuclei", "-version"], 
                              capture_output=True, 
                              check=False)
        if result.returncode != 0:
            return False
    except FileNotFoundError:
        return False

    return True
