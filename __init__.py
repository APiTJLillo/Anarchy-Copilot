"""Anarchy Copilot - AI-powered bug bounty suite."""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.resolve()

# Data directory for storing scan results, configs, etc.
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

try:
    from recon_module import (
        ReconManager,
        ReconResult,
        ScanResult,
        ReconScheduler
    )
except ImportError:
    # Fall back to local import when package is not installed
    from .recon_module import (
        ReconManager,
        ReconResult,
        ScanResult,
        ReconScheduler
    )

__version__ = '0.1.0'
__all__ = [
    'ReconManager',
    'ReconResult',
    'ScanResult',
    'ReconScheduler',
    'ROOT_DIR',
    'DATA_DIR'
]
