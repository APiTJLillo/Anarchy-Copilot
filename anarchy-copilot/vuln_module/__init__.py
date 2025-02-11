"""Vulnerability Discovery Module for Anarchy Copilot."""

from .models import (
    VulnResult,
    VulnSeverity,
    ScanConfig
)
from .scanner.base import VulnScanner
from .scanner.nuclei import NucleiScanner
from .vuln_manager import VulnManager

__all__ = [
    'VulnResult',
    'VulnSeverity',
    'ScanConfig',
    'VulnScanner',
    'NucleiScanner',
    'VulnManager'
]

__version__ = '0.1.0'
