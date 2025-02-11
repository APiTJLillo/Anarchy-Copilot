"""Vulnerability Discovery Module for Anarchy Copilot."""

from .vuln_manager import VulnManager
from .models import (
    VulnResult,
    PayloadResult,
    VulnSeverity,
    PayloadType,
    ScanConfig
)
from .scanner.base import VulnScanner
from .scanner.nuclei import NucleiScanner

__all__ = [
    'VulnManager',
    'VulnResult',
    'PayloadResult',
    'VulnSeverity',
    'PayloadType',
    'ScanConfig',
    'VulnScanner',
    'NucleiScanner',
]

__version__ = '0.1.0'
