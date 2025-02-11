"""Models for reconnaissance module results."""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

import sqlalchemy.orm  # type: ignore

class ScanConfig:
    """Configuration for a reconnaissance scan."""

    def __init__(
        self,
        id: Optional[int] = None,
        target: str = "",
        network_scan_enabled: bool = False,
        port_scan_enabled: bool = False,
        dns_scan_enabled: bool = False,
        subdomain_scan_enabled: bool = False,
        directory_scan_enabled: bool = False,
        screenshot_enabled: bool = False,
        ports: Optional[List[int]] = None,
        wordlist: Optional[str] = None,
        output_dir: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize scan configuration.
        
        Args:
            id: Optional scan ID
            target: Target to scan
            network_scan_enabled: Enable network scanning
            port_scan_enabled: Enable port scanning
            dns_scan_enabled: Enable DNS enumeration
            subdomain_scan_enabled: Enable subdomain enumeration
            directory_scan_enabled: Enable directory scanning
            screenshot_enabled: Enable screenshot capture
            ports: List of ports to scan
            wordlist: Path to wordlist file
            output_dir: Path to output directory
            metadata: Additional metadata
        """
        self.id = id
        self.target = target
        self.network_scan_enabled = network_scan_enabled
        self.port_scan_enabled = port_scan_enabled
        self.dns_scan_enabled = dns_scan_enabled
        self.subdomain_scan_enabled = subdomain_scan_enabled
        self.directory_scan_enabled = directory_scan_enabled
        self.screenshot_enabled = screenshot_enabled
        self.ports = ports or []
        self.wordlist = wordlist
        self.output_dir = output_dir
        self.metadata = metadata or {}

class ReconResult:
    """Result from a reconnaissance operation."""
    def __init__(
        self,
        scan_type: str,
        domain: str,
        project_id: Optional[int],
        status: str,
        results: Dict[str, Any],
        error_message: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
        port: Optional[int] = None,
        service_info: Optional[str] = None,
        response_data: Optional[str] = None,
    ):
        self.scan_type = scan_type
        self.domain = domain
        self.project_id = project_id
        self.status = status
        self.results = results
        self.error_message = error_message
        self.start_time = start_time or datetime.utcnow()
        self.end_time = end_time or datetime.utcnow()
        self.metadata = metadata or {}
        self.path = path
        self.port = port
        self.service_info = service_info
        self.response_data = response_data

class ScanResult:
    """Container for scan-specific results."""
    def __init__(
        self,
        domains: Optional[List[str]] = None,
        open_ports: Optional[List[int]] = None,
        findings: Optional[List[Dict[str, Any]]] = None,
        service_info: Optional[Dict[str, str]] = None,
        response_data: Optional[Dict[str, Any]] = None
    ):
        self.domains = domains or []
        self.open_ports = open_ports or []
        self.findings = findings or []
        self.service_info = service_info or {}
        self.response_data = response_data or {}
