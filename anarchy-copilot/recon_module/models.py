"""Models for reconnaissance module results."""

from typing import Dict, Any, Optional, List
from datetime import datetime
import sqlalchemy.orm  # type: ignore

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
