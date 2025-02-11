"""Type stubs for recon module models."""

from typing import Dict, Any, Optional, List
from datetime import datetime
import sqlalchemy.orm  # type: ignore

class ReconResult:
    """Type stub for reconnaissance results."""
    scan_type: str
    domain: str
    project_id: Optional[int]
    status: str
    results: Dict[str, Any]
    error_message: Optional[str]
    start_time: datetime
    end_time: datetime
    metadata: Dict[str, Any]
    path: Optional[str]
    port: Optional[int]
    service_info: Optional[str]
    response_data: Optional[str]

class ScanResult:
    """Type stub for scan results."""
    domains: List[str]
    open_ports: List[int]
    findings: List[Dict[str, Any]]
    service_info: Dict[str, str]
    response_data: Dict[str, Any]
