"""Type definitions for recon tools."""

from typing import List, Dict, TypedDict, Optional

# Subdomain scanning types
class ToolResult(TypedDict):
    count: int
    duration: float
    results: List[str]
    error: Optional[str]

class ScanResult(TypedDict):
    domains: List[str]
    total_found: int
    total_duration: float
    tool_results: Dict[str, ToolResult]

# Port scanning types
class ServiceInfo(TypedDict):
    state: str
    service: str

class PortScanResult(TypedDict):
    total_ports_scanned: int
    open_ports: int
    found_ports: List[int]
    services: Dict[str, ServiceInfo]

# HTTP analysis types
class WebEndpoint(TypedDict):
    url: str
    title: Optional[str]
    status_code: int
    technologies: List[str]
    screenshot: Optional[str]

class HttpScanResult(TypedDict):
    total_endpoints: int
    endpoints: List[WebEndpoint]
    scan_duration: float

# Tech detection types
class TechResult(TypedDict):
    technologies: List[str]
    headers: Dict[str, str]
    cookies: List[str]

# Vulnerability scan types
class VulnFinding(TypedDict):
    template: str
    severity: str
    name: str
    matched: str
    description: Optional[str]

class VulnScanResult(TypedDict):
    total_findings: int
    findings: List[VulnFinding]
    scan_duration: float
