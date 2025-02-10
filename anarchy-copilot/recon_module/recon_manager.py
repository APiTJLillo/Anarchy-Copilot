"""Main reconnaissance management module."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import sqlalchemy.orm  # type: ignore

from .models import ReconResult
from .common.progress import ReconProgress
from .database import ReconDatabase
from .orchestrator import ScanOrchestrator
from .tools import (
    SubdomainScanner,
    PortScanner,
    WebScanner,
    VulnerabilityScanner
)
from .rate_limiter import RateLimiter
from .scheduler import ReconScheduler

class ReconManager:
    """Coordinates and manages all reconnaissance operations."""

    def __init__(self, db: sqlalchemy.orm.Session, project_id: Optional[int] = None):
        """Initialize recon manager with project context."""
        # Initialize common components
        self.db_handler = ReconDatabase(db, project_id)
        self.progress = ReconProgress()
        self.rate_limiter = RateLimiter()

        # Initialize scanners
        self.subdomain_scanner = SubdomainScanner(self.progress, self.rate_limiter)
        self.port_scanner = PortScanner(self.progress, self.rate_limiter)
        self.web_scanner = WebScanner(self.progress, self.rate_limiter)
        self.vuln_scanner = VulnerabilityScanner(self.progress, self.rate_limiter)

        # Initialize orchestrator
        self.orchestrator = ScanOrchestrator(
            self.db_handler,
            self.progress,
            self.subdomain_scanner,
            self.port_scanner,
            self.web_scanner,
            self.vuln_scanner
        )

        # Initialize scheduler
        self.scheduler = ReconScheduler(self.orchestrator, self.db_handler)

    async def run_tool(self, tool: str, domain: str, db: sqlalchemy.orm.Session) -> List[ReconResult]:
        """Run a specific reconnaissance tool."""
        # Use the orchestrator's full_scan method for a complete scan
        if tool == "full":
            return await self.orchestrator.full_scan(domain)
        else:
            raise ValueError(f"Unknown tool: {tool}. Use 'full' for a complete scan.")

    def schedule_scan(
        self,
        domain: str,
        interval_hours: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Schedule recurring scans for a domain."""
        self.scheduler.add_schedule(
            domain=domain,
            interval_hours=interval_hours,
            enabled=True,
            metadata=metadata
        )

    def update_schedule(
        self,
        domain: str,
        interval_hours: Optional[int] = None,
        enabled: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update an existing scan schedule."""
        self.scheduler.update_schedule(
            domain=domain,
            interval_hours=interval_hours,
            enabled=enabled,
            metadata=metadata
        )

    def remove_schedule(self, domain: str) -> None:
        """Remove a scheduled scan."""
        self.scheduler.remove_schedule(domain)

    def list_schedules(self) -> List[Dict[str, Any]]:
        """List all scheduled scans."""
        return self.scheduler.list_schedules()

    async def get_scan_history(
        self,
        domain: str,
        since: Optional[datetime] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get scan history with changes for a domain."""
        return await self.db_handler.get_change_history(domain, since, limit)

    def start_scheduler(self) -> None:
        """Start the scheduler for recurring scans."""
        self.scheduler.start()

    def stop_scheduler(self) -> None:
        """Stop the scheduler."""
        self.scheduler.stop()

    def categorize_domains(self, domains: List[str]) -> Dict[str, List[str]]:
        """Categorize domains based on certain criteria."""
        categories: Dict[str, List[str]] = {"internal": [], "external": []}
        for domain in domains:
            if "internal" in domain:
                categories["internal"].append(domain)
            else:
                categories["external"].append(domain)
        return categories

    def analyze_recon_data(self, recon_results: List[ReconResult]) -> List[ReconResult]:
        """Analyze recon data to flag anomalies or high-value targets using advanced heuristics."""
        flagged_results = []
        
        # Common ports for web apps and services
        common_ports = {80, 443, 8080, 8443}
        
        # High-value ports that often host sensitive services
        sensitive_ports = {
            21: "FTP",
            22: "SSH",
            23: "Telnet",
            445: "SMB",
            1433: "MSSQL",
            1521: "Oracle",
            3306: "MySQL",
            3389: "RDP",
            5432: "PostgreSQL",
            6379: "Redis",
            9200: "Elasticsearch",
            27017: "MongoDB"
        }
        
        # Interesting subdomain patterns
        sensitive_subdomain_patterns = [
            "admin", "dev", "stage", "test", "beta", "internal",
            "jenkins", "gitlab", "api", "vpn", "mail", "remote",
            "db", "database", "monitor", "grafana", "kibana",
            "confluence", "jira", "backup", "old", "legacy"
        ]
        
        # Keywords indicating sensitive endpoints
        sensitive_endpoints = [
            "phpinfo", "admin", "console", ".git", ".env",
            "wp-admin", "wp-config", "composer.json", "package.json",
            "config", "setup", "install", "debug", "test"
        ]
        
        for result in recon_results:
            reasons = []  # Track why this result was flagged
            
            # 1. Domain Analysis
            domain_lower = result.domain.lower()
            for pattern in sensitive_subdomain_patterns:
                if pattern in domain_lower:
                    reasons.append(f"Sensitive subdomain pattern: {pattern}")
                    break
            
            # 2. Port Analysis
            if result.port:
                if result.port not in common_ports:
                    if result.port in sensitive_ports:
                        reasons.append(f"Sensitive service port: {result.port} ({sensitive_ports[result.port]})")
                    else:
                        reasons.append(f"Uncommon port: {result.port}")
            
            # 3. Service Analysis
            if result.service_info:
                service_lower = result.service_info.lower()
                version_patterns = ["beta", "dev", "snapshot", "rc", "test"]
                for pattern in version_patterns:
                    if pattern in service_lower:
                        reasons.append(f"Non-production version detected: {pattern}")
                        break
            
            # 4. Content/Response Analysis
            if result.response_data:
                response_lower = result.response_data.lower()
                error_patterns = ["exception", "error", "stack trace", "debug"]
                for pattern in error_patterns:
                    if pattern in response_lower:
                        reasons.append(f"Error/Debug information exposed: {pattern}")
                        break
            
            # 5. Path/Endpoint Analysis
            if result.path:
                path_lower = result.path.lower()
                for endpoint in sensitive_endpoints:
                    if endpoint in path_lower:
                        reasons.append(f"Sensitive endpoint detected: {endpoint}")
                        break
            
            # Add the result if any reasons were found
            if reasons:
                # Store the reasons in the result's metadata
                if not result.metadata:
                    result.metadata = {}
                result.metadata["flag_reasons"] = reasons
                result.metadata["flagged_at"] = datetime.now().isoformat()
                result.metadata["flag_severity"] = self._calculate_severity(reasons)
                flagged_results.append(result)
        
        return flagged_results
    
    def _calculate_severity(self, reasons: List[str]) -> str:
        """Calculate severity based on the types of findings."""
        # Start with a base score
        score = 0
        
        # Scoring criteria
        for reason in reasons:
            if any(s in reason.lower() for s in ["admin", "vpn", "database", "backup"]):
                score += 3  # High-value targets
            elif any(s in reason.lower() for s in ["dev", "test", "stage"]):
                score += 2  # Potential for exposed dev/test environments
            elif "uncommon port" in reason.lower():
                score += 1  # Uncommon but not necessarily critical
            elif "error" in reason.lower() or "debug" in reason.lower():
                score += 2  # Information disclosure
        
        # Convert score to severity level
        if score >= 5:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"

    def get_status(self) -> Dict[str, Any]:
        """Get the current progress of the scan."""
        return self.orchestrator.progress.get_status()
