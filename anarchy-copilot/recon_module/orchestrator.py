"""Orchestrates and coordinates scan operations."""

from typing import List, Optional

from .common.progress import ReconProgress
from .database import ReconDatabase
from .tools import (
    SubdomainScanner, PortScanner, WebScanner, VulnerabilityScanner
)
from models import ReconResult

class ScanOrchestrator:
    """Manages and coordinates the execution of multiple scan types."""

    def __init__(
        self,
        db_handler: ReconDatabase,
        progress: ReconProgress,
        subdomain_scanner: SubdomainScanner,
        port_scanner: PortScanner,
        web_scanner: WebScanner,
        vuln_scanner: VulnerabilityScanner
    ):
        """Initialize orchestrator with scanners and handlers."""
        self.db = db_handler
        self.progress = progress
        self.subdomain_scanner = subdomain_scanner
        self.port_scanner = port_scanner
        self.web_scanner = web_scanner
        self.vuln_scanner = vuln_scanner

    async def _scan_single_domain(self, target: str) -> List[ReconResult]:
        """Run all scan types on a single domain."""
        results: List[ReconResult] = []

        # Port scanning
        self.progress.update(self.progress.current, self.progress.total, f"Port scanning {target}...")
        port_result = await self.port_scanner.scan(target)
        results.append(await self.db.save_result("port_scan", target, dict(port_result[0])))

        # Web analysis
        self.progress.update(self.progress.current, self.progress.total, f"Web analysis of {target}...")
        web_result = await self.web_scanner.scan_endpoint(target)
        results.append(await self.db.save_result("web_scan", target, web_result))

        # Vulnerability scanning
        self.progress.update(self.progress.current, self.progress.total, f"Vulnerability scanning {target}...")
        vuln_result = await self.vuln_scanner.scan_endpoint(
            target, [], ["low", "medium", "high", "critical"]
        )
        results.append(await self.db.save_result("vuln_scan", target, vuln_result))

        return results

    async def full_scan(self, domain: str) -> List[ReconResult]:
        """
        Run a complete reconnaissance scan including:
        - Subdomain enumeration
        - Port scanning
        - Web analysis
        - Vulnerability scanning
        """
        results: List[ReconResult] = []
        
        # Start with subdomain enumeration
        self.progress.update(0, 100, "Starting subdomain enumeration...")
        try:
            subdomain_results = await self.subdomain_scanner.scan(domain)
            subdomain_result = await self.db.save_result(
                "subdomain_scan", domain, dict(subdomain_results)
            )
            results.append(subdomain_result)
            
            # Get discovered domains
            discovered_domains = []
            if subdomain_result.status == "completed":
                discovered_domains = subdomain_result.results.get("domains", [])
            
            # Add the original domain if no subdomains found
            if not discovered_domains:
                discovered_domains = [domain]
            
            total_steps = len(discovered_domains) * 3  # port, web, and vuln scans per domain
            current_step = 0
            
            # Scan each discovered domain
            for target in discovered_domains:
                domain_results = await self._scan_single_domain(target)
                results.extend(domain_results)
                current_step += 3  # Increment for the 3 scans completed
                self.progress.update(current_step, total_steps, f"Scanning {target} completed.")

        except Exception as e:
            # Log the error but don't re-raise to allow partial results
            await self.db.save_result("full_scan", domain, {}, str(e))
        
        self.progress.update(100, 100, "Full scan completed.")
        return results
