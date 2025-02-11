"""ReconManager module for recon_module."""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from sqlalchemy.orm import Session

from recon_module.database import ReconDatabase
from recon_module.models import ReconResult, ScanConfig
from recon_module.orchestrator import ScanOrchestrator

class ReconManager:
    """Manages reconnaissance operations."""

    def __init__(self, db_session: Optional[Session] = None) -> None:
        """Initialize ReconManager.
        
        Args:
            db_session: Optional database session
        """
        self.db = ReconDatabase(db_session) if db_session else None
        self.orchestrator = ScanOrchestrator()

    async def run_scan(self, config: ScanConfig) -> List[Dict]:
        """Run scan based on configuration.
        
        Args:
            config: Scan configuration
            
        Returns:
            List of scan results
        """
        results = []

        # Run network scans if enabled
        if config.network_scan_enabled:
            network_results = await self.orchestrator.run_network_scan(config.target)
            results.extend(network_results)
            
            if config.port_scan_enabled and network_results:
                for result in network_results:
                    if result.get("host"):
                        port_results = await self.orchestrator.run_port_scan(
                            result["host"], 
                            config.ports or []
                        )
                        results.extend(port_results)

        # Run DNS enumeration if enabled
        if config.dns_scan_enabled:
            dns_results = await self.orchestrator.run_dns_enumeration(config.target)
            results.extend(dns_results)

        # Run subdomain enumeration if enabled
        if config.subdomain_scan_enabled and config.wordlist:
            subdomain_results = await self.orchestrator.run_subdomain_enumeration(
                config.target,
                Path(config.wordlist)
            )
            results.extend(subdomain_results)

        # Run directory scans if enabled
        if config.directory_scan_enabled and config.wordlist:
            dir_results = await self.orchestrator.run_directory_scan(
                config.target,
                Path(config.wordlist)
            )
            results.extend(dir_results)

        # Take screenshots if enabled
        if config.screenshot_enabled and config.output_dir:
            urls = [result["url"] for result in results if "url" in result]
            if urls:
                screenshot_results = await self.orchestrator.take_screenshots(
                    urls,
                    Path(config.output_dir)
                )
                results.extend(screenshot_results)

        # Save results to database if available
        if self.db:
            for result in results:
                scan_result = ReconResult(
                    config_id=config.id,
                    tool=result.get("tool", "unknown"),
                    data=result
                )
                self.db.add(scan_result)
            self.db.commit()

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get the current progress of the scan."""
        return self.orchestrator.progress.get_status()
