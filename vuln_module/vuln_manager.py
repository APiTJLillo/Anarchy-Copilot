"""Main vulnerability management module."""

from typing import Dict, List, Any, Optional, Type
import asyncio
from datetime import datetime

from .models import (
    VulnResult,
    PayloadResult,
    PayloadType,
    ScanConfig,
    VulnSeverity
)
from .scanner import (
    BaseVulnScanner,
    NucleiScanner
)

class VulnManager:
    """Manages vulnerability discovery operations."""

    # Map of scanner names to scanner classes
    DEFAULT_SCANNERS: Dict[str, Type[BaseVulnScanner]] = {
        "nuclei": NucleiScanner,
    }

    def __init__(self, project_id: Optional[int] = None):
        """Initialize vulnerability manager."""
        self.project_id = project_id
        self.active_scans: Dict[str, BaseVulnScanner] = {}
        self._running = False
        self._current_task: Optional[asyncio.Task] = None
        self._results_cache: Dict[str, List[VulnResult]] = {}

    async def scan_target(
        self,
        target: str,
        scanner_type: str = "nuclei",
        config: Optional[Dict[str, Any]] = None
    ) -> List[VulnResult]:
        """Perform vulnerability scan on a target."""
        if scanner_type not in self.DEFAULT_SCANNERS:
            raise ValueError(f"Unsupported scanner type: {scanner_type}")

        # Create scan configuration
        scan_config = self._create_scan_config(target, config or {})

        # Initialize scanner
        scanner_class = self.DEFAULT_SCANNERS[scanner_type]
        scanner = scanner_class(scan_config)
        scan_id = f"{scanner_type}_{target}_{datetime.now().timestamp()}"
        self.active_scans[scan_id] = scanner

        try:
            # Run the scan
            results = await scanner.scan()
            
            # Process and analyze results
            verified_results = []
            for result in results:
                if await scanner.verify_vulnerability(result):
                    verified_results.append(result)

            # Cache results
            self._results_cache[target] = verified_results
            
            return verified_results

        finally:
            # Cleanup
            if scan_id in self.active_scans:
                del self.active_scans[scan_id]

    def stop_scan(self, target: str) -> None:
        """Stop active scans for a target."""
        for scan_id, scanner in list(self.active_scans.items()):
            if target in scan_id:
                scanner.stop()
                del self.active_scans[scan_id]

    async def test_payload(
        self,
        target: str,
        payload: str,
        payload_type: PayloadType,
        scanner_type: str = "nuclei"
    ) -> PayloadResult:
        """Test a specific payload against a target."""
        if scanner_type not in self.DEFAULT_SCANNERS:
            raise ValueError(f"Unsupported scanner type: {scanner_type}")

        # Create minimal configuration for payload testing
        scan_config = ScanConfig(
            target=target,
            payload_types={payload_type},
            max_depth=1,
            threads=1,
            timeout=10
        )

        # Initialize scanner just for payload testing
        scanner_class = self.DEFAULT_SCANNERS[scanner_type]
        scanner = scanner_class(scan_config)

        try:
            await scanner.setup()
            result = await scanner.test_payload(payload, payload_type)
            return result
        finally:
            await scanner.cleanup()

    def get_scan_status(self, target: str) -> Dict[str, Any]:
        """Get status of active scans for a target."""
        status = {
            "active_scans": [],
            "total_vulnerabilities": 0,
            "by_severity": {sev.name: 0 for sev in VulnSeverity}
        }

        # Collect active scan information
        for scan_id, scanner in self.active_scans.items():
            if target in scan_id:
                status["active_scans"].append({
                    "id": scan_id,
                    "scanner": scanner.__class__.__name__,
                    "stats": scanner.stats
                })

        # Add cached results statistics
        if target in self._results_cache:
            results = self._results_cache[target]
            status["total_vulnerabilities"] = len(results)
            for result in results:
                status["by_severity"][result.severity.name] += 1

        return status

    def get_supported_scanners(self) -> Dict[str, Dict[str, Any]]:
        """Get information about supported vulnerability scanners."""
        return {
            name: {
                "class": scanner_class.__name__,
                "description": scanner_class.__doc__ or "",
                "supported_payload_types": [
                    p.name for p in PayloadType
                ]
            }
            for name, scanner_class in self.DEFAULT_SCANNERS.items()
        }

    def _create_scan_config(
        self,
        target: str,
        config: Dict[str, Any]
    ) -> ScanConfig:
        """Create scan configuration from user input."""
        return ScanConfig(
            target=target,
            payload_types=set(
                PayloadType[p] for p in config.get(
                    "payload_types",
                    [pt.name for pt in PayloadType]
                )
            ),
            max_depth=config.get("max_depth", 3),
            threads=config.get("threads", 10),
            timeout=config.get("timeout", 30),
            custom_headers=config.get("headers", {}),
            cookies=config.get("cookies", {}),
            proxy=config.get("proxy"),
            verify_ssl=config.get("verify_ssl", True),
            follow_redirects=config.get("follow_redirects", True),
            scope_constraints=config.get("scope", {}),
            rate_limit=config.get("rate_limit"),
            ai_assistance=config.get("ai_assistance", True)
        )

    def register_scanner(
        self,
        name: str,
        scanner_class: Type[BaseVulnScanner]
    ) -> None:
        """Register a new scanner implementation."""
        if not issubclass(scanner_class, BaseVulnScanner):
            raise ValueError(
                f"Scanner class must inherit from BaseVulnScanner"
            )
        self.DEFAULT_SCANNERS[name] = scanner_class
