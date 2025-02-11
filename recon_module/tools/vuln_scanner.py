"""Vulnerability scanning tools implementation."""

import asyncio
from typing import List, Dict, Any, Tuple, cast
from .types import VulnFinding, VulnScanResult
from .base import BaseReconTool
from recon_module.common.progress import ReconProgress
from recon_module.rate_limiter import RateLimiter

class NucleiScanner(BaseReconTool[VulnScanResult]):
    """Nuclei-based vulnerability scanner."""

    def __init__(self, progress: ReconProgress, rate_limiter: RateLimiter):
        super().__init__(progress, rate_limiter)
        self.tool_name = "nuclei"
        
    def estimate_scan_size(self, target: str) -> int:
        """Vulnerability scanning can be intensive."""
        return 50  # Default estimate per target

    async def scan(self, target: str) -> Tuple[VulnScanResult, float]:
        """Run nuclei vulnerability scan."""
        cmd = [
            "nuclei",
            "-u", target,
            "-json",
            "-severity", "low,medium,high,critical",
            "-rate-limit", "10"  # Conservative rate limiting
        ]
        
        results, duration = await self.run_command(cmd, target)
        
        findings: List[VulnFinding] = []
        for line in results:
            try:
                data = self.parse_json_line(line)
                if data:
                    findings.append(VulnFinding(
                        template=data.get("template-id", "unknown"),
                        severity=data.get("info", {}).get("severity", "unknown"),
                        name=data.get("info", {}).get("name", "Unknown Finding"),
                        matched=data.get("matched-at", ""),
                        description=data.get("info", {}).get("description")
                    ))
            except Exception as e:
                print(f"Error processing nuclei result: {e}")
                continue

        return (VulnScanResult(
            total_findings=len(findings),
            findings=findings,
            scan_duration=duration
        ), duration)

    async def scan_with_templates(
        self,
        target: str,
        templates: List[str],
        severity: List[str] = ["low", "medium", "high", "critical"]
    ) -> Tuple[VulnScanResult, float]:
        """
        Run nuclei with specific templates and severity levels.
        
        Args:
            target: Target URL or domain
            templates: List of template names or paths
            severity: List of severity levels to include
        """
        cmd = [
            "nuclei",
            "-u", target,
            "-json",
            "-severity", ",".join(severity),
            "-rate-limit", "10",
            "-t", ",".join(templates)
        ]
        
        results, duration = await self.run_command(cmd, target)
        
        findings: List[VulnFinding] = []
        for line in results:
            try:
                data = self.parse_json_line(line)
                if data:
                    findings.append(VulnFinding(
                        template=data.get("template-id", "unknown"),
                        severity=data.get("info", {}).get("severity", "unknown"),
                        name=data.get("info", {}).get("name", "Unknown Finding"),
                        matched=data.get("matched-at", ""),
                        description=data.get("info", {}).get("description")
                    ))
            except Exception as e:
                print(f"Error processing nuclei result: {e}")
                continue

        return (VulnScanResult(
            total_findings=len(findings),
            findings=findings,
            scan_duration=duration
        ), duration)

class VulnerabilityScanner:
    """Coordinator for vulnerability scanning tools."""

    def __init__(self, progress: ReconProgress, rate_limiter: RateLimiter):
        self.progress = progress
        self.nuclei = NucleiScanner(progress, rate_limiter)

    async def scan_endpoint(
        self,
        url: str,
        templates: List[str] = [],
        severity: List[str] = ["low", "medium", "high", "critical"]
    ) -> Dict[str, Any]:
        """Run vulnerability scan against a single endpoint."""
        if templates:
            result, duration = await self.nuclei.scan_with_templates(url, templates, severity)
        else:
            result, duration = await self.nuclei.scan(url)

        return {
            "url": url,
            "scan_result": result,
            "duration": duration,
            "templates_used": templates if templates else ["default"],
            "severity_levels": severity
        }

    async def scan_endpoints(
        self,
        urls: List[str],
        templates: List[str] = [],
        severity: List[str] = ["low", "medium", "high", "critical"]
    ) -> Dict[str, Any]:
        """Run vulnerability scans against multiple endpoints."""
        results = {}
        total_duration = 0.0
        total_findings = 0

        for url in urls:
            try:
                result = await self.scan_endpoint(url, templates, severity)
                results[url] = result
                total_duration += result["duration"]
                total_findings += result["scan_result"]["total_findings"]
            except Exception as e:
                print(f"Error scanning {url}: {e}")
                results[url] = {"error": str(e)}

        return {
            "endpoint_results": results,
            "total_duration": total_duration,
            "total_scanned": len(urls),
            "total_findings": total_findings,
            "successful_scans": len([r for r in results.values() if "error" not in r])
        }
