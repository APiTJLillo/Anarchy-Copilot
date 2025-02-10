"""Subdomain enumeration tool implementations."""

from typing import List, Dict, Any, Tuple, Optional
from .types import ToolResult, ScanResult
from .base import BaseReconTool
from recon_module.common.progress import ReconProgress
from recon_module.rate_limiter import RateLimiter

class AmassScanner(BaseReconTool[List[str]]):
    def __init__(self, progress: ReconProgress, rate_limiter: RateLimiter):
        super().__init__(progress, rate_limiter)
        self.tool_name = "amass"

    async def scan(self, domain: str) -> Tuple[List[str], float]:
        """Run Amass passive enumeration."""
        cmd = ["amass", "enum", "-passive", "-d", domain]
        return await self.run_command(cmd, domain)

class SubfinderScanner(BaseReconTool[List[str]]):
    def __init__(self, progress: ReconProgress, rate_limiter: RateLimiter):
        super().__init__(progress, rate_limiter)
        self.tool_name = "subfinder"

    async def scan(self, domain: str) -> Tuple[List[str], float]:
        """Run Subfinder enumeration."""
        cmd = ["subfinder", "-d", domain]
        return await self.run_command(cmd, domain)

class AssetfinderScanner(BaseReconTool[List[str]]):
    def __init__(self, progress: ReconProgress, rate_limiter: RateLimiter):
        super().__init__(progress, rate_limiter)
        self.tool_name = "assetfinder"

    async def scan(self, domain: str) -> Tuple[List[str], float]:
        """Run Assetfinder enumeration."""
        cmd = ["assetfinder", "--subs-only", domain]
        return await self.run_command(cmd, domain)

class DnsxScanner(BaseReconTool[List[str]]):
    def __init__(self, progress: ReconProgress, rate_limiter: RateLimiter):
        super().__init__(progress, rate_limiter)
        self.tool_name = "dnsx"
    
    def estimate_scan_size(self, target: str) -> int:
        """DNS operations are generally lightweight."""
        return 1

    async def scan(self, domain: str) -> Tuple[List[str], float]:
        """Run DNSx enumeration."""
        cmd = ["dnsx", "-d", domain, "-silent"]
        return await self.run_command(cmd, domain, estimated_size=1)

class SubdomainScanner:
    """Coordinator for running multiple subdomain scanning tools."""
    
    def __init__(self, progress: ReconProgress, rate_limiter: RateLimiter):
        self.progress = progress
        self.scanners = {
            "amass": AmassScanner(progress, rate_limiter),
            "subfinder": SubfinderScanner(progress, rate_limiter),
            "assetfinder": AssetfinderScanner(progress, rate_limiter),
            "dnsx": DnsxScanner(progress, rate_limiter)
        }

    async def scan(self, domain: str, tools: Optional[List[str]] = None) -> ScanResult:
        """
        Run selected subdomain enumeration tools against a target.
        
        Args:
            domain: Target domain to scan
            tools: List of tool names to use, or None for all tools
        
        Returns:
            ScanResult containing aggregated scan data
        """
        if tools is None:
            tools = list(self.scanners.keys())

        all_results: List[str] = []
        tool_results: Dict[str, ToolResult] = {}
        total_duration = 0.0

        for tool in tools:
            if tool not in self.scanners:
                continue

            scanner = self.scanners[tool]
            try:
                results, duration = await scanner.scan(domain)
                all_results.extend(results)
                tool_results[tool] = ToolResult(
                    count=len(results),
                    duration=duration,
                    results=results,
                    error=None
                )
                total_duration += duration
            except Exception as e:
                error_msg = str(e)
                print(f"Error running {tool}: {error_msg}")
                tool_results[tool] = ToolResult(
                    count=0,
                    duration=0.0,
                    results=[],
                    error=error_msg
                )

        # Remove duplicates and sort
        unique_results = sorted(list(set(all_results)))

        return ScanResult(
            domains=unique_results,
            total_found=len(unique_results),
            total_duration=total_duration,
            tool_results=tool_results
        )
