"""Web scanning and analysis tools."""

import asyncio
from typing import List, Dict, Any, Tuple, cast
from .types import WebEndpoint, HttpScanResult, TechResult
from .base import BaseReconTool
from recon_module.common.progress import ReconProgress
from recon_module.rate_limiter import RateLimiter
from recon_module.common.screenshot import ScreenshotManager

class HttpxScanner(BaseReconTool[HttpScanResult]):
    """HTTP endpoint analyzer using httpx."""

    def __init__(self, progress: ReconProgress, rate_limiter: RateLimiter):
        super().__init__(progress, rate_limiter)
        self.tool_name = "httpx"
        self.screenshot_manager = ScreenshotManager()
        
    def estimate_scan_size(self, target: str) -> int:
        """HTTP scanning is moderate weight."""
        return 10  # Default estimate per target

    async def scan(self, target: str) -> Tuple[HttpScanResult, float]:
        """Run httpx scan with screenshots."""
        cmd = [
            "httpx",
            "-title",
            "-tech-detect",
            "-status-code",
            "-json",
            "-domain", target
        ]
        
        results, duration = await self.run_command(cmd, target)
        
        # Process endpoints
        endpoints: List[WebEndpoint] = []
        for line in results:
            try:
                data = self.parse_json_line(line)
                if data:
                    # Take screenshot
                    screenshot = await self.screenshot_manager.take_screenshot(data.get("url", ""))
                    
                    endpoints.append(WebEndpoint(
                        url=data.get("url", ""),
                        title=data.get("title"),
                        status_code=data.get("status-code", 0),
                        technologies=data.get("technologies", []),
                        screenshot=screenshot
                    ))
            except Exception as e:
                print(f"Error processing httpx result: {e}")
                continue

        return (HttpScanResult(
            total_endpoints=len(endpoints),
            endpoints=endpoints,
            scan_duration=duration
        ), duration)

class WebtechDetector(BaseReconTool[TechResult]):
    """Technology stack detector."""

    def __init__(self, progress: ReconProgress, rate_limiter: RateLimiter):
        super().__init__(progress, rate_limiter)
        self.tool_name = "webtech"

    async def scan(self, target: str) -> Tuple[TechResult, float]:
        """Run webtech detection."""
        cmd = ["webtech", "-u", target, "--json"]
        results, duration = await self.run_command(cmd, target)

        # Process results (usually single line JSON)
        for line in results:
            try:
                data = self.parse_json_line(line)
                if data:
                    return (TechResult(
                        technologies=data.get("tech", []),
                        headers=data.get("headers", {}),
                        cookies=data.get("cookies", [])
                    ), duration)
            except Exception as e:
                print(f"Error processing webtech result: {e}")
                continue

        # Return empty result if no valid data found
        return (TechResult(
            technologies=[],
            headers={},
            cookies=[]
        ), duration)

class WebScanner:
    """Coordinator for web scanning tools."""

    def __init__(self, progress: ReconProgress, rate_limiter: RateLimiter):
        self.progress = progress
        self.httpx = HttpxScanner(progress, rate_limiter)
        self.webtech = WebtechDetector(progress, rate_limiter)

    async def scan_endpoint(self, url: str) -> Dict[str, Any]:
        """Run all web scans against a single endpoint."""
        http_result, http_duration = await self.httpx.scan(url)
        tech_result, tech_duration = await self.webtech.scan(url)

        return {
            "http_analysis": http_result,
            "tech_detection": tech_result,
            "total_duration": http_duration + tech_duration
        }

    async def scan_endpoints(self, urls: List[str]) -> Dict[str, Any]:
        """Run web scans against multiple endpoints."""
        results = {}
        total_duration = 0.0

        for url in urls:
            try:
                result = await self.scan_endpoint(url)
                results[url] = result
                total_duration += result["total_duration"]
            except Exception as e:
                print(f"Error scanning {url}: {e}")
                results[url] = {"error": str(e)}

        return {
            "endpoint_results": results,
            "total_duration": total_duration,
            "total_scanned": len(urls),
            "successful_scans": len([r for r in results.values() if "error" not in r])
        }
