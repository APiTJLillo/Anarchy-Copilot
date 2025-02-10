"""Network scanning and service detection tools."""

import asyncio
from typing import List, Dict, Any, Tuple, cast, Optional
from .types import ServiceInfo, PortScanResult
from .base import BaseReconTool
from recon_module.common.progress import ReconProgress
from recon_module.rate_limiter import RateLimiter

class PortScanner(BaseReconTool[PortScanResult]):
    """Combined masscan and nmap port scanner."""
    
    def __init__(self, progress: ReconProgress, rate_limiter: RateLimiter):
        super().__init__(progress, rate_limiter)
        self.tool_name = "portscan"
        
    def _get_nmap_timing(self, rpm: int) -> str:
        """Get appropriate nmap timing template based on rate limit."""
        if rpm <= 100:
            return "-T2"  # Sneaky timing for very low rates
        elif rpm <= 300:
            return "-T3"  # Normal timing for moderate rates
        elif rpm <= 800:
            return "-T4"  # Aggressive timing for higher rates
        else:
            return "-T5"  # Insane timing for very high rates
        
    def estimate_scan_size(self, target: str) -> int:
        """Port scanning is resource intensive."""
        return 65535  # Full port range

    async def _parse_masscan_output(self, output: List[str]) -> List[int]:
        """Parse masscan output to get open ports."""
        open_ports = []
        for line in output:
            if "port" in line:
                try:
                    port = int(line.split("port ")[1].split("/")[0])
                    open_ports.append(port)
                except (IndexError, ValueError):
                    continue
        return open_ports

    async def _parse_nmap_output(self, output: List[str]) -> Dict[str, ServiceInfo]:
        """Parse nmap output to get service details."""
        services: Dict[str, ServiceInfo] = {}
        for line in output:
            if "/tcp" in line or "/udp" in line:
                parts = line.split()
                if len(parts) >= 3:
                    port = parts[0].split("/")[0]
                    services[port] = ServiceInfo(
                        state=parts[1],
                        service=" ".join(parts[2:])
                    )
        return services

    async def scan(self, target: str) -> Tuple[PortScanResult, float]:
        """Run port scanning using masscan and nmap with default settings."""
        return await self.scan_with_options(target)

    async def scan_with_options(self, target: str, rate: Optional[int] = None) -> Tuple[PortScanResult, float]:
        """
        Run port scanning using masscan and nmap.
        
        Args:
            target: Target IP or domain
            rate: Optional packets per second for masscan (will be capped by rate limiter settings)
        
        Returns:
            Tuple of (port scan results, scan duration)
        """
        # Get masscan rate limits
        masscan_rpm, _, _ = self.rate_limiter.get_tool_limits("masscan")
        
        # If rate is not provided or exceeds limit, use the rate limiter's value
        if rate is None or rate > masscan_rpm:
            rate = masscan_rpm

        # First use masscan for quick port discovery
        self.progress.update(10, 100, "Running masscan for port discovery...")
        
        # Before running masscan, ensure it's within rate limits
        can_scan, message = self.rate_limiter.can_scan(target, "masscan")
        if not can_scan:
            raise Exception(f"Masscan rate limit exceeded: {message}")
            
        cmd = ["masscan", target, f"-p1-65535", f"--rate={rate}"]
        masscan_results, mass_duration = await self.run_command(
            cmd, target, estimated_size=65535, 
            progress_start=10, progress_end=50,
            tool_name="masscan"  # Override tool name for proper rate limiting
        )
        
        # Parse masscan results
        open_ports = await self._parse_masscan_output(masscan_results)
        
        if not open_ports:
            result = PortScanResult(
                total_ports_scanned=65535,
                open_ports=0,
                found_ports=[],
                services={}
            )
            return result, mass_duration
        
        # Before running nmap, ensure it's within rate limits
        can_scan, message = self.rate_limiter.can_scan(target, "nmap")
        if not can_scan:
            raise Exception(f"Nmap rate limit exceeded: {message}")

        # Then use nmap for service detection on open ports
        self.progress.update(60, 100, "Running nmap service detection...")
        ports_str = ",".join(map(str, open_ports))
        
        # Use timing template based on rate limits
        nmap_rpm, _, _ = self.rate_limiter.get_tool_limits("nmap")
        timing_template = self._get_nmap_timing(nmap_rpm)
        
        cmd = ["nmap", "-sV", timing_template, "-p", ports_str, target]
        
        nmap_results, nmap_duration = await self.run_command(
            cmd, target, estimated_size=len(open_ports),
            progress_start=60, progress_end=90,
            tool_name="nmap"  # Override tool name for proper rate limiting
        )
        
        # Parse nmap results
        services = await self._parse_nmap_output(nmap_results)
        
        total_duration = mass_duration + nmap_duration
        
        result = PortScanResult(
            total_ports_scanned=65535,
            open_ports=len(open_ports),
            found_ports=open_ports,
            services=services
        )
        return result, total_duration

    async def scan_port_range(self, target: str, port_range: str) -> Tuple[PortScanResult, float]:
        """
        Scan a specific port range instead of all ports.
        
        Args:
            target: Target IP or domain
            port_range: Port range (e.g., "80,443" or "1000-2000")
        """
        # Before running nmap, ensure it's within rate limits
        can_scan, message = self.rate_limiter.can_scan(target, "nmap")
        if not can_scan:
            raise Exception(f"Nmap rate limit exceeded: {message}")

        self.progress.update(10, 100, f"Scanning port range: {port_range}")
        
        # Use timing template based on rate limits
        nmap_rpm, _, _ = self.rate_limiter.get_tool_limits("nmap")
        timing_template = self._get_nmap_timing(nmap_rpm)
        
        # Use nmap directly for smaller ranges
        cmd = ["nmap", "-sV", timing_template, "-p", port_range, target]
        results, duration = await self.run_command(
            cmd, target, estimated_size=100,  # Reasonable default
            progress_start=10, progress_end=90,
            tool_name="nmap"  # Override tool name for proper rate limiting
        )
        
        services = await self._parse_nmap_output(results)
        ports = [int(port) for port in services.keys()]
        
        result = PortScanResult(
            total_ports_scanned=len(ports),
            open_ports=len(ports),
            found_ports=ports,
            services=services
        )
        return result, duration
