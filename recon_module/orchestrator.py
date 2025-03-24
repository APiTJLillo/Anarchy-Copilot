"""Orchestrator module for recon_module."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .tools import (
    DirectoryScanner,
    DNSEnumerator,
    NetworkScanner,
    PortScanner,
    ScreenshotTool,
    SubdomainEnumerator
)

class ScanOrchestrator:
    """Orchestrates various recon tools."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize scan orchestrator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    async def run_network_scan(self, target: str) -> List[Dict[str, Any]]:
        """Run network scan.
        
        Args:
            target: Target network/host
            
        Returns:
            List of scan results
        """
        scanner = NetworkScanner(target, self.config)
        return await scanner.run()

    async def run_port_scan(self, target: str, ports: List[int]) -> List[Dict[str, Any]]:
        """Run port scan.
        
        Args:
            target: Target host
            ports: List of ports to scan
            
        Returns:
            List of scan results
        """
        scanner = PortScanner(target, ports, self.config)
        return await scanner.run()

    async def run_dns_enumeration(self, domain: str) -> List[Dict[str, Any]]:
        """Run DNS enumeration.
        
        Args:
            domain: Domain to enumerate
            
        Returns:
            List of enumeration results
        """
        enumerator = DNSEnumerator(domain, self.config)
        return await enumerator.run()

    async def run_subdomain_enumeration(self, domain: str, wordlist: Path) -> List[Dict[str, Any]]:
        """Run subdomain enumeration.
        
        Args:
            domain: Domain to enumerate
            wordlist: Path to subdomain wordlist
            
        Returns:
            List of enumeration results
        """
        enumerator = SubdomainEnumerator(domain, wordlist, self.config)
        return await enumerator.run()

    async def run_directory_scan(self, url: str, wordlist: Path) -> List[Dict[str, Any]]:
        """Run directory scan.
        
        Args:
            url: Target URL
            wordlist: Path to directory wordlist
            
        Returns:
            List of scan results
        """
        scanner = DirectoryScanner(url, wordlist, self.config)
        return await scanner.run()

    async def take_screenshots(self, urls: List[str], output_dir: Path) -> List[Dict[str, Any]]:
        """Take screenshots of URLs.
        
        Args:
            urls: List of URLs
            output_dir: Directory to save screenshots
            
        Returns:
            List of screenshot results
        """
        tool = ScreenshotTool(urls, output_dir, self.config)
        return await tool.run()
