"""Web tools for recon_module."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from recon_module.common.screenshot import ScreenshotManager
from .base import ReconTool

class DirectoryScanner(ReconTool):
    """Directory scanner implementation."""

    def __init__(self, url: str, wordlist: Path, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the directory scanner.
        
        Args:
            url (str): Target URL to scan
            wordlist (Path): Path to wordlist file
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.url = url
        self.wordlist = wordlist

    async def run(self) -> List[Dict[str, Any]]:
        """Scan for directories.
        
        Returns:
            List of scan results as dictionaries
        """
        # TODO: Implement directory scanning
        return self.results

    async def validate(self) -> bool:
        """Validate scanner configuration.
        
        Returns:
            True if valid, False otherwise
        """
        return True

class ScreenshotTool(ReconTool):
    """Screenshot tool implementation."""

    def __init__(self, urls: List[str], output_dir: Path, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the screenshot tool.
        
        Args:
            urls: List of URLs to screenshot
            output_dir: Directory to save screenshots
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.urls = urls
        self.output_dir = output_dir
        self.screenshot_manager = ScreenshotManager(output_dir)

    async def run(self) -> List[Dict[str, Any]]:
        """Take screenshots of URLs.
        
        Returns:
            List of screenshot results as dictionaries
        """
        results = []
        for url in self.urls:
            filepath = await self.screenshot_manager.capture(url)
            if filepath:
                results.append({
                    "url": url,
                    "screenshot": str(filepath)
                })
        self.results = results
        return self.results

    async def validate(self) -> bool:
        """Validate tool configuration.
        
        Returns:
            True if valid, False otherwise
        """
        return self.output_dir.exists() and self.output_dir.is_dir()
