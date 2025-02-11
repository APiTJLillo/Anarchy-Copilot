"""Base vulnerability scanner interface."""

from abc import ABC, abstractmethod
from typing import List

from ..models import VulnResult, ScanConfig

class VulnScanner(ABC):
    """Base class for vulnerability scanners."""

    def __init__(self, config: ScanConfig):
        """Initialize scanner with configuration."""
        self.config = config
        
    @abstractmethod
    async def scan(self) -> List[VulnResult]:
        """Perform vulnerability scan and return results.
        
        Returns:
            List[VulnResult]: List of vulnerability findings
        """
        pass

    @abstractmethod
    async def verify(self, vuln: VulnResult) -> bool:
        """Verify if a vulnerability finding is a true positive.
        
        Args:
            vuln: The vulnerability finding to verify
            
        Returns:
            bool: True if vulnerability is confirmed, False otherwise
        """
        pass
        
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup any resources used by the scanner."""
        pass
