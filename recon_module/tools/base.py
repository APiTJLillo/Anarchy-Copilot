"""Base classes for recon tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class ReconTool(ABC):
    """Base class for all recon tools."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the recon tool.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.results: List[Dict[str, Any]] = []

    @abstractmethod
    async def run(self) -> List[Dict[str, Any]]:
        """Run the recon tool.
        
        Returns:
            List of results as dictionaries
        """
        raise NotImplementedError

    @abstractmethod
    async def validate(self) -> bool:
        """Validate that the tool is properly configured.
        
        Returns:
            True if valid, False otherwise
        """
        raise NotImplementedError

    def get_results(self) -> List[Dict[str, Any]]:
        """Get the results from the last run.
        
        Returns:
            List of results as dictionaries
        """
        return self.results
