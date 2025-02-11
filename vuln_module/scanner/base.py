"""Base interface for vulnerability scanners."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
import asyncio
from datetime import datetime

from ..models import VulnResult, ScanConfig, PayloadResult, PayloadType

class BaseVulnScanner(ABC):
    """Abstract base class for vulnerability scanners."""

    def __init__(self, config: ScanConfig):
        """Initialize scanner with configuration."""
        self.config = config
        self._running = False
        self._current_task: Optional[asyncio.Task] = None
        self._start_time: Optional[datetime] = None
        self._stats: Dict[str, Any] = {
            "requests_sent": 0,
            "vulnerabilities_found": 0,
            "payloads_tested": 0,
            "errors": 0
        }

    @property
    def stats(self) -> Dict[str, Any]:
        """Get current scanner statistics."""
        duration = None
        if self._start_time:
            duration = (datetime.now() - self._start_time).total_seconds()
        
        return {
            **self._stats,
            "duration": duration,
            "running": self._running
        }

    @abstractmethod
    async def setup(self) -> None:
        """Prepare scanner for execution."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        pass

    @abstractmethod
    async def _scan_target(self) -> AsyncIterator[VulnResult]:
        """Perform the actual scanning."""
        pass

    async def scan(self) -> List[VulnResult]:
        """Run vulnerability scan on the target."""
        results: List[VulnResult] = []
        self._start_time = datetime.now()
        self._running = True

        try:
            await self.setup()
            async for result in self._scan_target():
                results.append(result)
                self._stats["vulnerabilities_found"] += 1
        finally:
            self._running = False
            await self.cleanup()

        return results

    def stop(self) -> None:
        """Stop the current scan."""
        self._running = False
        if self._current_task:
            self._current_task.cancel()

    @abstractmethod
    async def verify_vulnerability(self, result: VulnResult) -> bool:
        """Verify if a vulnerability is a true positive."""
        pass

    @abstractmethod
    async def test_payload(self, payload: str, payload_type: PayloadType) -> PayloadResult:
        """Test a specific payload against the target."""
        pass

    @abstractmethod
    async def get_supported_payloads(self) -> Dict[PayloadType, List[str]]:
        """Get list of supported payload types and example payloads."""
        pass

    async def rate_limit(self) -> None:
        """Apply rate limiting if configured."""
        if self.config.rate_limit:
            await asyncio.sleep(1 / self.config.rate_limit)
