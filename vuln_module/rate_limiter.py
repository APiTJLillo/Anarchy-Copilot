"""Rate limiting utilities."""
import asyncio
from datetime import datetime

class RateLimiter:
    """Manages rate limiting for operations."""

    def __init__(self, rate: int):
        """Initialize rate limiter.
        
        Args:
            rate: Maximum operations per second
        """
        self.rate = rate
        self.interval = 1.0 / rate if rate > 0 else 0
        self._last_time = datetime.now().timestamp()

    async def acquire(self) -> None:
        """Wait until operation is allowed by rate limit."""
        if self.rate <= 0:
            return

        now = datetime.now().timestamp()
        elapsed = now - self._last_time
        if elapsed < self.interval:
            await asyncio.sleep(self.interval - elapsed)

        self._last_time = datetime.now().timestamp()
