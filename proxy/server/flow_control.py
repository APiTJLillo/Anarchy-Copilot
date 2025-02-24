"""Flow control management for proxy connections."""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger("proxy.core")

class FlowControl:
    """Handle flow control for proxy connections."""
    
    def __init__(self, transport: asyncio.Transport):
        self.transport = transport
        self.read_paused = False
        self.write_paused = False
        self.buffer_size = 0
        self._high_water = 65536  # 64KB
        self._low_water = 16384   # 16KB

    def pause_reading(self) -> None:
        """Pause reading when buffer is full."""
        if not self.read_paused:
            self.read_paused = True
            if hasattr(self.transport, 'pause_reading'):
                self.transport.pause_reading()
                logger.debug("Transport reading paused")

    def resume_reading(self) -> None:
        """Resume reading when buffer drains."""
        if self.read_paused:
            self.read_paused = False
            if hasattr(self.transport, 'resume_reading'):
                self.transport.resume_reading()
                logger.debug("Transport reading resumed")

    def pause_writing(self) -> None:
        """Pause writing when downstream buffer is full."""
        if not self.write_paused:
            self.write_paused = True
            logger.debug("Writing paused")

    def resume_writing(self) -> None:
        """Resume writing when downstream buffer drains."""
        if self.write_paused:
            self.write_paused = False
            logger.debug("Writing resumed")

    def update_buffer(self, size_change: int) -> None:
        """Update buffer size and manage flow control."""
        self.buffer_size += size_change
        
        if self.buffer_size > self._high_water and not self.read_paused:
            self.pause_reading()
        elif self.buffer_size <= self._low_water and self.read_paused:
            self.resume_reading()
