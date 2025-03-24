"""Buffer management for TLS traffic."""
import logging
from typing import Optional

logger = logging.getLogger("proxy.core")

class BufferManager:
    """Manages buffering of TLS traffic data."""

    def __init__(self, max_size: int = 262144):  # 256KB default
        """Initialize buffer manager.
        
        Args:
            max_size: Maximum buffer size in bytes
        """
        self._buffer = bytearray()
        self._max_size = max_size

    def add_data(self, data: bytes) -> None:
        """Add data to the buffer.
        
        Args:
            data: Data to add
        """
        if len(self._buffer) + len(data) > self._max_size:
            logger.warning("Buffer would exceed max size, clearing old data")
            self._buffer.clear()
        self._buffer.extend(data)

    def get_next_chunk(self, size: Optional[int] = None) -> Optional[bytes]:
        """Get next chunk of data from buffer.
        
        Args:
            size: Size of chunk to get, or None for all data
            
        Returns:
            Chunk of data or None if buffer is empty
        """
        if not self._buffer:
            return None

        if size is None:
            data = bytes(self._buffer)
            self._buffer.clear()
            return data

        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    @property
    def size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._buffer) == 0
