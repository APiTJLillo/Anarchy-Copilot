"""Base protocol for proxy implementations."""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger("proxy.core")

class BaseProxyProtocol(asyncio.Protocol):
    """Base protocol class for proxy implementations."""

    def __init__(self, connection_id: str):
        """Initialize base protocol."""
        self._connection_id = connection_id
        self.transport = None
        self._paused = False
        self._closed = False
        self._buffer = bytearray()
        self._write_buffer = bytearray()
        self._write_paused = False

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle new connection."""
        self.transport = transport
        logger.debug(f"[{self._connection_id}] Connection established")

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection loss."""
        self._closed = True
        if exc:
            logger.error(f"[{self._connection_id}] Connection lost with error: {exc}")
        else:
            logger.debug(f"[{self._connection_id}] Connection closed")

    def pause_writing(self) -> None:
        """Handle write buffer full."""
        self._write_paused = True
        logger.debug(f"[{self._connection_id}] Writing paused")

    def resume_writing(self) -> None:
        """Handle write buffer ready."""
        self._write_paused = False
        logger.debug(f"[{self._connection_id}] Writing resumed")
        # Try to write any buffered data
        if self._write_buffer and not self._closed:
            self.transport.write(bytes(self._write_buffer))
            self._write_buffer.clear()

    def write(self, data: bytes) -> None:
        """Write data to transport with buffering."""
        if self._closed:
            return
            
        if self._write_paused:
            self._write_buffer.extend(data)
        else:
            if self._write_buffer:
                # Write buffered data first
                self._write_buffer.extend(data)
                data = bytes(self._write_buffer)
                self._write_buffer.clear()
            self.transport.write(data)

    def close(self) -> None:
        """Close the protocol."""
        if not self._closed and self.transport:
            self.transport.close()
            self._closed = True

    def data_received(self, data: bytes) -> None:
        """Handle received data."""
        if not self._closed:
            asyncio.create_task(self._handle_data(data))

    async def _handle_data(self, data: bytes) -> None:
        """Handle received data asynchronously."""
        try:
            await self.handle_data(data)
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error handling data: {e}")
            if self.transport and not self.transport.is_closing():
                self.transport.close()

    async def handle_data(self, data: bytes) -> None:
        """Handle received data. Override this method in subclasses."""
        pass
