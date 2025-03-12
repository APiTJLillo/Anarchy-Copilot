"""Base protocol implementation for proxy server."""
import asyncio
import logging
from typing import Optional, Dict, Any
from uuid import uuid4

from ..custom_protocol import TunnelProtocol

logger = logging.getLogger("proxy.core")

class BaseProxyProtocol(TunnelProtocol):
    """Base class for proxy protocols with common functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._connection_id = str(uuid4())
        self._buffer = bytearray()
        self._tunnel_established = False
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._is_closing = False
        self.transport: Optional[asyncio.Transport] = None
        self._last_state_check = 0.0

    async def _register_connection(self, protocol_type: str) -> None:
        """Register connection with state tracking.
        
        Args:
            protocol_type: Type of protocol being registered
        """
        from ..state import proxy_state
        await proxy_state.add_connection(self._connection_id, {
            "type": protocol_type,
            "status": "initializing",
            "bytes_received": 0,
            "bytes_sent": 0
        })

    async def _cleanup(self, error: Optional[str] = None) -> None:
        """Clean up connection resources."""
        from ..state import proxy_state
        try:
            # Update state
            await proxy_state.update_connection(self._connection_id, "status", "closing")
            if error:
                await proxy_state.update_connection(self._connection_id, "error", error)
            
            # Close transport
            if self.transport and not self.transport.is_closing():
                self.transport.close()
                
            # Final state update
            await proxy_state.update_connection(self._connection_id, "status", "closed")
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Cleanup error: {e}")
        finally:
            # Remove from state tracking after delay
            asyncio.create_task(self._delayed_state_cleanup())
    
    async def _delayed_state_cleanup(self, delay: int = 30) -> None:
        """Remove connection from state tracking after delay."""
        from ..state import proxy_state
        await asyncio.sleep(delay)
        await proxy_state.remove_connection(self._connection_id)

    def _parse_authority(self, authority: str) -> "tuple[str, int]":
        """Parse host and port from authority string.
        
        Args:
            authority: Authority string in host:port format
            
        Returns:
            Tuple of (host, port)
            
        Raises:
            ValueError: If authority is invalid
        """
        try:
            if not authority:
                raise ValueError("Empty authority")
                
            if ":" in authority:
                host, port = authority.rsplit(":", 1)
                return host.strip("[]"), int(port)
            else:
                return authority, 443
                
        except Exception as e:
            raise ValueError(f"Invalid authority format: {authority}") from e

    def pause_reading(self) -> None:
        """Pause reading when buffers are full."""
        if self.transport:
            self.transport.pause_reading()

    def resume_reading(self) -> None:
        """Resume reading when buffers drain."""
        if self.transport:
            self.transport.resume_reading()

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection loss."""
        if exc:
            logger.error(f"[{self._connection_id}] Connection lost with error: {exc}")
        asyncio.create_task(self._cleanup(error=str(exc) if exc else None))

    def data_received(self, data: bytes) -> None:
        """Handle received data."""
        if not self._is_closing and self.transport and not self.transport.is_closing():
            try:
                # Create task for async processing
                task = asyncio.create_task(self._process_data(data))
                task.add_done_callback(self._handle_process_error)
            except Exception as e:
                logger.error(f"[{self._connection_id}] Error in base data_received: {e}")
                asyncio.create_task(self._cleanup(error=str(e)))

    def _handle_process_error(self, task: "asyncio.Task[None]") -> None:
        """Handle errors from async data processing."""
        try:
            exc = task.exception()
            if exc:
                logger.error(f"[{self._connection_id}] Error processing data: {exc}", exc_info=exc)
                asyncio.create_task(self._cleanup(str(exc)))
        except asyncio.CancelledError:
            pass

    async def _process_data(self, data: bytes) -> None:
        """Process received data asynchronously."""
        try:
            # Implement basic data processing
            if self.transport and not self.transport.is_closing():
                if hasattr(self, '_buffer'):
                    self._buffer.extend(data)
                else:
                    logger.warning(f"[{self._connection_id}] No buffer available")
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error processing data: {e}")
            await self._cleanup(error=str(e))

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle new connection."""
        self.transport = transport
        asyncio.create_task(self._register_connection("base"))
