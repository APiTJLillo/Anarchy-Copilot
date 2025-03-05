"""TunnelProtocol implementation for CONNECT method tunneling."""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger('proxy.core')

class TunnelProtocol(asyncio.Protocol):
    """Protocol for tunneling data between client and server."""
    
    def __init__(self, connection_id: str):
        """Initialize tunnel protocol.
        
        Args:
            connection_id: Unique connection ID for logging.
        """
        self._connection_id = connection_id
        self._transport: Optional[asyncio.Transport] = None
        self._tunnel: Optional[asyncio.Transport] = None
        self._write_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._write_task: Optional[asyncio.Task] = None
        self._closed = False
        
        logger.debug(f"[{self._connection_id}] TunnelProtocol initialized")

    @property
    def transport(self) -> Optional[asyncio.Transport]:
        """Get the transport."""
        return self._transport

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle connection established."""
        self._transport = transport
        self._write_task = asyncio.create_task(self._process_write_queue())
        logger.debug(f"[{self._connection_id}] Tunnel transport established")

    def data_received(self, data: bytes) -> None:
        """Handle received data."""
        if self._closed or not self._tunnel:
            logger.warning(f"[{self._connection_id}] Received data but tunnel is not ready")
            return
            
        try:
            self._write_queue.put_nowait(data)
            logger.debug(f"[{self._connection_id}] Queued {len(data)} bytes for tunnel")
        except asyncio.QueueFull:
            logger.warning(f"[{self._connection_id}] Write queue full, pausing reads")
            self._transport.pause_reading()

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection lost."""
        logger.debug(f"[{self._connection_id}] Tunnel connection lost: {exc if exc else 'clean shutdown'}")
        self._closed = True
        if self._write_task:
            self._write_task.cancel()
        if self._tunnel and not self._tunnel.is_closing():
            self._tunnel.close()

    def set_tunnel(self, tunnel: asyncio.Transport) -> None:
        """Set tunnel transport."""
        self._tunnel = tunnel
        logger.debug(f"[{self._connection_id}] Tunnel transport set")

    async def _process_write_queue(self) -> None:
        """Process write queue."""
        try:
            while not self._closed:
                data = await self._write_queue.get()
                if self._tunnel and not self._tunnel.is_closing():
                    self._tunnel.write(data)
                    logger.debug(f"[{self._connection_id}] Forwarded {len(data)} bytes through tunnel")
                if self._transport and self._transport.is_paused_reading():
                    self._transport.resume_reading()
                    logger.debug(f"[{self._connection_id}] Resumed reading")
                self._write_queue.task_done()
                
        except asyncio.CancelledError:
            logger.debug(f"[{self._connection_id}] Write queue processing cancelled")
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error processing write queue: {e}")
            self.connection_lost(e)
