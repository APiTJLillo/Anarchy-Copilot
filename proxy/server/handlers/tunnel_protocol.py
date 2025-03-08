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
        super().__init__()
        self._connection_id = connection_id
        self._transport: Optional[asyncio.Transport] = None
        self._tunnel: Optional[asyncio.Transport] = None
        self._write_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._write_task: Optional[asyncio.Task] = None
        self._closed = False
        self._bytes_received = 0
        self._bytes_sent = 0
        
        logger.debug(f"[{self._connection_id}] TunnelProtocol initialized")

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle connection established."""
        super().connection_made(transport)
        self._transport = transport
        self._write_task = asyncio.create_task(self._process_write_queue())
        
        # Log connection details
        try:
            sock = transport.get_extra_info('socket')
            if sock:
                local_addr = sock.getsockname()
                remote_addr = sock.getpeername()
                logger.debug(
                    f"[{self._connection_id}] Tunnel transport established - "
                    f"Local: {local_addr}, Remote: {remote_addr}"
                )
        except Exception as e:
            logger.warning(f"[{self._connection_id}] Could not get socket info: {e}")
            
        logger.debug(f"[{self._connection_id}] Tunnel transport established")

    @property
    def transport(self) -> Optional[asyncio.Transport]:
        """Get the transport."""
        return self._transport

    @transport.setter 
    def transport(self, value: Optional[asyncio.Transport]) -> None:
        """Set the transport."""
        self._transport = value
        if value:
            logger.debug(f"[{self._connection_id}] Transport updated")

    def data_received(self, data: bytes) -> None:
        """Handle received data."""
        if self._tunnel and not self._closed:
            try:
                self._bytes_received += len(data)
                self._tunnel.write(data)
                logger.debug(f"[{self._connection_id}] Forwarded {len(data)} bytes to tunnel")
            except Exception as e:
                logger.error(f"[{self._connection_id}] Error forwarding data to tunnel: {e}")

    async def _process_write_queue(self) -> None:
        """Process the write queue."""
        async def process_queue():
            while not self._closed:
                try:
                    data = await self._write_queue.get()
                    if self._transport and not self._transport.is_closing():
                        self._transport.write(data)
                        self._bytes_sent += len(data)
                        logger.debug(f"[{self._connection_id}] Wrote {len(data)} bytes from queue")
                except asyncio.CancelledError:
                    logger.debug(f"[{self._connection_id}] Write queue processing cancelled")
                    break
                except Exception as e:
                    logger.error(f"[{self._connection_id}] Error processing write queue: {e}")
                    break
            logger.debug(f"[{self._connection_id}] Write queue processing stopped")

        return asyncio.create_task(process_queue())

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection lost event with proper cleanup."""
        try:
            self._closed = True
            
            if self._write_task and not self._write_task.done():
                self._write_task.cancel()
                logger.debug(f"[{self._connection_id}] Cancelled write task")
                
            if self._transport:
                try:
                    if not self._transport.is_closing():
                        self._transport.close()
                        logger.debug(f"[{self._connection_id}] Closed transport")
                except Exception as e:
                    logger.warning(f"[{self._connection_id}] Error closing transport: {e}")
                self._transport = None
            
            if exc:
                logger.error(f"[{self._connection_id}] Connection lost with error: {exc}")
            else:
                logger.debug(f"[{self._connection_id}] Connection closed cleanly")
            
            # Log final statistics
            logger.info(
                f"[{self._connection_id}] Tunnel closed - "
                f"Total bytes received: {self._bytes_received}, "
                f"Total bytes sent: {self._bytes_sent}"
            )
            
            super().connection_lost(exc)
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error during connection cleanup: {e}")

    def set_tunnel(self, tunnel: asyncio.Transport) -> None:
        """Set tunnel transport."""
        self._tunnel = tunnel
        logger.debug(f"[{self._connection_id}] Tunnel transport set")
