"""HTTPS interception protocol implementation."""
import asyncio
import logging
import ssl
from typing import Optional, Callable, ClassVar, Dict, Any, TYPE_CHECKING, Tuple

from async_timeout import timeout as async_timeout

if TYPE_CHECKING:
    from ..certificates import CertificateAuthority
    from ..handlers.connect_factory import ConnectConfig

from .base import BaseProxyProtocol
from .error_handler import ErrorHandler
from .buffer_manager import BufferManager
from .state_manager import StateManager
from .tls_handler import TlsHandler
from ..tls_helper import cert_manager
from ..handlers.http import HttpRequestHandler

# Set logging level to debug for more verbose output
logger = logging.getLogger("proxy.core")
logger.setLevel(logging.DEBUG)

# Also enable debug logging for SSL/TLS operations
logging.getLogger("ssl").setLevel(logging.DEBUG)

class HttpsInterceptProtocol(BaseProxyProtocol):
    """Protocol for intercepting HTTPS traffic using modular components."""

    # Class level settings
    _ca_instance: ClassVar[Optional['CertificateAuthority']] = None
    _transport_retry_attempts: int = 3
    _transport_retry_delay: float = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize managers and handlers
        self._state_manager = StateManager(self._connection_id)
        self._error_handler = ErrorHandler(self._connection_id, self.transport)
        self._buffer_manager = BufferManager(self._connection_id, self.transport)
        self._tls_handler = TlsHandler(
            self._connection_id,
            self._state_manager,
            self._error_handler
        )

        # Initialize handlers
        self._connect_handler = None  # Will be initialized in connection_made
        self._http_handler = HttpRequestHandler(self._connection_id)

        # Initialize protocol state
        self._remote_transport: Optional[asyncio.Transport] = None
        self._tunnel: Optional[asyncio.Transport] = None
        self._setup_initial_state()
        
        logger.debug(f"[{self._connection_id}] HttpsInterceptProtocol initialized")
        
    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle new connection."""
        super().connection_made(transport)
        
        # Import here to avoid circular dependency
        from ..handlers.connect_factory import create_connect_handler, ConnectConfig
        
        # Initialize ConnectHandler using factory
        config = ConnectConfig(
            connection_id=self._connection_id,
            transport=transport,
            connect_timeout=30,
            read_timeout=60
        )
        self._connect_handler = create_connect_handler(
            config,
            self._state_manager,
            self._error_handler,
            tls_handler=self._tls_handler  # Pass existing TLS handler
        )
        
        logger.debug(f"[{self._connection_id}] Connection established")

    def _setup_initial_state(self) -> None:
        """Set up initial protocol state."""
        self._state_manager.set_intercept_enabled(bool(self._ca_instance))
        logger.debug(f"[{self._connection_id}] TLS interception enabled: {bool(self._ca_instance)}")

    async def handle_request(self, request) -> None:
        """Handle HTTPS interception requests."""
        if request.method == b"CONNECT":
            try:
                target = request.target.decode()
                host, port = self._parse_authority(target)
                
                logger.debug(f"[{self._connection_id}] Handling CONNECT request for {host}:{port}")
                
                # Check if ConnectHandler is initialized
                if not self._connect_handler:
                    raise RuntimeError("Connection handler not initialized")

                try:
                    logger.debug(f"[{self._connection_id}] Establishing tunnel for {host}:{port}")
                    
                    # First establish the tunnel
                    await self._connect_handler.handle_connect(
                        protocol=self,
                        host=host,
                        port=port,
                        intercept_tls=self._state_manager.is_intercept_enabled()
                    )
                    
                    # Verify tunnel was established
                    if not self._connect_handler.server_transport:
                        raise RuntimeError("Failed to establish tunnel - no server transport")
                    
                    # Only after tunnel is established, send 200 Connection Established
                    response = (
                        b"HTTP/1.1 200 Connection Established\r\n"
                        b"Connection: keep-alive\r\n"
                        b"Proxy-Agent: AnarchyProxy\r\n\r\n"
                    )
                    self.transport.write(response)
                    logger.debug(f"[{self._connection_id}] Sent Connection Established response")
                        
                    logger.info(f"[{self._connection_id}] Tunnel established successfully")
                    await self._state_manager.update_status("established")
                    
                except Exception as e:
                    logger.error(f"[{self._connection_id}] Failed to establish tunnel: {e}")
                    # Clean up any partial connections
                    if self._connect_handler:
                        self._connect_handler.close()
                    raise
                    
            except Exception as e:
                logger.error(f"[{self._connection_id}] CONNECT handling failed: {e}")
                # Clean up resources
                if hasattr(self, '_connect_handler') and self._connect_handler:
                    self._connect_handler.close()
                if not self.transport.is_closing():
                    error_response = (
                        b"HTTP/1.1 502 Bad Gateway\r\n"
                        b"Connection: close\r\n"
                        b"Content-Length: 0\r\n\r\n"
                    )
                    try:
                        self.transport.write(error_response)
                    except Exception:
                        logger.warning(f"[{self._connection_id}] Cannot send error - transport closed")
                raise
        else:
            # Handle non-CONNECT requests
            await self._http_handler.handle_request(request)

    def _parse_authority(self, authority: str) -> Tuple[str, int]:
        """Parse host and port from authority string."""
        try:
            if ':' in authority:
                host, port = authority.rsplit(':', 1)
                return host, int(port)
            return authority, 443  # Default HTTPS port
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to parse authority '{authority}': {e}")
            raise ValueError(f"Invalid authority format: {authority}")

    async def _cleanup(self, error: Optional[str] = None) -> None:
        """Clean up connection resources."""
        try:
            logger.debug(f"[{self._connection_id}] Starting cleanup, error: {error}")
            
            # Update state
            await self._state_manager.update_status("closing", error=error)
            
            # Close handlers
            try:
                if self._connect_handler:
                    self._connect_handler.close()
                if self._http_handler:
                    self._http_handler.close()
            except Exception as e:
                logger.warning(f"[{self._connection_id}] Error during handler cleanup: {e}")
                
            # Clear buffers
            self._buffer_manager.clear_buffers()
            
            # Final state update and cleanup
            await self._state_manager.update_status("closed")
            await super()._cleanup(error=error)
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Cleanup error: {e}")

    @classmethod
    def create_protocol_factory(cls) -> Callable[..., 'HttpsInterceptProtocol']:
        """Create a factory function for the protocol."""
        if not cert_manager.ca:
            logger.error("Certificate Authority not initialized")
            raise RuntimeError("Certificate Authority not initialized")
        
        # Configure class-level CA instance
        cls._ca_instance = cert_manager.ca
        
        def protocol_factory(*args, **kwargs):
            # Create protocol instance with proper initialization
            protocol = cls(*args, **kwargs)
            protocol._state_manager.set_intercept_enabled(True)
            logger.info(f"Created HTTPS intercept protocol {protocol._connection_id}")
            return protocol
            
        return protocol_factory

    def _send_error_response(self, status_code: int, message: str) -> None:
        """Send an HTTP error response."""
        try:
            if self.transport and not self.transport.is_closing():
                status_text = {
                    400: "Bad Request",
                    403: "Forbidden", 
                    404: "Not Found",
                    500: "Internal Server Error",
                    502: "Bad Gateway",
                    503: "Service Unavailable",
                    504: "Gateway Timeout"
                }.get(status_code, "Unknown Error")

                response = (
                    f"HTTP/1.1 {status_code} {status_text}\r\n"
                    f"Content-Type: text/plain\r\n"
                    f"Content-Length: {len(message)}\r\n"
                    f"Connection: close\r\n"
                    f"\r\n"
                    f"{message}"
                ).encode()
                self.transport.write(response)
                logger.debug(f"[{self._connection_id}] Sent error response: {status_code} {message}")
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to send error response: {e}")

    def set_tunnel(self, tunnel: asyncio.Transport) -> None:
        """Set the tunnel transport for bidirectional forwarding."""
        self._tunnel = tunnel
        logger.debug(f"[{self._connection_id}] Set tunnel transport")

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection lost event with proper cleanup."""
        try:
            if self._connect_handler:
                self._connect_handler.close()
            
            if exc:
                logger.error(f"[{self._connection_id}] Connection lost with error: {exc}")
            else:
                logger.debug(f"[{self._connection_id}] Connection closed cleanly")
            
            # Clean up state
            self._state_manager.clear_state()
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error during connection cleanup: {e}")
        finally:
            super().connection_lost(exc)

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
        
        logger.debug(f"[{self._connection_id}] TunnelProtocol initialized")

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle connection established."""
        super().connection_made(transport)
        self._transport = transport
        self._write_task = asyncio.create_task(self._process_write_queue())
        logger.debug(f"[{self._connection_id}] Tunnel transport established")

    @property
    def transport(self) -> Optional[asyncio.Transport]:
        """Get the transport."""
        return self._transport

    @transport.setter 
    def transport(self, value: Optional[asyncio.Transport]) -> None:
        """Set the transport."""
        self._transport = value

    def data_received(self, data: bytes) -> None:
        """Handle received data."""
        if self._tunnel and not self._closed:
            self._tunnel.write(data)

    def _process_write_queue(self) -> None:
        """Process the write queue."""
        async def process_queue():
            while not self._closed:
                try:
                    data = await self._write_queue.get()
                    if self._transport and not self._transport.is_closing():
                        self._transport.write(data)
                except asyncio.CancelledError:
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
                
            if self._transport:
                try:
                    if not self._transport.is_closing():
                        self._transport.close()
                except Exception:
                    pass
                self._transport = None
            
            if exc:
                logger.error(f"[{self._connection_id}] Connection lost with error: {exc}")
            else:
                logger.debug(f"[{self._connection_id}] Connection closed cleanly")
            
            super().connection_lost(exc)
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error during connection cleanup: {e}")
