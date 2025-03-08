"""HTTPS interception protocol implementation."""
import asyncio
import logging
import ssl
from typing import Optional, Callable, ClassVar, Dict, Any, TYPE_CHECKING, Tuple
import time

from async_timeout import timeout as async_timeout

if TYPE_CHECKING:
    from ..certificates import CertificateAuthority
    from ..handlers.connect_factory import ConnectConfig

from .base import BaseProxyProtocol
from .error_handler import ErrorHandler
from .buffer_manager import BufferManager
from .state_manager import StateManager
from .tls_handler import TlsHandler
from .types import Request
from ..tls_helper import cert_manager
from ..handlers.http import HttpRequestHandler
from ...interceptors.database import DatabaseInterceptor
from database import AsyncSessionLocal
from sqlalchemy import text

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
    _database_interceptor: Optional[DatabaseInterceptor] = None

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
        self._database_interceptor = None  # Will be initialized in _setup_database_interceptor

        # Initialize protocol state
        self._remote_transport: Optional[asyncio.Transport] = None
        self._tunnel: Optional[asyncio.Transport] = None
        self._pending_data = []  # Buffer for data received before tunnel setup
        self._tunnel_established = False
        self._setup_initial_state()
        
        logger.debug(f"[{self._connection_id}] HttpsInterceptProtocol initialized")
        
    async def _setup_database_interceptor(self) -> None:
        """Initialize the database interceptor with the active session."""
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    text("SELECT * FROM proxy_sessions WHERE is_active = true ORDER BY start_time DESC LIMIT 1")
                )
                active_session = result.first()
                if active_session:
                    self._database_interceptor = DatabaseInterceptor(active_session.id)
                    logger.debug(f"[{self._connection_id}] Initialized database interceptor for session {active_session.id}")
                else:
                    logger.warning(f"[{self._connection_id}] No active session found for database interceptor")
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to setup database interceptor: {e}")

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle new connection."""
        super().connection_made(transport)
        
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
            tls_handler=self._tls_handler
        )
        
        # Setup database interceptor
        asyncio.create_task(self._setup_database_interceptor())
        logger.debug(f"[{self._connection_id}] Connection established")

    def _setup_initial_state(self) -> None:
        """Set up initial protocol state."""
        self._state_manager.set_intercept_enabled(bool(self._ca_instance))
        logger.debug(f"[{self._connection_id}] TLS interception enabled: {bool(self._ca_instance)}")

    async def handle_request(self, request: Request) -> None:
        """Handle HTTPS interception request."""
        try:
            # Convert method to string if it's bytes
            method = request.method.decode() if isinstance(request.method, bytes) else request.method
            target = request.target.decode() if isinstance(request.target, bytes) else request.target
            
            if method != "CONNECT":
                # For non-CONNECT requests, delegate to HTTP handler and database interceptor
                logger.debug(f"[{self._connection_id}] Handling non-CONNECT request: {method} {target}")
                if self._database_interceptor:
                    # Create intercepted request object
                    intercepted_request = Request(
                        method=method,
                        target=target,
                        headers=request.headers,
                        body=request.body
                    )
                    # Let database interceptor process request
                    await self._database_interceptor.intercept(intercepted_request)
                    
                # Continue with normal handling
                await self._http_handler.handle_request(Request(
                    method=method,
                    target=target,
                    headers=request.headers,
                    body=request.body
                ))
                return

            # Parse target from request
            host, port = self._parse_authority(target)
            
            # Log request details and state
            logger.debug(f"[{self._connection_id}] Handling CONNECT request for {host}:{port}")
            logger.debug(f"[{self._connection_id}] Current transport state: {self.transport is not None and not self.transport.is_closing()}")
            logger.debug(f"[{self._connection_id}] Current tunnel state: {self._tunnel is not None and not self._tunnel.is_closing() if self._tunnel else False}")
            
            # Send 200 Connection Established immediately
            response = (
                b"HTTP/1.1 200 Connection Established\r\n"
                b"Connection: keep-alive\r\n"
                b"Proxy-Agent: AnarchyProxy\r\n\r\n"
            )
            self.transport.write(response)
            logger.debug(f"[{self._connection_id}] Sent Connection Established response")
            
            # Check if ConnectHandler is initialized
            if not self._connect_handler:
                raise RuntimeError("Connection handler not initialized")
            
            # Now handle the connection with TLS interception
            try:
                logger.debug(f"[{self._connection_id}] Starting CONNECT handling with transport type: {type(self.transport)}")
                logger.debug(f"[{self._connection_id}] Transport extra info: {self.transport.get_extra_info('socket') is not None}")
                
                await self._connect_handler.handle_connect(
                    self,
                    host=host,
                    port=port,
                    intercept_tls=self._state_manager.is_intercept_enabled()
                )
                
                # Verify server transport is available
                if not self._connect_handler.server_transport:
                    raise RuntimeError("Server transport not available after connection")
                
                # Log success and transport states
                logger.debug(f"[{self._connection_id}] Connection handler completed. Transport states:")
                logger.debug(f"[{self._connection_id}] - Client transport: {self.transport is not None and not self.transport.is_closing()}")
                logger.debug(f"[{self._connection_id}] - Server transport: {self._connect_handler.server_transport is not None and not self._connect_handler.server_transport.is_closing()}")
                logger.debug(f"[{self._connection_id}] - Tunnel transport: {self._tunnel is not None and not self._tunnel.is_closing() if self._tunnel else False}")
                
                logger.info(f"[{self._connection_id}] Successfully established tunnel to {host}:{port}")
                await self._state_manager.update_status("established")
                
                # Wait for the connection to be closed
                logger.debug(f"[{self._connection_id}] Waiting for connection to complete")
                while not self.transport.is_closing():
                    await asyncio.sleep(0.1)
                    # Log periodic state checks
                    if not hasattr(self, '_last_state_check') or time.time() - self._last_state_check > 5:
                        logger.debug(
                            f"[{self._connection_id}] Connection state check - "
                            f"Client: {not self.transport.is_closing()}, "
                            f"Server: {not self._connect_handler.server_transport.is_closing() if self._connect_handler.server_transport else False}, "
                            f"Tunnel: {not self._tunnel.is_closing() if self._tunnel else False}"
                        )
                        self._last_state_check = time.time()
                logger.debug(f"[{self._connection_id}] Connection closed")
                
            except Exception as e:
                logger.error(f"[{self._connection_id}] Failed to establish tunnel: {e}", exc_info=True)
                # Don't send error response here since we already sent 200
                return
                
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error handling request: {e}", exc_info=True)
            self._send_error_response(400, str(e))

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
                if self._database_interceptor:
                    await self._database_interceptor.close()
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
        self._tunnel_established = True
        logger.debug(f"[{self._connection_id}] Set tunnel transport")
        
        # Forward any pending data
        if self._pending_data and self._tunnel and not self._tunnel.is_closing():
            logger.debug(f"[{self._connection_id}] Forwarding {len(self._pending_data)} buffered chunks after tunnel setup")
            for data in self._pending_data:
                self._tunnel.write(data)
            self._pending_data.clear()

    def data_received(self, data: bytes) -> None:
        """Schedule data handling in event loop."""
        asyncio.create_task(self._handle_data(data))
        
    async def _handle_data(self, data: bytes) -> None:
        """Handle received data with detailed logging and buffering."""
        try:
            if not self._tunnel_established:
                # Buffer data if tunnel not ready
                logger.debug(f"[{self._connection_id}] Buffering {len(data)} bytes until tunnel is established")
                self._pending_data.append(data)
                return

            if self._tunnel and not self._tunnel.is_closing():
                # First send any pending data
                if self._pending_data:
                    logger.debug(f"[{self._connection_id}] Forwarding {len(self._pending_data)} buffered chunks")
                    for buffered in self._pending_data:
                        # Pass through database interceptor if available
                        if self._database_interceptor:
                            request = Request(method="", target="", headers={}, body=buffered)
                            await self._database_interceptor.intercept(request)
                        self._tunnel.write(buffered)
                    self._pending_data.clear()
                
                # Then send current data
                logger.debug(f"[{self._connection_id}] Forwarding {len(data)} bytes to tunnel")
                # Pass through database interceptor if available
                if self._database_interceptor:
                    request = Request(method="", target="", headers={}, body=data)
                    await self._database_interceptor.intercept(request)
                self._tunnel.write(data)
                logger.debug(f"[{self._connection_id}] Data forwarded successfully")
            else:
                logger.warning(
                    f"[{self._connection_id}] Cannot forward data - "
                    f"Tunnel exists: {self._tunnel is not None}, "
                    f"Tunnel closing: {self._tunnel.is_closing() if self._tunnel else True}"
                )
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error forwarding data: {e}")

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection lost event with proper cleanup and logging."""
        try:
            if exc:
                logger.error(f"[{self._connection_id}] Connection lost with error: {exc}")
            else:
                logger.debug(f"[{self._connection_id}] Connection closed cleanly")
            
            # Log final connection state
            logger.debug(
                f"[{self._connection_id}] Final connection state - "
                f"Transport closing: {self.transport.is_closing() if self.transport else True}, "
                f"Tunnel exists: {self._tunnel is not None}, "
                f"Tunnel closing: {self._tunnel.is_closing() if self._tunnel else True}"
            )
            
            # Clean up state
            self._state_manager.clear_state()
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error during connection cleanup: {e}")
        finally:
            super().connection_lost(exc)
