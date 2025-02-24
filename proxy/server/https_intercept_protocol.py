"""HTTPS interception protocol for TLS MITM."""
import asyncio
import logging
import ssl
from typing import Optional, Dict, Any, Callable, ClassVar, TYPE_CHECKING
from uuid import uuid4
from datetime import datetime
from async_timeout import timeout as async_timeout

if TYPE_CHECKING:
    from .certificates import CertificateAuthority

from .custom_protocol import TunnelProtocol
from .tls_helper import cert_manager
from .state import proxy_state
from .handlers.http import HttpRequestHandler
from .handlers.connect import ConnectHandler

logger = logging.getLogger("proxy.core")

class HttpsInterceptProtocol(TunnelProtocol):
    """Protocol for intercepting HTTPS traffic using modular components."""

    # Class level settings
    _ca_instance: ClassVar[Optional['CertificateAuthority']] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize TLS state tracking
        self._connection_id = str(uuid4())
        self._tls_state: Dict[str, Any] = {
            "hostname": None,
            "client_hello_seen": False,
            "handshake_complete": False,
            "sni_hostname": None,
            "alpn_protocols": None,
            "cipher": None,
            "version": None
        }
        
        # Initialize protocol state
        self._intercept_enabled = bool(self._ca_instance)
        self._tunnel_established = False
        self._buffer = bytearray()
        self._client_hello_buffer = bytearray()
        
        # Initialize handlers with better timeouts
        self._connect_handler = ConnectHandler(
            connection_id=self._connection_id,
            transport=self.transport,
            connect_timeout=10,
            read_timeout=30
        )
        self._http_handler = HttpRequestHandler(self._connection_id)
        
        # Register with state tracking
        asyncio.create_task(self._register_connection())

    async def _register_connection(self) -> None:
        """Register connection with state tracking."""
        await proxy_state.add_connection(self._connection_id, {
            "type": "https",
            "status": "initializing",
            "tls_state": self._tls_state,
            "intercept_enabled": self._intercept_enabled,
            "bytes_received": 0,
            "bytes_sent": 0
        })

    async def _handle_connect(self, host: str, port: int) -> None:
        """Handle CONNECT by performing TLS MITM."""
        try:
            logger.debug(f"[{self._connection_id}] Handling CONNECT request for {host}:{port}")
            await proxy_state.update_connection(self._connection_id, "status", "connecting")
            
            # Update TLS state
            self._tls_state["hostname"] = host
            
            # Send 200 Connection Established
            # Clear buffers before sending response to ensure clean state
            self._buffer.clear()
            self._tunnel_buffer.clear()
            self._client_hello_buffer.clear()

            # Send 200 Connection Established
            response = b"HTTP/1.1 200 Connection Established\r\nConnection: keep-alive\r\n\r\n"
            try:
                async with async_timeout(5) as cm:
                    if self.transport and not self.transport.is_closing():
                        # Ensure clean write
                        if hasattr(self.transport, 'pause_reading'):
                            self.transport.pause_reading()
                        self.transport.write(response)
                        await asyncio.sleep(0.1)  # Brief pause to ensure response is sent
                        
                        # Wait for write buffer to drain
                        if hasattr(self.transport, 'get_write_buffer_size'):
                            while self.transport.get_write_buffer_size() > 0:
                                await asyncio.sleep(0.01)
                        
                        # Resume reading for TLS handshake
                        if hasattr(self.transport, 'resume_reading'):
                            self.transport.resume_reading()
            except asyncio.TimeoutError:
                logger.error(f"[{self._connection_id}] Timeout sending 200 response")
                raise

            # Set up TLS interception if enabled and CA is available
            try:
                if self._intercept_enabled and self._ca_instance:
                    # Resume reading before TLS handshake
                    if hasattr(self.transport, 'resume_reading'):
                        self.transport.resume_reading()
                    # Get SSL context for the host
                    ssl_context = cert_manager.get_context(host)
                    
                    # Create TLS connection
                    async with async_timeout(15):
                        success = await self._establish_tls_tunnel(host, port, ssl_context)
                        if not success:
                            raise RuntimeError("Failed to establish TLS tunnel")
                else:
                    # Direct tunnel without interception
                    success = await self._connect_handler.handle_connect(host, port, intercept=False)
                    if not success:
                        raise RuntimeError("Failed to establish direct tunnel")
                
                # Update state
                await proxy_state.update_connection(self._connection_id, "status", "established")
                self._tunnel_established = True
                
                # Process any buffered data
                if self._buffer:
                    if self._remote_transport and not self._remote_transport.is_closing():
                        self._remote_transport.write(bytes(self._buffer))
                    self._buffer.clear()
                
            except Exception as e:
                logger.error(f"[{self._connection_id}] {'TLS' if self._intercept_enabled else 'Direct'} tunnel setup failed: {e}")
                await proxy_state.update_connection(self._connection_id, "error", str(e))
                raise

        except Exception as e:
            logger.error(f"[{self._connection_id}] CONNECT handling failed: {e}")
            await self._cleanup(error=str(e))

    async def _establish_tls_tunnel(self, host: str, port: int, ssl_context: ssl.SSLContext) -> bool:
        """Establish TLS tunnel with server and client."""
        try:
            # Connect to remote server first
            loop = asyncio.get_event_loop()
            self._remote_transport, _ = await loop.create_connection(
                lambda: self._connect_handler,
                host=host,
                port=port,
                ssl=ssl_context,
                server_hostname=host
            )
            
            # Update remote transport settings
            if hasattr(self._remote_transport, 'set_write_buffer_limits'):
                self._remote_transport.set_write_buffer_limits(high=262144)  # 256KB
            
            # Update TLS state
            ssl_obj = self._remote_transport.get_extra_info('ssl_object')
            if ssl_obj:
                self._tls_state.update({
                    "handshake_complete": True,
                    "cipher": ssl_obj.cipher(),
                    "version": ssl_obj.version(),
                    "alpn_protocols": ssl_obj.selected_alpn_protocol()
                })
            
            # Update connection tracking
            await proxy_state.update_connection(
                self._connection_id, 
                "tls_info", 
                self._tls_state
            )
            
            return True
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] TLS tunnel establishment failed: {e}")
            return False

    def data_received(self, data: bytes) -> None:
        """Process incoming data with TLS handling."""
        try:
            # Update metrics using create_task to handle the coroutine
            asyncio.create_task(
                proxy_state.update_connection(
                    self._connection_id,
                    "bytes_received",
                    len(data)
                )
            )
            
            if not self._tunnel_established:
                # Log incoming data size for debugging
                logger.debug(f"[{self._connection_id}] Received {len(data)} bytes before tunnel established")
                
                # Only buffer data if it's a reasonable size for TLS records (max 16KB + overhead)
                if len(data) <= 16640:  # 16KB + 384 bytes overhead
                    if not self._tls_state["handshake_complete"]:
                        # Track total buffered size
                        new_buffer_size = len(self._buffer) + len(data)
                        if new_buffer_size <= 16640:
                            self._buffer.extend(data)
                            logger.debug(f"[{self._connection_id}] Buffered {len(data)} bytes, total buffer size: {new_buffer_size}")
                        else:
                            logger.warning(f"[{self._connection_id}] Buffer would exceed TLS record size, discarding data")
                            self._buffer.clear()
                return
            
            # Forward data through tunnel with size logging
            logger.debug(f"[{self._connection_id}] Forwarding {len(data)} bytes through tunnel")
            target = (
                self._connect_handler.server_transport 
                if self.transport == self._connect_handler.client_transport
                else self._connect_handler.client_transport
            )
            
            if target and not target.is_closing():
                target.write(data)
                # Handle flow control for large writes
                if len(data) > 65536:
                    asyncio.create_task(self._handle_large_write())
                    
        except Exception as e:
            logger.error(f"[{self._connection_id}] Data handling error: {e}")
            asyncio.create_task(self._cleanup(error=str(e)))

    async def _cleanup(self, error: Optional[str] = None) -> None:
        """Clean up connection resources."""
        try:
            # Update state
            await proxy_state.update_connection(self._connection_id, "status", "closing")
            if error:
                await proxy_state.update_connection(self._connection_id, "error", error)
            
            # Close handlers and transports
            self._connect_handler.close()
            self._http_handler.close()
            
            if self.transport and not self.transport.is_closing():
                self.transport.close()
            if self._remote_transport and not self._remote_transport.is_closing():
                self._remote_transport.close()
                
            # Clear buffers
            self._buffer.clear()
            self._client_hello_buffer.clear()
            
            # Final state update
            await proxy_state.update_connection(self._connection_id, "status", "closed")
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Cleanup error: {e}")
        finally:
            # Remove from state tracking after delay
            asyncio.create_task(self._delayed_state_cleanup())
    
    async def _delayed_state_cleanup(self, delay: int = 30) -> None:
        """Remove connection from state tracking after delay."""
        await asyncio.sleep(delay)
        await proxy_state.remove_connection(self._connection_id)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection loss."""
        if exc:
            logger.error(f"[{self._connection_id}] Connection lost with error: {exc}")
        asyncio.create_task(self._cleanup(error=str(exc) if exc else None))

    def pause_reading(self) -> None:
        """Pause reading when buffers are full."""
        if self.transport:
            self.transport.pause_reading()

    def resume_reading(self) -> None:
        """Resume reading when buffers drain."""
        if self.transport:
            self.transport.resume_reading()

    @classmethod
    def create_protocol_factory(cls) -> Callable[..., 'HttpsInterceptProtocol']:
        """Create a factory function for the protocol."""
        # Get CA from cert manager
        cls._ca_instance = cert_manager.ca
        
        def protocol_factory(*args, **kwargs):
            return cls(*args, **kwargs)
        return protocol_factory
