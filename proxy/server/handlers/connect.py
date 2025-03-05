"""CONNECT method handling for HTTPS interception."""
import asyncio
import logging
import socket
import ssl
import time
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, Union, List

from ..flow_control import FlowControl
from ..state import proxy_state
from .tunnel_protocol import TunnelProtocol
from ..tls_helper import cert_manager
from .errors import (
    ProxyError, ConnectionError, TunnelError, SSLError,
    HandshakeError, TimeoutError, get_error_for_exception, format_error_response
)
from ..protocol.base_types import TlsCapableProtocol, TlsContextProvider, TlsHandlerBase
from ..protocol.tls_factory import create_tls_handler, TlsHandlerConfig
from ..tls.context_wrapper import get_server_context, get_client_context

logger = logging.getLogger("proxy.core")

class ConnectHandler:
    """Handles CONNECT requests with TLS interception."""

    def __init__(self, connection_id: str, state_manager: Any, error_handler: Any, tls_handler: Optional[TlsHandlerBase] = None):
        """Initialize CONNECT handler.
        
        Args:
            connection_id: Unique connection identifier
            state_manager: State management instance
            error_handler: Error handling instance
            tls_handler: Optional existing TLS handler instance
        """
        self._connection_id = connection_id
        self._state_manager = state_manager
        self._error_handler = error_handler
        self._remote_transport: Optional[asyncio.Transport] = None
        
        # Use provided TLS handler or create new one
        if tls_handler is not None:
            self._tls_handler = tls_handler
        else:
            # Create TLS handler with proper configuration
            config = TlsHandlerConfig(
                connection_id=connection_id,
                state_manager=state_manager,
                error_handler=error_handler,
                loop=asyncio.get_event_loop()
            )
            self._tls_handler = create_tls_handler(config)
        
        self._client_protocol: Optional[TlsCapableProtocol] = None
        self._server_protocol: Optional[TlsCapableProtocol] = None
        self._client_transport: Optional[asyncio.Transport] = None
        self._server_transport: Optional[asyncio.Transport] = None
        self._closing = False
        
        # Initialize connection state
        self._connection_start = datetime.now()
        self._bytes_sent = 0
        self._bytes_received = 0
        self._last_activity = time.time()
        
        logger.info(f"[{connection_id}] Initialized CONNECT handler")

    @property
    def server_transport(self) -> Optional[asyncio.Transport]:
        """Get server transport."""
        return self._server_transport

    async def handle_connect(self, protocol: TlsCapableProtocol,
                           host: str, port: int,
                           intercept_tls: bool = True) -> None:
        """Handle CONNECT request."""
        try:
            logger.info(f"[{self._connection_id}] Handling CONNECT request for {host}:{port}")
            
            # Store client protocol
            self._client_protocol = protocol
            
            # Connect to remote server first
            loop = asyncio.get_event_loop()
            connect_start = time.time()
            
            try:
                # Create server protocol
                self._server_protocol = TunnelProtocol(
                    connection_id=f"{self._connection_id}-server"
                )
                
                # Connect to remote server
                transport, _ = await loop.create_connection(
                    lambda: self._server_protocol,
                    host=host,
                    port=port
                )
                
                connect_time = time.time() - connect_start
                logger.info(f"[{self._connection_id}] Connected to {host}:{port} in {connect_time:.3f}s")
                
                # Store remote transport
                self._remote_transport = transport
                
                # Set up bidirectional tunnel
                self._server_protocol.set_tunnel(protocol.transport)
                protocol.set_tunnel(transport)
                
            except Exception as e:
                logger.error(f"[{self._connection_id}] Failed to connect to {host}:{port}: {e}")
                raise ConnectionError(f"Failed to connect to {host}:{port}: {str(e)}")
            
            if intercept_tls:
                # Set up TLS interception
                logger.info(f"[{self._connection_id}] Setting up TLS interception for {host}")
                await self._setup_tls_intercept(host)
            else:
                # Set up direct tunnel
                logger.info(f"[{self._connection_id}] Setting up direct tunnel for {host}")
                self._setup_direct_tunnel()
            
            # Start monitoring after everything is set up
            self._monitor_data_transfer()
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to handle CONNECT: {e}")
            self._error_handler.handle_error(e)
            self.close()
            raise

    async def _setup_tls_intercept(self, host: str) -> None:
        """Set up TLS interception."""
        try:
            logger.debug(f"[{self._connection_id}] Starting TLS interception setup for {host}")
            
            # First wrap the remote connection with TLS
            client_transport, wrapped_client = await self._tls_handler.wrap_client(
                self._server_protocol,
                server_hostname=host
            )
            
            # Now wrap the client connection
            server_transport, wrapped_server = await self._tls_handler.wrap_server(
                self._client_protocol,
                server_hostname=host
            )
            
            # Store the TLS transports
            self._client_transport = client_transport
            self._server_transport = server_transport
            
            # Set up bidirectional tunneling
            wrapped_client.set_tunnel(server_transport)
            wrapped_server.set_tunnel(client_transport)
            
            logger.info(f"[{self._connection_id}] TLS interception established")
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to setup TLS interception: {e}")
            raise

    def _setup_direct_tunnel(self) -> None:
        """Set up direct tunnel without TLS interception."""
        if not self._client_protocol or not self._server_protocol:
            raise RuntimeError("Protocols not initialized")
            
        try:
            # Set up bidirectional tunnel
            if not hasattr(self._client_protocol, '_transport'):
                raise RuntimeError("Client protocol transport not initialized")
                
            self._client_protocol.set_tunnel(self._remote_transport)
            self._server_protocol.set_tunnel(self._client_protocol._transport)
            
            logger.info(f"[{self._connection_id}] Direct tunnel established")
            
            # Start monitoring data transfer
            self._monitor_data_transfer()
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to set up direct tunnel: {e}")
            raise

    def _monitor_data_transfer(self) -> None:
        """Monitor data transfer through the tunnel."""
        async def update_stats():
            while not self._closing:
                try:
                    # Get current transfer stats
                    if self._client_transport:
                        client_stats = self._client_transport.get_extra_info('ssl_object')
                        if client_stats:
                            self._bytes_sent = client_stats.bytes_written
                            self._bytes_received = client_stats.bytes_read
                            
                    if self._server_transport:
                        server_stats = self._server_transport.get_extra_info('ssl_object')
                        if server_stats:
                            self._bytes_sent += server_stats.bytes_written
                            self._bytes_received += server_stats.bytes_read
                    
                    # Update TLS handler stats
                    self._tls_handler.update_connection_stats(
                        self._connection_id,
                        bytes_sent=self._bytes_sent,
                        bytes_received=self._bytes_received,
                        last_activity=time.time()
                    )
                    
                    # Log transfer rates periodically
                    if self._bytes_sent > 0 or self._bytes_received > 0:
                        duration = (datetime.now() - self._connection_start).total_seconds()
                        if duration > 0:
                            send_rate = self._bytes_sent / duration / 1024  # KB/s
                            recv_rate = self._bytes_received / duration / 1024  # KB/s
                            logger.debug(
                                f"[{self._connection_id}] Transfer rates - "
                                f"Upload: {send_rate:.1f} KB/s, Download: {recv_rate:.1f} KB/s"
                            )
                    
                except Exception as e:
                    logger.error(f"[{self._connection_id}] Error monitoring data transfer: {e}")
                    if "Transport is closing" in str(e):
                        break
                
                await asyncio.sleep(5)  # Update every 5 seconds
        
        asyncio.create_task(update_stats())

    def close(self) -> None:
        """Close all connections."""
        if self._closing:
            return
            
        self._closing = True
        logger.info(f"[{self._connection_id}] Closing CONNECT handler")
        
        # Log final statistics
        duration = (datetime.now() - self._connection_start).total_seconds()
        logger.info(
            f"[{self._connection_id}] Connection summary - "
            f"Duration: {duration:.1f}s, "
            f"Bytes sent: {self._bytes_sent}, "
            f"Bytes received: {self._bytes_received}"
        )
        
        # Close transports
        for transport in [self._client_transport, self._server_transport, self._remote_transport]:
            if transport and not transport.is_closing():
                transport.close()
        
        # Update TLS handler
        self._tls_handler.close_connection(self._connection_id)
