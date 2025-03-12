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
from ..protocol.state_manager import StateManager
from ..tls.context_wrapper import get_server_context, get_client_context
from async_timeout import timeout

logger = logging.getLogger("proxy.core")

class ConnectHandler:
    """Handles CONNECT requests with TLS interception."""

    def __init__(self, connection_id: str, state_manager=None, error_handler=None, tls_handler=None):
        """Initialize the proxy server."""
        self._connection_id = connection_id
        self._state_manager = state_manager
        self._error_handler = error_handler
        self._client_protocol = None
        self._server_protocol = None
        self._remote_transport = None
        self._client_transport = None
        self._server_transport = None
        self._tls_handler = tls_handler
        self._connect_start_time = None
        self._last_activity = time.time()
        self._monitor_task = None
        self._stall_timeout = 30.0  # 30 seconds stall detection
        self._closing = False
        self._bytes_sent = 0
        self._bytes_received = 0
        self._connection_start = datetime.now()
        
        logger.debug(f"[{self._connection_id}] ConnectHandler initialized")

    @property
    def server_transport(self):
        """Get the server transport."""
        return self._server_transport

    @server_transport.setter 
    def server_transport(self, value):
        """Set the server transport."""
        self._server_transport = value

    async def _connect_to_server(self, host: str, port: int, protocol: TunnelProtocol) -> Tuple[asyncio.Transport, asyncio.Protocol]:
        """Connect to remote server and return transport."""
        loop = asyncio.get_event_loop()
        try:
            # Create connection with retry logic and timeout
            for attempt in range(3):
                try:
                    logger.debug(f"[{self._connection_id}] Attempting connection to {host}:{port} (attempt {attempt + 1})")
                    
                    # Resolve hostname first
                    try:
                        import socket
                        ip_address = socket.gethostbyname(host)
                        logger.debug(f"[{self._connection_id}] Resolved {host} to {ip_address}")
                    except socket.gaierror as e:
                        logger.error(f"[{self._connection_id}] Failed to resolve {host}: {e}")
                        raise ConnectionError(f"Failed to resolve {host}: {e}")
                    
                    # Add timeout to connection attempt using async_timeout
                    try:
                        logger.debug(f"[{self._connection_id}] Starting connection attempt with 5s timeout")
                        async with timeout(5.0):  # 5 second timeout per attempt
                            logger.debug(f"[{self._connection_id}] Creating connection to {ip_address}:{port}")
                            transport, _ = await loop.create_connection(
                                lambda: protocol,
                                host=ip_address,  # Use resolved IP
                                port=port
                            )
                            logger.debug(f"[{self._connection_id}] Connection created successfully")
                    except asyncio.TimeoutError as e:
                        logger.error(f"[{self._connection_id}] Connection timed out: {e}")
                        raise
                    except Exception as e:
                        logger.error(f"[{self._connection_id}] Connection creation failed: {e}")
                        raise
                        
                    # Verify transport is valid
                    if not transport:
                        logger.error(f"[{self._connection_id}] Transport creation failed - transport is None")
                        raise ConnectionError("Transport creation failed - transport is None")
                    
                    # Get socket from transport for additional info
                    try:
                        sock = transport.get_extra_info('socket')
                        if sock:
                            local_addr = sock.getsockname()
                            remote_addr = sock.getpeername()
                            logger.debug(
                                f"[{self._connection_id}] Connection established - "
                                f"Local: {local_addr}, Remote: {remote_addr}"
                            )
                            
                            # Log socket options
                            logger.debug(
                                f"[{self._connection_id}] Socket options - "
                                f"Blocking: {sock.getblocking()}, "
                                f"Timeout: {sock.gettimeout()}"
                            )
                            
                            # Check socket state
                            try:
                                error_code = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                                if error_code:
                                    logger.error(f"[{self._connection_id}] Socket error code: {error_code}")
                                else:
                                    logger.debug(f"[{self._connection_id}] Socket is in good state")
                            except Exception as e:
                                logger.warning(f"[{self._connection_id}] Could not check socket error state: {e}")
                        else:
                            logger.warning(f"[{self._connection_id}] Could not get socket info from transport")
                    except Exception as e:
                        logger.warning(f"[{self._connection_id}] Error getting socket info: {e}")
                    
                    # Log transport state
                    logger.debug(
                        f"[{self._connection_id}] Transport state - "
                        f"Is closing: {transport.is_closing()}, "
                        f"Extra info: {transport.get_extra_info('socket') is not None}"
                    )
                    
                    logger.debug(f"[{self._connection_id}] Connected to {host}:{port} on attempt {attempt + 1}")
                    
                    # Log protocol state
                    logger.debug(
                        f"[{self._connection_id}] Protocol state - "
                        f"Transport set: {protocol.transport is not None}"
                    )
                    
                    return transport, protocol
                    
                except asyncio.TimeoutError:
                    logger.warning(f"[{self._connection_id}] Connection attempt {attempt + 1} timed out")
                    if attempt == 2:  # Last attempt
                        raise TimeoutError(f"Connection to {host}:{port} timed out after 3 attempts")
                        
                except (ConnectionRefusedError, OSError) as e:
                    logger.warning(f"[{self._connection_id}] Connection attempt {attempt + 1} failed: {e}")
                    if attempt == 2:  # Last attempt
                        raise ConnectionError(f"Failed to connect to {host}:{port}: {str(e)}")
                        
                # Brief delay before retry
                if attempt < 2:  # Don't delay on last attempt
                    logger.debug(f"[{self._connection_id}] Waiting 0.5s before retry")
                    await asyncio.sleep(0.5)
                    
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to connect to {host}:{port}: {e}", exc_info=True)
            # Clean up protocol if needed
            if hasattr(protocol, 'close'):
                try:
                    protocol.close()
                    logger.debug(f"[{self._connection_id}] Closed protocol after connection failure")
                except Exception as close_error:
                    logger.warning(f"[{self._connection_id}] Error closing protocol: {close_error}")
            raise ConnectionError(f"Failed to connect to {host}:{port}: {str(e)}")

    async def handle_connect(self, protocol: Any, host: str, port: int, intercept_tls: bool = True) -> None:
        """Handle CONNECT request by establishing tunnel and optionally intercepting TLS."""
        start_time = time.time()
        self._bytes_sent = 0
        self._bytes_received = 0

        try:
            logger.info(f"[{self._connection_id}] Handling CONNECT request for {host}:{port} (TLS interception: {intercept_tls})")
            
            # Create server-side tunnel protocol
            tunnel_protocol = TunnelProtocol(f"{self._connection_id}-server")
            
            # Connect to remote server
            logger.debug(f"[{self._connection_id}] Connecting to remote server {host}:{port}")
            transport, _ = await self._connect_to_server(host, port, tunnel_protocol)
            
            # Store server transport
            self.server_transport = transport
            connect_time = time.time() - start_time
            logger.info(f"[{self._connection_id}] Connected to {host}:{port} in {connect_time:.3f}s")

            # Set up initial tunnel
            logger.debug(f"[{self._connection_id}] Setting up initial tunnel")
            logger.debug(f"[{self._connection_id}] Initial tunnel state - Protocol transport: {protocol.transport is not None}")
            
            tunnel_protocol.transport = transport
            logger.debug(f"[{self._connection_id}] Set tunnel protocol transport")
            
            protocol.set_tunnel(transport)
            logger.debug(f"[{self._connection_id}] Set protocol tunnel")
            
            tunnel_protocol._tunnel = protocol.transport
            logger.debug(f"[{self._connection_id}] Set tunnel protocol tunnel")

            if intercept_tls:
                logger.info(f"[{self._connection_id}] Setting up TLS interception for {host}")
                try:
                    logger.debug(f"[{self._connection_id}] Starting TLS interception setup")
                    logger.debug(f"[{self._connection_id}] Protocol state before TLS setup - Has transport: {hasattr(protocol, 'transport')}, Transport valid: {protocol.transport is not None and not protocol.transport.is_closing() if hasattr(protocol, 'transport') else False}")
                    logger.debug(f"[{self._connection_id}] Tunnel protocol state before TLS setup - Has transport: {hasattr(tunnel_protocol, 'transport')}, Transport valid: {tunnel_protocol.transport is not None and not tunnel_protocol.transport.is_closing() if hasattr(tunnel_protocol, 'transport') else False}")
                    
                    await self._setup_tls_interception(protocol, host, tunnel_protocol)
                    logger.debug(f"[{self._connection_id}] TLS interception setup completed successfully")
                except Exception as e:
                    logger.error(f"[{self._connection_id}] Failed to setup TLS interception: {e}", exc_info=True)
                    raise

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{self._connection_id}] Failed to handle CONNECT after {duration:.3f}s: {e}", exc_info=True)
            raise
            
    async def _setup_tls_interception(self, protocol: Any, host: str, tunnel_protocol: TunnelProtocol) -> None:
        """Set up TLS interception for the connection."""
        logger.debug(f"[{self._connection_id}] Starting TLS interception setup for {host}")
        
        # Verify TLS handler is properly initialized
        if not self._tls_handler:
            logger.error(f"[{self._connection_id}] TLS handler not initialized")
            raise RuntimeError("TLS handler not initialized")
            
        # Log TLS handler details
        logger.debug(f"[{self._connection_id}] TLS handler type: {type(self._tls_handler)}")
        
        # Set up client-side TLS first
        logger.debug(f"[{self._connection_id}] Setting up client-side TLS")
        try:
            async with timeout(10.0):
                # Log initial state
                logger.debug(f"[{self._connection_id}] Initial transport state - Protocol: {type(protocol)}, Has transport: {hasattr(protocol, 'transport')}")
                
                if hasattr(protocol, 'transport') and protocol.transport:
                    sock = protocol.transport.get_extra_info('socket')
                    if sock:
                        logger.debug(
                            f"[{self._connection_id}] Socket state before TLS - "
                            f"Type: {type(sock)}, Family: {sock.family}, "
                            f"Blocking: {sock.getblocking()}, Timeout: {sock.gettimeout()}"
                        )
                
                # Get client context with detailed logging
                try:
                    logger.debug(f"[{self._connection_id}] Getting client TLS context for {host}")
                    client_context = self._tls_handler.get_context(host, is_server=False)
                    logger.debug(
                        f"[{self._connection_id}] Client context created - "
                        f"Protocol: {client_context.protocol}, "
                        f"Verify mode: {client_context.verify_mode}, "
                        f"Options: {client_context.options}"
                    )
                except Exception as e:
                    logger.error(f"[{self._connection_id}] Failed to get client context: {e}", exc_info=True)
                    raise

                # Attempt client wrap with detailed error tracking
                try:
                    logger.debug(f"[{self._connection_id}] Starting client wrap for {host}")
                    client_transport = await self._tls_handler.wrap_client(
                        protocol.transport,
                        server_hostname=host
                    )
                    logger.debug(f"[{self._connection_id}] Client wrap succeeded")
                except Exception as e:
                    logger.error(f"[{self._connection_id}] Client wrap failed: {e}", exc_info=True)
                    raise
                
                # Verify client transport
                if not client_transport:
                    raise RuntimeError("Client TLS transport creation failed")
                
                # Get SSL info
                ssl_obj = client_transport.get_extra_info('ssl_object')
                if ssl_obj:
                    logger.debug(
                        f"[{self._connection_id}] Client SSL established - "
                        f"Version: {ssl_obj.version()}, "
                        f"Cipher: {ssl_obj.cipher()}"
                    )
                    
                logger.debug(f"[{self._connection_id}] Client-side TLS setup completed")
                
        except asyncio.TimeoutError as e:
            logger.error(f"[{self._connection_id}] Client-side TLS setup timed out: {e}")
            raise
        except Exception as e:
            logger.error(f"[{self._connection_id}] Client-side SSL handshake failed: {e}", exc_info=True)
            raise

        # Set up server-side TLS
        logger.debug(f"[{self._connection_id}] Setting up server-side TLS")
        try:
            async with timeout(10.0):
                # Verify tunnel protocol transport
                if not tunnel_protocol.transport or tunnel_protocol.transport.is_closing():
                    raise RuntimeError("Server transport is not valid or is closing")
                
                # Get server context
                server_context = self._tls_handler.get_context(host, is_server=True)
                logger.debug(
                    f"[{self._connection_id}] Server TLS context - "
                    f"Protocol: {server_context.protocol}, "
                    f"Verify mode: {server_context.verify_mode}"
                )
                    
                # Attempt server-side TLS wrap with detailed error handling
                try:
                    server_transport = await self._tls_handler.wrap_server(
                        tunnel_protocol.transport,
                        server_hostname=host
                    )
                    logger.debug(f"[{self._connection_id}] Server wrap completed successfully")
                except Exception as wrap_error:
                    logger.error(
                        f"[{self._connection_id}] Server wrap failed: {wrap_error}",
                        exc_info=True
                    )
                    raise
                
                # Verify server transport
                if not server_transport:
                    raise RuntimeError("Server TLS transport creation failed")
                
                # Get SSL info
                ssl_obj = server_transport.get_extra_info('ssl_object')
                if ssl_obj:
                    logger.debug(
                        f"[{self._connection_id}] Server SSL established - "
                        f"Version: {ssl_obj.version()}, "
                        f"Cipher: {ssl_obj.cipher()}"
                    )
                    
                logger.debug(f"[{self._connection_id}] Server-side TLS setup completed")
                
        except asyncio.TimeoutError as e:
            logger.error(f"[{self._connection_id}] Server-side TLS setup timed out: {e}")
            raise
        except Exception as e:
            logger.error(f"[{self._connection_id}] Server-side SSL handshake failed: {e}", exc_info=True)
            raise

        # Update transports
        try:
            logger.debug(f"[{self._connection_id}] Updating protocol transports")
            
            # Log transport states before update
            logger.debug(
                f"[{self._connection_id}] Transport states before update - "
                f"Client closing: {client_transport.is_closing() if client_transport else True}, "
                f"Server closing: {server_transport.is_closing() if server_transport else True}"
            )
            
            protocol.transport = client_transport
            tunnel_protocol.transport = server_transport
            tunnel_protocol._tunnel = client_transport
            
            # Verify final transport state
            if not protocol.transport or protocol.transport.is_closing():
                raise RuntimeError("Final client transport is not valid")
            if not tunnel_protocol.transport or tunnel_protocol.transport.is_closing():
                raise RuntimeError("Final server transport is not valid")
                
            logger.info(f"[{self._connection_id}] TLS interception setup completed for {host}")
            await self._state_manager.update_status("tls_established")
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to update transports: {e}")
            raise RuntimeError(f"Failed to update transports: {e}")

    async def _monitor_connection(self):
        """Monitor connection for stalls."""
        while True:
            try:
                await asyncio.sleep(1.0)
                now = time.time()
                if now - self._last_activity > self._stall_timeout:
                    logger.warning(
                        f"[{self._connection_id}] Connection stall detected - "
                        f"No activity for {now - self._last_activity:.1f}s"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self._connection_id}] Monitor error: {e}")

    def _setup_direct_tunnel(self) -> None:
        """Set up direct tunnel without TLS interception."""
        if not self._client_protocol or not self._server_protocol:
            raise RuntimeError("Protocols not initialized")
            
        try:
            logger.debug(f"[{self._connection_id}] Setting up direct tunnel")
            start_time = time.time()
            
            # Set up bidirectional tunnel
            if not hasattr(self._client_protocol, '_transport'):
                raise RuntimeError("Client protocol transport not initialized")
                
            # Store the remote transport as server transport for consistency
            self._server_transport = self._remote_transport
            
            # Set up tunnel connections
            self._client_protocol.set_tunnel(self._remote_transport)
            self._server_protocol.set_tunnel(self._client_protocol._transport)
            
            # Verify server transport is set up
            if not self._server_transport:
                raise RuntimeError("Server transport not initialized in direct tunnel mode")
            
            duration = time.time() - start_time
            logger.info(
                f"[{self._connection_id}] Direct tunnel established in {duration:.3f}s"
            )
            
            # Start monitoring data transfer
            self._monitor_data_transfer()
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to set up direct tunnel: {e}")
            raise

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

    def _monitor_data_transfer(self) -> None:
        """Monitor data transfer through the tunnel."""
        async def update_stats():
            while not self._closing:
                try:
                    # Get current transfer stats from transports
                    if self._client_transport:
                        ssl_object = self._client_transport.get_extra_info('ssl_object')
                        if ssl_object:
                            # Use transport's extra info for stats
                            stats = self._client_transport.get_extra_info('socket')
                            if stats:
                                self._bytes_sent += getattr(stats, 'bytes_sent', 0)
                                self._bytes_received += getattr(stats, 'bytes_received', 0)
                            
                    if self._server_transport:
                        ssl_object = self._server_transport.get_extra_info('ssl_object')
                        if ssl_object:
                            # Use transport's extra info for stats
                            stats = self._server_transport.get_extra_info('socket')
                            if stats:
                                self._bytes_sent += getattr(stats, 'bytes_sent', 0)
                                self._bytes_received += getattr(stats, 'bytes_received', 0)
                    
                    # Update TLS handler stats
                    if self._tls_handler:
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
