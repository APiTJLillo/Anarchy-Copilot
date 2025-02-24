"""CONNECT method handling for HTTPS interception."""
import asyncio
import logging
import socket
import ssl
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, Union
from async_timeout import timeout as async_timeout

from ..flow_control import FlowControl
from ..state import proxy_state
from ..tunnel_protocol import TunnelProtocol
from ..tls_helper import cert_manager
from .errors import (
    ProxyError, ConnectionError, TunnelError, SSLError,
    HandshakeError, TimeoutError, get_error_for_exception, format_error_response
)

logger = logging.getLogger("proxy.core")

class ConnectHandler:
    """Handles CONNECT method and TLS tunnel setup."""
    
    def __init__(self, connection_id: str, transport: asyncio.Transport, 
                 connect_timeout: int = 10, read_timeout: int = 30):
        self.connection_id = connection_id
        self.transport = transport
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._tunnel_start_time: Optional[datetime] = None
        # Transfer statistics
        self._transfer_stats = {
            "bytes_sent": 0,
            "bytes_received": 0,
            "peak_send_rate": 0,
            "peak_receive_rate": 0,
            "last_activity": None,
            "transfer_start": None,
            "transfer_durations": []
        }
        
        # Initialize connection components
        self._client_transport: Optional[asyncio.Transport] = None
        self._server_transport: Optional[asyncio.Transport] = None
        self._tls_server_transport: Optional[asyncio.Transport] = None
        self._flow_control = FlowControl(transport)
        self._ssl_context: Optional[ssl.SSLContext] = None
        
        # Register with state tracking
        asyncio.create_task(self._init_connection_state())

    async def _init_connection_state(self) -> None:
        """Initialize connection state tracking."""
        await proxy_state.add_connection(self.connection_id, {
            "type": "tunnel",
            "status": "initialized",
            "connect_timeout": self._connect_timeout,
            "read_timeout": self._read_timeout,
            "bytes_sent": 0,
            "bytes_received": 0,
            "created_at": datetime.now().isoformat()
        })

    async def handle_connect(self, host: str, port: int, intercept: bool = False) -> bool:
        """Handle CONNECT request and establish tunnel."""
        try:
            # Update connection state
            await self._update_state("connecting", host=host, port=port, intercept=intercept)
            logger.debug(f"[{self.connection_id}] Establishing {'intercepted' if intercept else 'direct'} tunnel to {host}:{port}")
            
            # Create socket options
            sock_opts = {
                socket.SOL_SOCKET: [
                    (socket.SO_KEEPALIVE, 1),
                    (socket.SO_RCVBUF, 262144),  # 256KB buffer
                    (socket.SO_SNDBUF, 262144)  # 256KB buffer
                ],
                socket.IPPROTO_TCP: [
                    (socket.TCP_NODELAY, 1)
                ]
            }
            
            # Establish connection with retries
            loop = asyncio.get_event_loop()
            for attempt in range(3):
                try:
                    async with async_timeout(self._connect_timeout) as cm:
                        # Create tunnel protocol
                        tunnel_protocol = self._create_tunnel_protocol()
                        
                        # Handle TLS interception or direct connection
                        if intercept:
                            # Get TLS context for interception
                            self._ssl_context = await self._get_ssl_context(host)
                            if not self._ssl_context:
                                raise HandshakeError("Failed to create SSL context")
                                
                            # Establish TLS connection
                            self._server_transport, _ = await self._establish_tls_connection(
                                tunnel_protocol, host, port, self._ssl_context
                            )
                        else:
                            # Direct connection without TLS interception
                            self._server_transport, _ = await loop.create_connection(
                                lambda: tunnel_protocol,
                                host=host,
                                port=port
                            )
                        
                        # Configure connection
                        self._configure_connection(tunnel_protocol, sock_opts)
                        
                        # Update state and return success
                        self._tunnel_start_time = datetime.now()
                        await self._update_state("established", 
                                               is_tls=bool(self._ssl_context),
                                               transport_id=id(self._server_transport))
                        return True
                        
                except (ConnectionRefusedError, asyncio.TimeoutError) as e:
                    error = get_error_for_exception(e)
                    if attempt == 2:
                        logger.error(f"[{self.connection_id}] Connection failed after retries: {error.message}")
                        raise error
                    await asyncio.sleep(0.5 * (attempt + 1))
                    await self._update_state("retrying", attempt=attempt + 1, error=str(e))
            
            return False

        except ProxyError as e:
            logger.error(f"[{self.connection_id}] {e.code.name}: {e.message}")
            await self._send_error(e)
            return False
            
        except Exception as e:
            error = get_error_for_exception(e)
            logger.error(f"[{self.connection_id}] Unexpected error: {error.message}")
            await self._send_error(error)
            return False

    def _create_tunnel_protocol(self) -> TunnelProtocol:
        """Create tunnel protocol instance with transfer tracking."""
        protocol = TunnelProtocol(
            client_transport=self.transport,
            flow_control=self._flow_control,
            connection_id=self.connection_id,
            buffer_size=262144  # 256KB buffer
        )
        
        # Set up transfer tracking callbacks
        protocol.on_data_sent = self._on_data_sent
        protocol.on_data_received = self._on_data_received
        
        return protocol

    def _on_data_sent(self, nbytes: int) -> None:
        """Track data sent through tunnel."""
        now = datetime.now()
        stats = self._transfer_stats
        
        # Update basic stats
        stats["bytes_sent"] += nbytes
        stats["last_activity"] = now
        
        # Initialize transfer tracking if needed
        if stats["transfer_start"] is None:
            stats["transfer_start"] = now
        
        # Calculate send rate
        duration = (now - stats["transfer_start"]).total_seconds()
        if duration > 0:
            current_rate = nbytes / duration
            stats["peak_send_rate"] = max(stats["peak_send_rate"], current_rate)
            
        # Update state
        asyncio.create_task(self._update_state("transferring", 
            bytes_sent=stats["bytes_sent"],
            send_rate=current_rate if duration > 0 else 0
        ))

    def _on_data_received(self, nbytes: int) -> None:
        """Track data received through tunnel."""
        now = datetime.now()
        stats = self._transfer_stats
        
        # Update basic stats
        stats["bytes_received"] += nbytes
        stats["last_activity"] = now
        
        # Initialize transfer tracking if needed
        if stats["transfer_start"] is None:
            stats["transfer_start"] = now
            
        # Calculate receive rate
        duration = (now - stats["transfer_start"]).total_seconds()
        if duration > 0:
            current_rate = nbytes / duration
            stats["peak_receive_rate"] = max(stats["peak_receive_rate"], current_rate)
            
        # Update state
        asyncio.create_task(self._update_state("transferring",
            bytes_received=stats["bytes_received"],
            receive_rate=current_rate if duration > 0 else 0
        ))

    def _finalize_transfer_stats(self) -> Dict[str, Any]:
        """Calculate final transfer statistics."""
        stats = self._transfer_stats
        now = datetime.now()
        
        if stats["transfer_start"] and stats["last_activity"]:
            duration = (stats["last_activity"] - stats["transfer_start"]).total_seconds()
            stats["transfer_durations"].append(duration)
            
            return {
                "total_bytes_sent": stats["bytes_sent"],
                "total_bytes_received": stats["bytes_received"],
                "peak_send_rate": stats["peak_send_rate"],
                "peak_receive_rate": stats["peak_receive_rate"],
                "average_send_rate": stats["bytes_sent"] / duration if duration > 0 else 0,
                "average_receive_rate": stats["bytes_received"] / duration if duration > 0 else 0,
                "total_duration": duration,
                "idle_time": (now - stats["last_activity"]).total_seconds() if stats["last_activity"] else 0
            }
        
        return {
            "total_bytes_sent": stats["bytes_sent"],
            "total_bytes_received": stats["bytes_received"],
            "peak_send_rate": 0,
            "peak_receive_rate": 0,
            "average_send_rate": 0,
            "average_receive_rate": 0,
            "total_duration": 0,
            "idle_time": 0
        }

    async def _establish_tls_connection(self, protocol: TunnelProtocol,
                                      host: str, port: int,
                                      ssl_context: ssl.SSLContext) -> Tuple[asyncio.Transport, Any]:
        """Establish TLS connection with retry logic."""
        loop = asyncio.get_event_loop()
        last_error = None
        
        for attempt in range(3):
            try:
                async with async_timeout(self._connect_timeout) as cm:
                    transport, protocol = await loop.create_connection(
                        lambda: protocol,
                        host=host,
                        port=port,
                        ssl=ssl_context,
                        server_hostname=host,
                        ssl_handshake_timeout=10.0
                    )
                    return transport, protocol
            except (ConnectionRefusedError, asyncio.TimeoutError, ssl.SSLError) as e:
                last_error = get_error_for_exception(e)
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                break
                
        raise last_error or TunnelError("TLS connection failed")

    def _configure_connection(self, protocol: TunnelProtocol, sock_opts: Dict) -> None:
        """Configure connection parameters and socket options."""
        # Store transports
        self._client_transport = self.transport
        
        # Configure socket options
        sock = self._server_transport.get_extra_info('socket')
        if sock:
            try:
                for level, options in sock_opts.items():
                    for opt, val in options:
                        try:
                            sock.setsockopt(level, opt, val)
                        except (AttributeError, socket.error) as e:
                            logger.warning(f"[{self.connection_id}] Could not set socket option {opt}: {e}")
                            
                # Set TCP keepalive parameters if supported
                if hasattr(socket, 'TCP_KEEPIDLE'):  # Linux
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
                if hasattr(socket, 'TCP_KEEPINTVL'):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                if hasattr(socket, 'TCP_KEEPCNT'):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
                    
            except Exception as e:
                logger.warning(f"[{self.connection_id}] Error configuring socket: {e}")

    async def _get_ssl_context(self, host: str) -> Optional[ssl.SSLContext]:
        """Get SSL context for TLS interception."""
        try:
            return cert_manager.get_context(host)
        except Exception as e:
            logger.error(f"[{self.connection_id}] Failed to get SSL context: {e}")
            await self._update_state("error", error=f"SSL context creation failed: {e}")
            return None

    async def _update_state(self, status: str, **kwargs) -> None:
        """Update connection state with additional info."""
        try:
            # Get transfer stats
            stats = self._transfer_stats
            transfer_info = {
                "bytes_sent": stats["bytes_sent"],
                "bytes_received": stats["bytes_received"],
                "peak_send_rate": stats["peak_send_rate"],
                "peak_receive_rate": stats["peak_receive_rate"]
            }

            # Calculate current rates if transfer is active
            if stats["transfer_start"] and stats["last_activity"]:
                duration = (datetime.now() - stats["transfer_start"]).total_seconds()
                if duration > 0:
                    transfer_info.update({
                        "current_send_rate": stats["bytes_sent"] / duration,
                        "current_receive_rate": stats["bytes_received"] / duration,
                    })

            state_update = {
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "transfer": transfer_info,
                **kwargs
            }

            # Add tunnel info if available
            if self._server_transport:
                tunnel_info = {
                    "transport_id": id(self._server_transport),
                    "is_tls": bool(self._ssl_context),
                    "local_address": self._get_socket_info(self._server_transport, 'peername'),
                    "remote_address": self._get_socket_info(self._server_transport, 'sockname')
                }
                state_update["tunnel"] = tunnel_info

            await proxy_state.update_connection(self.connection_id, "tunnel_state", state_update)
        except Exception as e:
            logger.error(f"[{self.connection_id}] Failed to update state: {e}")

    def _get_socket_info(self, transport: asyncio.Transport, info_type: str) -> Optional[str]:
        """Get socket address information."""
        try:
            if hasattr(transport, 'get_extra_info'):
                info = transport.get_extra_info(info_type)
                if info:
                    host, port = info
                    return f"{host}:{port}"
            return None
        except Exception:
            return None

    async def _send_error(self, error: ProxyError) -> None:
        """Send error response to client."""
        if self.transport and not self.transport.is_closing():
            response = format_error_response(error)
            self.transport.write(response)
            await self._update_state("error", error=error.message, code=error.code.value)

    async def close(self) -> None:
        """Clean up resources."""
        try:
            # Close transports
            for transport in [self._server_transport, self._tls_server_transport]:
                if transport and not transport.is_closing():
                    transport.close()

            # Calculate and update final transfer stats
            final_stats = self._finalize_transfer_stats()
            await self._update_state("closed", **final_stats)
            
            # Clear references
            self._server_transport = None
            self._tls_server_transport = None
            self._ssl_context = None
            
        except Exception as e:
            logger.error(f"[{self.connection_id}] Error during cleanup: {e}")

    @property
    def client_transport(self) -> Optional[asyncio.Transport]:
        """Get the client-side transport."""
        return self._tls_server_transport or self._client_transport

    @property
    def server_transport(self) -> Optional[asyncio.Transport]:
        """Get the server-side transport."""
        return self._server_transport
