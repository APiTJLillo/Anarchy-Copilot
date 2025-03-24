"""Handles HTTPS tunneling for the proxy server."""
import asyncio
import logging
import socket
import ssl
import time
from typing import Optional, Tuple, Any, Dict, Callable, Union
from fastapi import Request
from starlette.websockets import WebSocket

from ..config import ProxyConfig
from ..interceptor import InterceptedRequest, InterceptedResponse

logger = logging.getLogger("proxy.core")

class TunnelStats:
    """Track tunnel statistics."""

    def __init__(self, tunnel_id: str):
        self.tunnel_id = tunnel_id
        self.start_time = time.time()
        self.bytes_sent = 0
        self.bytes_received = 0
        self.last_activity = time.time()

    def log(self):
        """Log current statistics."""
        duration = time.time() - self.start_time
        logger.info(f"[{self.tunnel_id}] Tunnel statistics:")
        logger.info(f"[{self.tunnel_id}] Duration: {duration:.2f}s")
        logger.info(f"[{self.tunnel_id}] Bytes sent: {self.bytes_sent}")
        logger.info(f"[{self.tunnel_id}] Bytes received: {self.bytes_received}")

class TunnelConnection:
    """Represents a tunnel connection."""
    
    def __init__(self, tunnel_id: str, client_socket: socket.socket, remote_socket: socket.socket):
        self.tunnel_id = tunnel_id
        self.client_socket = client_socket
        self.remote_socket = remote_socket
        self.client_reader: Optional[asyncio.StreamReader] = None
        self.client_writer: Optional[asyncio.StreamWriter] = None
        self.remote_reader: Optional[asyncio.StreamReader] = None
        self.remote_writer: Optional[asyncio.StreamWriter] = None
        self.stats = TunnelStats(tunnel_id)
        self._tasks = []
        self._closed = False

    async def setup(self):
        """Set up async stream handlers."""
        loop = asyncio.get_event_loop()
        
        try:
            # Set up client side with timeout
            client_reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(client_reader)
            await asyncio.wait_for(
                loop.create_connection(lambda: protocol, sock=self.client_socket),
                timeout=5.0
            )
            self.client_reader = client_reader
            
            # Set up remote side with timeout
            remote_reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(remote_reader)
            await asyncio.wait_for(
                loop.create_connection(lambda: protocol, sock=self.remote_socket),
                timeout=5.0
            )
            self.remote_reader = remote_reader
            
        except asyncio.TimeoutError as e:
            raise ConnectionError("Timeout while setting up tunnel streams") from e
        except Exception as e:
            raise ConnectionError(f"Failed to setup tunnel streams: {e}") from e

    async def start(self):
        """Start bidirectional forwarding."""
        self._tasks = [
            asyncio.create_task(self._forward_client_to_remote()),
            asyncio.create_task(self._forward_remote_to_client())
        ]
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def close(self):
        """Close the tunnel connection."""
        if self._closed:
            return
        
        self._closed = True
        
        # Cancel tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close sockets
        for sock in [self.client_socket, self.remote_socket]:
            if sock:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except:
                    pass
                try:
                    sock.close()
                except:
                    pass

        # Log final stats
        self.stats.log()

    async def _forward_client_to_remote(self):
        """Forward data from client to remote."""
        try:
            while not self._closed:
                try:
                    # Add timeout to prevent infinite blocking
                    data = await asyncio.wait_for(
                        asyncio.get_event_loop().sock_recv(self.client_socket, 8192),
                        timeout=300.0  # 5 minute timeout
                    )
                    if not data:
                        logger.debug(f"[{self.tunnel_id}] Client closed connection")
                        break
                    
                    self.stats.bytes_sent += len(data)
                    self.stats.last_activity = time.time()
                    logger.debug(f"[{self.tunnel_id}] C->R: {len(data)} bytes")
                    
                    await asyncio.wait_for(
                        asyncio.get_event_loop().sock_sendall(self.remote_socket, data),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.tunnel_id}] Timeout in client->remote forwarding")
                    break
                except (ConnectionError, BrokenPipeError) as e:
                    logger.debug(f"[{self.tunnel_id}] Connection error in client->remote: {e}")
                    break
                except Exception as e:
                    logger.error(f"[{self.tunnel_id}] Error in client->remote forwarding: {e}")
                    break
        finally:
            if not self._closed:
                await self.close()

    async def _forward_remote_to_client(self):
        """Forward data from remote to client."""
        try:
            while not self._closed:
                try:
                    # Add timeout to prevent infinite blocking
                    data = await asyncio.wait_for(
                        asyncio.get_event_loop().sock_recv(self.remote_socket, 8192),
                        timeout=300.0  # 5 minute timeout
                    )
                    if not data:
                        logger.debug(f"[{self.tunnel_id}] Remote closed connection")
                        break
                    
                    self.stats.bytes_received += len(data)
                    self.stats.last_activity = time.time()
                    logger.debug(f"[{self.tunnel_id}] R->C: {len(data)} bytes")
                    
                    await asyncio.wait_for(
                        asyncio.get_event_loop().sock_sendall(self.client_socket, data),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.tunnel_id}] Timeout in remote->client forwarding")
                    break
                except (ConnectionError, BrokenPipeError) as e:
                    logger.debug(f"[{self.tunnel_id}] Connection error in remote->client: {e}")
                    break
                except Exception as e:
                    logger.error(f"[{self.tunnel_id}] Error in remote->client forwarding: {e}")
                    break
        finally:
            if not self._closed:
                await self.close()

class TunnelManager:
    """Manages HTTPS tunneling connections."""

    def __init__(self, config: ProxyConfig):
        self.config = config
        self._active_tunnels: Dict[str, TunnelConnection] = {}

    async def create_tunnel(
        self,
        scope: Dict[str, Any],
        receive: Callable,
        send: Callable,
        websocket: Optional[WebSocket] = None
    ) -> None:
        """Create a tunnel for HTTPS/WebSocket connections."""
        headers = dict(scope.get("headers", []))
        host_header = headers.get(b"host", b"").decode()
        
        if not host_header:
            await self._send_error(send, 400, "Missing Host header")
            return

        # Parse host:port and validate
        try:
            host, port = self._parse_host_port(host_header)
            if port is None:
                await self._send_error(send, 400, "Invalid port number")
                return
        except ValueError as e:
            await self._send_error(send, 400, str(e))
            return

        # Check scope
        if not self.config.is_in_scope(host):
            await self._send_error(send, 403, f"Host {host} not in scope")
            return

        # Delegate to TunnelProtocol
        protocol = scope.get('_protocol')
        if not protocol or not isinstance(protocol, TunnelProtocol):
            logger.error(f"TunnelProtocol not available for {host}:{port}")
            await self._send_error(send, 502, "Protocol handler not available")
            return

        # Protocol will handle the rest of the tunnel setup and data forwarding
        await protocol.handle_connect(host, port)

    def _parse_host_port(self, host_header: str) -> Tuple[str, Optional[int]]:
        """Parse host and port from Host header."""
        try:
            if ':' in host_header:
                host, port_str = host_header.split(':')
                return host, int(port_str)
            return host_header, 443
        except ValueError:
            return host_header, None

    async def _send_error(self, send: Callable, status: int, message: str) -> None:
        """Send error response through ASGI interface."""
        try:
            await send({
                'type': 'http.response.start',
                'status': status,
                'headers': [(b'content-type', b'text/plain')]
            })
            await send({
                'type': 'http.response.body',
                'body': message.encode(),
                'more_body': False
            })
        except Exception as e:
            logger.error(f"Error sending error response: {e}")
