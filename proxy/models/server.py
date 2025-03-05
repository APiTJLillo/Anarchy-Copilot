import asyncio
import os
import socket
from typing import Optional

from ..utils.logging import logger
from ..utils.constants import NetworkConfig, EnvVars, HTTPStatus
from .server_state import ServerState
from .ssl_context import SSLContextManager
from .connection import ProxyConnection

class ProxyServer:
    """Main proxy server implementation handling connection management and lifecycle."""
    
    def __init__(self, host: str = None, port: int = None,
                 cert_path: str = None, key_path: str = None):
        self.state = ServerState()
        self._server: Optional[asyncio.AbstractServer] = None
        self._socket: Optional[socket.socket] = None
        self.host = host or os.getenv(EnvVars.HOST_ENV_VAR, '0.0.0.0')
        self.port = int(port or os.getenv(EnvVars.PORT_ENV_VAR, str(NetworkConfig.DEFAULT_PORT)))
        self._cleanup_lock = asyncio.Lock()
        
        # Initialize SSL manager
        self.ssl_manager = SSLContextManager(
            cert_path or os.getenv(EnvVars.CERT_PATH_ENV_VAR, 'certs/ca.crt'),
            key_path or os.getenv(EnvVars.KEY_PATH_ENV_VAR, 'certs/ca.key')
        )

    async def cleanup_resources(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            # Close all active connections
            for conn in list(self.state.active_connections):
                try:
                    await conn.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up connection: {e}")
            self.state.active_connections.clear()

            # Clean up SSL resources
            try:
                self.ssl_manager.cleanup_resources()
            except Exception as e:
                logger.error(f"Error cleaning up SSL manager: {e}")

            # Clear server references
            self._server = None
            self._socket = None

            # Trigger final garbage collection
            import gc
            gc.collect()

    def _configure_socket(self, sock: socket.socket) -> socket.socket:
        """Configure socket with performance and reliability settings."""
        # Enable address/port reuse
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, 'SO_REUSEPORT'):  # Not available on all platforms
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            
        # Keepalive settings
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPIDLE, NetworkConfig.KEEPALIVE_TIME)
        sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPINTVL, NetworkConfig.KEEPALIVE_INTERVAL)
        sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPCNT, NetworkConfig.KEEPALIVE_COUNT)
        
        # Enable TCP_NODELAY (disable Nagle's algorithm)
        sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        
        # Increase socket buffer sizes
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, NetworkConfig.SOCKET_BUFFER_SIZE)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, NetworkConfig.SOCKET_BUFFER_SIZE)
        
        return sock

    def close(self) -> None:
        """Close the server and initiate cleanup."""
        # Set shutdown flag
        self.state.is_shutting_down = True

        # Close server
        if self._server:
            try:
                self._server.close()
                # Close all server sockets
                for sock in self._server.sockets:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except (OSError, socket.error):
                        pass
                    try:
                        sock.close()
                    except (OSError, socket.error):
                        pass
            except Exception as e:
                logger.error(f"Error closing server: {e}")

        # Close main socket
        if self._socket:
            try:
                try:
                    self._socket.shutdown(socket.SHUT_RDWR)
                except (OSError, socket.error):
                    pass
                self._socket.close()
            except Exception as e:
                logger.error(f"Error closing socket: {e}")

        # Set shutdown event
        self.state.shutdown_event.set()

    async def start(self) -> None:
        """Start the proxy server."""
        try:
            # Create and configure socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket = self._configure_socket(self._socket)
            self._socket.bind((self.host, self.port))

            # Start statistics monitoring
            await self.state.start_stats_monitoring()
            
            # Create server
            self._server = await asyncio.start_server(
                self.handle_client,
                sock=self._socket,
                start_serving=False,
                backlog=NetworkConfig.BACKLOG
            )
            
            addr = self._server.sockets[0].getsockname()
            logger.info(f'Starting proxy server on {addr}')
            
            # Start accepting connections
            await self._server.start_serving()
            
            # Keep server running
            async with self._server:
                await self._server.serve_forever()
                
        except Exception as e:
            logger.error(f"Error starting proxy server: {e}")
            self.close()
            await self.cleanup_resources()
            raise

    async def handle_client(self, reader: asyncio.StreamReader,
                          writer: asyncio.StreamWriter) -> None:
        """Handle incoming client connection."""
        client_addr = writer.get_extra_info('peername')
        proxy_conn = None
        
        try:
            if self.state.is_shutting_down:
                logger.info(f"Server is shutting down, rejecting connection from {client_addr}")
                writer.close()
                await writer.wait_closed()
                return

            # Create proxy connection
            proxy_conn = ProxyConnection(reader, writer, self.ssl_manager, self.state)
            self.state.active_connections.add(proxy_conn)

            try:
                # Read the initial request line
                request_line = await reader.readline()
                if not request_line:
                    logger.warning(f"Empty request from {client_addr}")
                    return

                try:
                    method, url, version = request_line.decode().strip().split(' ')
                except ValueError:
                    logger.warning(f"Invalid request line from {client_addr}: {request_line}")
                    writer.write(HTTPStatus.BAD_REQUEST[1])
                    await writer.drain()
                    return

                if method == 'CONNECT':
                    try:
                        hostname, port = url.split(':')
                        port = int(port)
                    except ValueError:
                        logger.warning(f"Invalid CONNECT url from {client_addr}: {url}")
                        writer.write(HTTPStatus.BAD_REQUEST[1])
                        await writer.drain()
                        return

                    logger.info(f"CONNECT request from {client_addr} to {hostname}:{port}")

                    try:
                        # Send 200 Connection Established
                        writer.write(HTTPStatus.OK[1])
                        await writer.drain()

                        # Update connection stats
                        self.state.stats['total_connections'] += 1
                        self.state.stats['active_connections'] = len(self.state.active_connections)

                        # Handle the connection
                        await proxy_conn.handle_connect(hostname, port)
                    except ssl.SSLError as e:
                        logger.error(f"SSL error for {hostname}:{port} from {client_addr}: {str(e)}")
                        writer.write(HTTPStatus.BAD_GATEWAY[1])
                        await writer.drain()
                    except ConnectionRefusedError:
                        logger.error(f"Connection refused to {hostname}:{port} from {client_addr}")
                        writer.write(HTTPStatus.GATEWAY_TIMEOUT[1])
                        await writer.drain()
                    except Exception as e:
                        logger.error(f"Error handling CONNECT to {hostname}:{port} from {client_addr}: {str(e)}")
                        writer.write(HTTPStatus.INTERNAL_ERROR[1])
                        await writer.drain()
                else:
                    # Handle non-CONNECT requests (not implemented)
                    logger.info(f"Got {method} request from {client_addr} to {url}")
                    writer.write(HTTPStatus.NOT_IMPLEMENTED[1])
                    await writer.drain()

            except ConnectionError as e:
                logger.debug(f"Connection error with {client_addr}: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error handling {client_addr}: {str(e)}")

        finally:
            if proxy_conn in self.state.active_connections:
                self.state.active_connections.remove(proxy_conn)
            if proxy_conn:
                await proxy_conn.cleanup()
