"""TLS tunnel establishment and certificate management."""
import asyncio
import errno
import logging
import select
import socket
import ssl
import time
import os
from typing import Optional, Tuple, Dict, Any, Union, List, cast
from contextlib import suppress
from datetime import datetime

from async_timeout import timeout

from ..tls_helper import cert_manager
from .ssl_transport import SslTransport
from ..tls.context_wrapper import get_server_context, get_client_context
from .base_types import TlsCapableProtocol, TlsContextProvider
from ..certificates import CertificateAuthority
from .connection_handler import ConnectionHandler

logger = logging.getLogger("proxy.core")

class _SslTransport(asyncio.Transport):
    """Basic SSL transport implementation for fallback mode."""
    
    def __init__(self, tls_handler: 'TlsHandler', loop: asyncio.AbstractEventLoop, ssl_sock: ssl.SSLSocket, protocol: asyncio.Protocol):
        """Initialize SSL transport with robust socket access."""
        super().__init__()
        self._tls_handler = tls_handler
        self._loop = loop
        self._ssl_sock = ssl_sock
        self._protocol = protocol
        self._closing = False
        self._write_buffer = bytearray()
        self._read_buffer = bytearray()
        self._read_paused = False
        self._write_paused = False
        self._read_task = None
        self._write_task = None
        self._extra = {}
        
        # Initialize write buffer limits
        self._high_water = 64 * 1024  # 64 KiB default high-water mark
        self._low_water = 16 * 1024   # 16 KiB default low-water mark
        
        # Get raw socket with enhanced error handling
        try:
            # First try direct socket access if it's already a raw socket
            if isinstance(ssl_sock, socket.socket) and not isinstance(ssl_sock, ssl.SSLSocket):
                self._sock = ssl_sock
            elif isinstance(ssl_sock, ssl.SSLSocket):
                # For SSLSocket, try multiple methods
                if hasattr(ssl_sock, '_socket'):
                    self._sock = ssl_sock._socket
                elif hasattr(ssl_sock, '_sock'):
                    self._sock = ssl_sock._sock
                elif hasattr(ssl_sock, 'fileno'):
                    try:
                        fileno = ssl_sock.fileno()
                        if fileno >= 0:
                            self._sock = socket.fromfd(fileno, socket.AF_INET, socket.SOCK_STREAM)
                    except (socket.error, IOError) as e:
                        logger.error(f"Failed to get socket from fileno: {e}")
                
                # Try unwrapping as last resort
                if not self._sock:
                    try:
                        self._sock = ssl_sock.unwrap()
                    except Exception as e:
                        logger.error(f"Failed to unwrap SSL socket: {e}")
                        raise
            else:
                raise RuntimeError(f"Unsupported socket type: {type(ssl_sock)}")
                
            # Set socket to non-blocking mode
            self._sock.setblocking(False)
            
            # Initialize transport info with enhanced error handling
            try:
                self._extra = {
                    'socket': self._sock,
                    'ssl_object': ssl_sock,
                    'peername': self._sock.getpeername(),
                    'socket_family': self._sock.family,
                    'socket_protocol': self._sock.proto,
                    'cipher': ssl_sock.cipher() if isinstance(ssl_sock, ssl.SSLSocket) else None,
                    'version': ssl_sock.version() if isinstance(ssl_sock, ssl.SSLSocket) else None,
                    'socket_state': 'connected' if self._sock.fileno() >= 0 else 'closed',
                    'bytes_sent': 0,
                    'bytes_received': 0
                }
            except Exception as e:
                logger.error(f"Failed to initialize transport info: {e}")
                raise RuntimeError(f"Transport initialization failed: {e}")
            
            # Configure socket for optimal performance
            try:
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32768)
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32768)
            except socket.error as e:
                logger.warning(f"Could not set socket options: {e}")
                
            # Log successful initialization with connection details
            logger.debug(f"SSL Transport initialized: version={self._extra.get('version')}, cipher={self._extra.get('cipher')}")
            self._protocol.connection_made(self)
            
            # Start reading data
            self._start_reading()
            
        except Exception as e:
            logger.error(f"Failed to initialize SSL transport: {e}")
            raise

    def _start_reading(self) -> None:
        """Start reading data from the socket."""
        if not self._read_task:
            self._read_task = self._loop.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        """Read data from the socket and pass it to the protocol."""
        BUFFER_SIZE = 32768  # 32KB buffer
        
        while not self._closing:
            try:
                # Check if socket is still valid
                if not self._ssl_sock or getattr(self._ssl_sock, '_closed', True):
                    logger.debug("SSL socket closed, stopping read loop")
                    break
                    
                # Wait for data to be available
                try:
                    await self._loop.sock_recv(self._ssl_sock, 0)  # Wait for data
                except (ConnectionError, socket.error) as e:
                    if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                        logger.error(f"Socket error while waiting for data: {e}")
                        break
                    continue
                    
                # Read and decrypt data directly from SSL socket
                try:
                    data = await self._loop.sock_recv(self._ssl_sock, BUFFER_SIZE)
                    if not data:
                        logger.debug("Connection closed by peer")
                        break
                        
                    # Let TLS handler process the decrypted data
                    await self._tls_handler.process_decrypted_data(data, self._protocol)
                    # Update stats
                    self._extra['bytes_received'] += len(data)
                            
                except (ConnectionError, socket.error) as e:
                    if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                        logger.error(f"Socket error while reading: {e}")
                        break
                    continue
                    
            except Exception as e:
                logger.error(f"Error in read loop: {e}")
                break
                
        # Clean up
        self._read_task = None
        if not self._closing:
            self.close()

    def write(self, data: bytes) -> None:
        """Write data to the transport."""
        if self._closing:
            return
            
        if not data:
            return
            
        if not self._write_buffer:
            # Try to write immediately
            try:
                if isinstance(self._ssl_sock, ssl.SSLSocket):
                    self._ssl_sock.send(data)
                else:
                    self._sock.send(data)
                return
            except (BlockingIOError, InterruptedError, ssl.SSLWantWriteError):
                pass
            except Exception as e:
                logger.error(f"Error writing data: {e}")
                self.close()
                return
                
        # Buffer the data
        self._write_buffer.extend(data)
        
        # Start write loop if not already running
        if not self._write_task:
            self._write_task = self._loop.create_task(self._write_loop())
            
        # Check write buffer size
        if len(self._write_buffer) > self._high_water and not self._write_paused:
            self._write_paused = True
            try:
                self._protocol.pause_writing()
            except Exception as e:
                logger.error(f"Error pausing protocol: {e}")

    async def _write_loop(self) -> None:
        """Write buffered data to the socket."""
        while self._write_buffer and not self._closing:
            try:
                # Try to write data
                if isinstance(self._ssl_sock, ssl.SSLSocket):
                    sent = self._ssl_sock.send(self._write_buffer)
                else:
                    sent = self._sock.send(self._write_buffer)
                    
                if sent:
                    # Update stats
                    self._extra['bytes_sent'] += sent
                    
                    # Remove sent data from buffer
                    del self._write_buffer[:sent]
                    
                    # Resume writing if buffer is below low water mark
                    if self._write_paused and len(self._write_buffer) <= self._low_water:
                        self._write_paused = False
                        try:
                            self._protocol.resume_writing()
                        except Exception as e:
                            logger.error(f"Error resuming protocol: {e}")
                            
                else:
                    # No data was sent, wait for socket to be writable
                    await self._loop.sock_sendall(self._sock, b'')
                    
            except (BlockingIOError, InterruptedError, ssl.SSLWantWriteError):
                # Wait for socket to be writable
                await self._loop.sock_sendall(self._sock, b'')
            except Exception as e:
                logger.error(f"Error in write loop: {e}")
                break
                
        # Clean up
        self._write_task = None
        if not self._closing:
            self.close()

    def close(self) -> None:
        """Close the transport."""
        if self._closing:
            return
            
        self._closing = True
        
        # Cancel read/write tasks
        if self._read_task:
            self._read_task.cancel()
        if self._write_task:
            self._write_task.cancel()
            
        # Close SSL socket
        if self._ssl_sock:
            try:
                self._ssl_sock.unwrap()
            except Exception:
                pass
            try:
                self._ssl_sock.close()
            except Exception:
                pass
            self._ssl_sock = None
            
        # Close raw socket
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            
        # Notify protocol
        try:
            self._protocol.connection_lost(None)
        except Exception as e:
            logger.error(f"Error notifying protocol of connection loss: {e}")

    def abort(self) -> None:
        """Forcefully close the transport."""
        self.close()

    def get_write_buffer_size(self) -> int:
        """Return the current size of the write buffer."""
        return len(self._write_buffer)

    def get_write_buffer_limits(self) -> Tuple[int, int]:
        """Return the write buffer limits."""
        return (self._low_water, self._high_water)

    def set_write_buffer_limits(self, high: Optional[int] = None, low: Optional[int] = None) -> None:
        """Set new write buffer limits."""
        if high is None:
            high = 64 * 1024
        if low is None:
            low = high // 4
            
        if low > high:
            raise ValueError("high must be greater than low")
            
        self._high_water = high
        self._low_water = low

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        """Get optional transport information."""
        return self._extra.get(name, default)

    def is_closing(self) -> bool:
        """Return True if the transport is closing or closed."""
        return self._closing

    def pause_reading(self) -> None:
        """Pause the receiving end of the transport."""
        self._read_paused = True

    def resume_reading(self) -> None:
        """Resume the receiving end of the transport."""
        self._read_paused = False
        if not self._read_task:
            self._start_reading()

class TlsHandler:
    """Handles TLS tunnel establishment and certificate management."""

    # Cipher configurations from most secure to most compatible
    CIPHER_SUITES = {
        "modern": (
            "ECDHE-ECDSA-AES256-GCM-SHA384:"
            "ECDHE-RSA-AES256-GCM-SHA384:"
            "ECDHE-ECDSA-CHACHA20-POLY1305:"
            "ECDHE-RSA-CHACHA20-POLY1305:"
            "ECDHE-ECDSA-AES128-GCM-SHA256:"
            "ECDHE-RSA-AES128-GCM-SHA256"
        ),
        "intermediate": (
            "ECDHE-RSA-AES128-GCM-SHA256:"
            "ECDHE-RSA-AES256-GCM-SHA384:"
            "DHE-RSA-AES128-GCM-SHA256:"
            "DHE-RSA-AES256-GCM-SHA384"
        ),
        "old": (
            "ECDHE-RSA-AES128-SHA:"
            "AES128-SHA:"
            "DES-CBC3-SHA"
        )
    }

    def _get_raw_socket(self, ssl_sock: Union[ssl.SSLSocket, socket.socket, Any]) -> socket.socket:
        """Get the raw socket from an SSL socket or transport."""
        try:
            # Case 1: Already a raw socket
            if isinstance(ssl_sock, socket.socket) and not isinstance(ssl_sock, ssl.SSLSocket):
                return ssl_sock

            # Case 2: Transport object handling (check this first)
            if hasattr(ssl_sock, 'get_extra_info'):
                sock = ssl_sock.get_extra_info('socket')
                if isinstance(sock, socket.socket):
                    return sock
                    
            # Case 3: SSL Socket handling
            if isinstance(ssl_sock, ssl.SSLSocket):
                # Try getting through _socket attribute first
                if hasattr(ssl_sock, '_socket'):
                    sock = ssl_sock._socket
                    if isinstance(sock, socket.socket):
                        return sock
                    
                # Try getting through _sock attribute
                if hasattr(ssl_sock, '_sock'):
                    sock = ssl_sock._sock
                    if isinstance(sock, socket.socket):
                        return sock
                    
                # Try getting through fileno
                try:
                    fileno = ssl_sock.fileno()
                    if fileno >= 0:
                        sock = socket.fromfd(fileno, socket.AF_INET, socket.SOCK_STREAM)
                        if isinstance(sock, socket.socket):
                            return sock
                except (socket.error, IOError):
                    pass

                # Try getting through SSLObject
                if hasattr(ssl_sock, '_sslobj'):
                    sslobj = ssl_sock._sslobj
                    if hasattr(sslobj, '_sslobj'):
                        sock = getattr(sslobj, '_sslobj', None)
                        if isinstance(sock, socket.socket):
                            return sock

            # Case 4: Try getting socket from transport attributes
            for attr in ['_socket', '_sock', 'socket', 'transport']:
                if hasattr(ssl_sock, attr):
                    sock = getattr(ssl_sock, attr)
                    if isinstance(sock, socket.socket):
                        return sock
                    # Handle nested transport case
                    if hasattr(sock, 'get_extra_info'):
                        inner_sock = sock.get_extra_info('socket')
                        if isinstance(inner_sock, socket.socket):
                            return inner_sock

            # Case 5: Try unwrapping if it's an SSL socket
            if isinstance(ssl_sock, ssl.SSLSocket):
                try:
                    sock = ssl_sock.unwrap()
                    if isinstance(sock, socket.socket):
                        return sock
                except (ssl.SSLError, OSError):
                    pass

            # Case 6: Try getting through transport's protocol
            if hasattr(ssl_sock, '_protocol'):
                protocol = getattr(ssl_sock, '_protocol')
                if hasattr(protocol, 'transport'):
                    transport = protocol.transport
                    if hasattr(transport, 'get_extra_info'):
                        sock = transport.get_extra_info('socket')
                        if isinstance(sock, socket.socket):
                            return sock

            # Log failure details with more context
            logger.error(f"[{self._connection_id}] Failed to get raw socket from {type(ssl_sock)}")
            logger.error(f"[{self._connection_id}] Available attributes: {dir(ssl_sock)}")
            if isinstance(ssl_sock, ssl.SSLSocket):
                logger.error(f"[{self._connection_id}] SSLSocket details:")
                logger.error(f"  - Server side: {ssl_sock.server_side}")
                logger.error(f"  - Server hostname: {ssl_sock.server_hostname}")
                logger.error(f"  - Session reused: {ssl_sock.session_reused}")
                logger.error(f"  - Cipher: {ssl_sock.cipher()}")
                logger.error(f"  - Version: {ssl_sock.version()}")
            
            raise RuntimeError(f"Could not extract raw socket from {type(ssl_sock)}")
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error extracting raw socket: {e}")
            raise RuntimeError(f"Could not extract raw socket: {str(e)}")

    def __init__(self, connection_id: str, state_manager: Any, error_handler: Any,
                 loop: Optional[asyncio.AbstractEventLoop] = None):
        """Initialize TLS handler with enhanced socket management."""
        self._connection_id = connection_id
        self._state_manager = state_manager
        self._error_handler = error_handler
        self._loop = loop or asyncio.get_running_loop()
        self._connection_stats: Dict[str, Dict[str, Any]] = {}
        
        # Socket management
        self._stored_sockets: Dict[int, socket.socket] = {}
        self._socket_states: Dict[int, Dict[str, Any]] = {}
        self._transports: Dict[int, asyncio.Transport] = {}
        self._max_retries = 3
        self._retry_delay = 0.5
        self._handshake_timeout = 30.0

        # HTTP message parsing state
        self._http_buffer = bytearray()
        self._current_request = None
        self._current_response = None
        self._is_request = True
        self._last_request = None
        self._database_interceptor = None

        # Initialize cert_manager with the event loop if needed
        if not cert_manager.is_running():
            asyncio.create_task(cert_manager.start(loop=self._loop))

    async def _ensure_interceptor(self) -> None:
        """Ensure database interceptor is initialized."""
        if not self._database_interceptor:
            from proxy.interceptors.database import DatabaseInterceptor
            self._database_interceptor = DatabaseInterceptor(self._connection_id)
            await self._database_interceptor._ensure_db()

    async def process_decrypted_data(self, data: bytes, protocol: asyncio.Protocol) -> None:
        """Process decrypted data before passing to protocol."""
        try:
            # Add data to HTTP buffer
            self._http_buffer.extend(data)
            
            # Try to extract complete HTTP messages
            while len(self._http_buffer) > 0:
                # For HTTP/2, check for complete frames
                if self._http_buffer.startswith(b'PRI * HTTP/2.0\r\n'):
                    logger.debug(f"[{self._connection_id}] Detected HTTP/2 connection preface")
                    # Pass HTTP/2 data directly to protocol
                    protocol.data_received(self._http_buffer)
                    self._http_buffer.clear()
                    break

                # For HTTP/1.x, look for message boundary
                if b'\r\n\r\n' not in self._http_buffer:
                    # Don't break immediately - check if we have a partial HTTP message
                    first_line = self._http_buffer.split(b'\r\n', 1)[0] if b'\r\n' in self._http_buffer else None
                    if not first_line or (
                        not any(first_line.startswith(method) for method in [b'GET ', b'POST ', b'PUT ', b'DELETE ', b'HEAD ', b'OPTIONS ', b'PATCH ', b'CONNECT ']) and
                        not first_line.startswith(b'HTTP/')
                    ):
                        # Not an HTTP message, pass directly to protocol
                        protocol.data_received(self._http_buffer)
                        self._http_buffer.clear()
                    break  # Wait for more data
                    
                # Split headers and potential body
                header_block, rest = self._http_buffer.split(b'\r\n\r\n', 1)
                header_lines = header_block.split(b'\r\n')
                
                if not header_lines:
                    # Invalid HTTP message
                    protocol.data_received(self._http_buffer)
                    self._http_buffer.clear()
                    break
                    
                first_line = header_lines[0].decode('utf-8', errors='ignore')
                
                # Parse headers
                headers = {}
                content_length = 0
                for line in header_lines[1:]:
                    try:
                        line = line.decode('utf-8', errors='ignore')
                        if ': ' in line:
                            name, value = line.split(': ', 1)
                            headers[name] = value
                            if name.lower() == 'content-length':
                                content_length = int(value)
                    except Exception as e:
                        logger.debug(f"[{self._connection_id}] Error parsing header line: {e}")
                        continue

                # Check if we have the complete message
                if len(rest) < content_length:
                    # Wait for more data
                    break

                # Check if this is a request or response
                if first_line.startswith(('GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH', 'CONNECT')):
                    # Parse request
                    try:
                        method, target, *_ = first_line.split(' ')
                        
                        body = rest[:content_length] if content_length > 0 else None
                        
                        # Create request object
                        from proxy.interceptor import InterceptedRequest
                        intercepted_request = InterceptedRequest(
                            method=method,
                            url=target,
                            headers=headers,
                            body=body,
                            connection_id=self._connection_id
                        )
                        
                        # Store for matching with response
                        self._current_request = intercepted_request
                        self._is_request = False
                        
                        # Pass to database interceptor
                        await self._ensure_interceptor()
                        if self._database_interceptor:
                            await self._database_interceptor.intercept(intercepted_request)
                            logger.info(f"[{self._connection_id}] Successfully intercepted HTTP request: {method} {target}")
                        
                        # Pass complete request to protocol
                        protocol.data_received(self._http_buffer[:len(header_block) + 4 + content_length])
                        
                        # Update buffer
                        self._http_buffer = bytearray(rest[content_length:])
                        
                    except Exception as e:
                        logger.error(f"[{self._connection_id}] Error processing request: {e}")
                        protocol.data_received(self._http_buffer)
                        self._http_buffer.clear()
                        
                elif first_line.startswith('HTTP/'):
                    # Parse response
                    try:
                        version, status_code, *reason = first_line.split(' ')
                        status_code = int(status_code)
                        
                        body = rest[:content_length] if content_length > 0 else None
                        
                        # Create response object
                        from proxy.interceptor import InterceptedResponse
                        intercepted_response = InterceptedResponse(
                            status_code=status_code,
                            headers=headers,
                            body=body,
                            connection_id=self._connection_id
                        )
                        
                        # Pass to database interceptor
                        await self._ensure_interceptor()
                        if self._database_interceptor:
                            await self._database_interceptor.intercept(intercepted_response, self._current_request)
                            logger.info(f"[{self._connection_id}] Successfully intercepted HTTP response: {status_code}")
                        
                        # Pass complete response to protocol
                        protocol.data_received(self._http_buffer[:len(header_block) + 4 + content_length])
                        
                        # Update buffer
                        self._http_buffer = bytearray(rest[content_length:])
                        
                        # Reset request flag
                        self._is_request = True
                        
                    except Exception as e:
                        logger.error(f"[{self._connection_id}] Error processing response: {e}")
                        protocol.data_received(self._http_buffer)
                        self._http_buffer.clear()
                else:
                    # Not an HTTP message
                    protocol.data_received(self._http_buffer)
                    self._http_buffer.clear()
                    
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error processing decrypted data: {e}")
            # Pass data directly to protocol on error
            protocol.data_received(data)

    async def handle_tls_connection(self, client_transport: asyncio.Transport,
                                host: str, port: int) -> Optional[asyncio.Transport]:
        """Handle TLS connection."""
        try:
            # Wait for cert_manager to be ready
            while not cert_manager.is_running():
                await asyncio.sleep(0.1)
                
            # Create SSL context with current event loop
            ssl_context = await cert_manager.get_context(host, is_server=True)
            return await self._establish_local_tls(client_transport, ssl_context, host)
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] TLS connection failed: {e}")
            raise

    async def wrap_client(self, protocol: TlsCapableProtocol,
                         server_hostname: str,
                         alpn_protocols: Optional[list[str]] = None) -> Tuple[asyncio.Transport, TlsCapableProtocol]:
        """Wrap client connection with TLS."""
        try:
            logger.debug(f"[{self._connection_id}] Starting client wrap for {server_hostname}")
            
            # Log protocol state
            logger.debug(f"[{self._connection_id}] Protocol type: {type(protocol)}")
            logger.debug(f"[{self._connection_id}] Protocol transport: {type(protocol.transport) if protocol.transport else None}")
            
            # Get client context
            try:
                ssl_context = get_client_context(server_hostname)
                logger.debug(
                    f"[{self._connection_id}] Created client SSL context - "
                    f"Protocol: {ssl_context.protocol}, "
                    f"Options: {ssl_context.options}, "
                    f"Verify mode: {ssl_context.verify_mode}"
                )
            except Exception as e:
                logger.error(f"[{self._connection_id}] Failed to create client SSL context: {e}", exc_info=True)
                raise
                
            # Configure ALPN if specified
            if alpn_protocols:
                try:
                    ssl_context.set_alpn_protocols(alpn_protocols)
                    logger.debug(f"[{self._connection_id}] Set ALPN protocols: {alpn_protocols}")
                except Exception as e:
                    logger.warning(f"[{self._connection_id}] Failed to set ALPN protocols: {e}")
                
            # Create SSL transport with detailed error handling
            try:
                logger.debug(f"[{self._connection_id}] Creating SSL transport with handshake timeout of 30s")
                transport = await self._create_ssl_transport(
                    protocol,
                    ssl_context,
                    server_side=False,
                    server_hostname=server_hostname,
                    handshake_timeout=30.0
                )
                
                # Verify transport creation
                if not transport:
                    raise RuntimeError("SSL transport creation failed - transport is None")
                    
                # Log transport state
                logger.debug(
                    f"[{self._connection_id}] Transport created successfully - "
                    f"Type: {type(transport)}, "
                    f"Is closing: {transport.is_closing()}, "
                    f"Has SSL: {transport.get_extra_info('ssl_object') is not None}"
                )
                
                # Get SSL info
                ssl_obj = transport.get_extra_info('ssl_object')
                if ssl_obj:
                    logger.debug(
                        f"[{self._connection_id}] SSL connection info - "
                        f"Version: {ssl_obj.version()}, "
                        f"Cipher: {ssl_obj.cipher()}, "
                        f"Server hostname: {ssl_obj.server_hostname}, "
                        f"Compression: {ssl_obj.compression()}"
                    )
                else:
                    logger.warning(f"[{self._connection_id}] No SSL object available in transport")
                
            except Exception as e:
                logger.error(f"[{self._connection_id}] Failed to create SSL transport: {e}", exc_info=True)
                raise
            
            logger.debug(f"[{self._connection_id}] Client wrap completed successfully")
            return transport, protocol
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to wrap client connection: {e}", exc_info=True)
            raise

    async def wrap_server(self, protocol: TlsCapableProtocol,
                         server_hostname: Optional[str] = None,
                         alpn_protocols: Optional[list[str]] = None) -> Tuple[asyncio.Transport, TlsCapableProtocol]:
        """Wrap server connection with TLS."""
        try:
            logger.debug(f"[{self._connection_id}] Wrapping server connection")
            
            # Create server context
            ssl_context = get_server_context(server_hostname) if server_hostname else ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Configure ALPN if specified
            if alpn_protocols:
                ssl_context.set_alpn_protocols(alpn_protocols)
            
            # Create SSL transport without server_hostname (server mode)
            transport = await self._create_ssl_transport(
                protocol=protocol,
                ssl_context=ssl_context,
                server_side=True,  # Explicitly set server side
                handshake_timeout=30.0
            )
            
            return transport, protocol
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to wrap server connection: {e}")
            raise

    async def _create_ssl_transport(self,
                                  protocol: TlsCapableProtocol,
                                  ssl_context: ssl.SSLContext,
                                  server_side: bool = False,
                                  server_hostname: Optional[str] = None,
                                  handshake_timeout: float = 30.0) -> asyncio.Transport:
        """Create SSL transport with optional delayed handshake for MITM debugging."""
        try:
            # Get raw socket from protocol's transport
            raw_socket = protocol.transport.get_extra_info('socket')
            if not raw_socket:
                raise RuntimeError("Could not get raw socket from transport")
            
            # Create new SSL socket without immediate handshake
            try:
                wrap_kwargs = {
                    'sock': raw_socket,
                    'server_side': server_side,
                    'do_handshake_on_connect': False
                }
                
                # Only add server_hostname for client-side connections
                if not server_side and server_hostname:
                    wrap_kwargs['server_hostname'] = server_hostname
                    
                logger.debug(f"[{self._connection_id}] Wrapping socket with SSL - kwargs: {wrap_kwargs}")
                ssl_sock = ssl_context.wrap_socket(**wrap_kwargs)
                
                # Set non-blocking mode
                ssl_sock.setblocking(False)
                logger.debug(f"[{self._connection_id}] SSL socket created and set to non-blocking mode")
                
            except Exception as e:
                logger.error(f"[{self._connection_id}] Failed to create SSL socket: {e}", exc_info=True)
                raise
                
            # Perform SSL handshake
            try:
                logger.debug(f"[{self._connection_id}] Starting SSL handshake")
                await self._do_handshake(ssl_sock, self._loop)
                logger.debug(f"[{self._connection_id}] SSL handshake completed successfully")
                
                # Log SSL connection info
                logger.debug(
                    f"[{self._connection_id}] SSL connection established - "
                    f"Version: {ssl_sock.version()}, "
                    f"Cipher: {ssl_sock.cipher()}, "
                    f"Compression: {ssl_sock.compression()}"
                )
                
            except Exception as e:
                logger.error(f"[{self._connection_id}] SSL handshake failed: {e}", exc_info=True)
                raise
                
            # Create transport
            try:
                # Create a new protocol that wraps the original one
                wrapper_protocol = type("TlsWrapperProtocol", (), {
                    "connection_made": lambda s, t: protocol.connection_made(t),
                    "data_received": lambda s, d: protocol.data_received(d),
                    "eof_received": lambda s: protocol.eof_received() if hasattr(protocol, 'eof_received') else None,
                    "connection_lost": lambda s, e: protocol.connection_lost(e),
                    "pause_writing": lambda s: protocol.pause_writing() if hasattr(protocol, 'pause_writing') else None,
                    "resume_writing": lambda s: protocol.resume_writing() if hasattr(protocol, 'resume_writing') else None,
                    "get_extra_info": lambda s, n, d=None: protocol.get_extra_info(n, d) if hasattr(protocol, 'get_extra_info') else d,
                    "transport": None
                })()
                
                # Create SSL transport
                transport = _SslTransport(
                    tls_handler=self,
                    loop=self._loop,
                    ssl_sock=ssl_sock,
                    protocol=wrapper_protocol
                )
                
                # Set transport on wrapper protocol
                wrapper_protocol.transport = transport
                
                # Log transport creation
                logger.debug(
                    f"[{self._connection_id}] Created SSL transport - "
                    f"Type: {type(transport)}, "
                    f"Protocol: {type(wrapper_protocol)}, "
                    f"Original protocol: {type(protocol)}"
                )
                
                return transport
                
            except Exception as e:
                logger.error(f"[{self._connection_id}] Failed to create SSL transport: {e}", exc_info=True)
                raise
                
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to create SSL transport: {e}", exc_info=True)
            raise

    def update_connection_stats(self, connection_id: str, **kwargs) -> None:
        """Update connection statistics."""
        if connection_id not in self._connection_stats:
            self._connection_stats[connection_id] = {}
        self._connection_stats[connection_id].update(kwargs)

    def get_connection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all connection statistics."""
        return self._connection_stats.copy()

    def close_connection(self, connection_id: str) -> None:
        """Close and cleanup a connection."""
        logger.debug(f"[{connection_id}] Closing connection")
        
        # Clean up any transports associated with this connection
        for transport_id, transport in list(self._transports.items()):
            if transport and not transport.is_closing():
                transport.close()
            self._transports.pop(transport_id, None)
            
        # Clean up any sockets
        for sock_id, socket in list(self._stored_sockets.items()):
            try:
                socket.close()
            except Exception as e:
                logger.debug(f"[{connection_id}] Error closing socket: {e}")
            self._stored_sockets.pop(sock_id, None)
            self._socket_states.pop(sock_id, None)
            
        # Clean up connection stats
        self._connection_stats.pop(connection_id, None)
        
        logger.debug(f"[{connection_id}] Connection cleanup complete")

    async def establish_tls_tunnel(self, host: str, port: int, 
                                 client_transport: asyncio.Transport,
                                 connection_id: str) -> Tuple[bool, Optional[asyncio.Transport]]:
        """Establish TLS tunnel with server and client using MITM handshake."""
        try:
            logger.info(f"[{connection_id}] Starting TLS tunnel establishment to {host}:{port}")
            
            # Create connection handler
            remote_handler = ConnectionHandler(f"{connection_id}_remote")
            
            # Connect to remote server
            logger.debug(f"[{connection_id}] Attempting to connect to remote server")
            transport = await self._loop.create_connection(
                lambda: remote_handler,
                host=host,
                port=port
            )
            
            if not transport or not transport[0]:
                logger.error(f"[{connection_id}] Failed to establish remote connection")
                raise RuntimeError("Failed to establish remote connection")
                
            logger.info(f"[{connection_id}] Successfully connected to remote server")
            
            # Start TLS handshake
            logger.debug(f"[{connection_id}] Starting TLS handshake")
            success = await self._perform_tls_handshake(transport[0], host)
            
            if not success:
                logger.error(f"[{connection_id}] TLS handshake failed")
                return False, None
                
            logger.info(f"[{connection_id}] TLS tunnel established successfully")
            return True, transport[0]
            
        except Exception as e:
            logger.error(f"[{connection_id}] Failed to establish TLS tunnel: {str(e)}")
            return False, None

    async def _perform_tls_handshake(self, transport: asyncio.Transport, host: str) -> bool:
        """Perform TLS handshake with enhanced logging and error recovery."""
        start_time = time.time()
        retry_count = 0
        
        while retry_count < self._max_retries:
            try:
                logger.debug(f"[{self._connection_id}] TLS handshake attempt {retry_count + 1}")
                
                # Get socket info for debugging
                sock = transport.get_extra_info('socket')
                if sock:
                    try:
                        blocking = not sock.getsockopt(socket.SOL_SOCKET, socket.SO_NONBLOCK)
                        logger.debug(f"[{self._connection_id}] Socket state: blocking={blocking}")
                    except Exception as e:
                        logger.warning(f"[{self._connection_id}] Could not get socket state: {e}")
                
                # Create SSL context
                ssl_context = get_client_context(host)
                logger.debug(f"[{self._connection_id}] Created SSL context for {host}")
                
                # Perform handshake
                async with timeout(self._handshake_timeout):
                    # Wrap socket with SSL
                    ssl_sock = ssl_context.wrap_socket(
                        sock,
                        server_hostname=host,
                        do_handshake_on_connect=False
                    )
                    
                    logger.debug(f"[{self._connection_id}] Starting SSL handshake")
                    ssl_sock.do_handshake()
                    
                    # Log successful handshake details
                    cipher = ssl_sock.cipher()
                    version = ssl_sock.version()
                    logger.info(
                        f"[{self._connection_id}] TLS handshake successful:\n"
                        f"  Protocol: {version}\n"
                        f"  Cipher: {cipher}\n"
                        f"  Time taken: {time.time() - start_time:.2f}s"
                    )
                    
                    return True
                    
            except asyncio.TimeoutError:
                duration = time.time() - start_time
                logger.error(f"[{self._connection_id}] Handshake timeout after {duration:.2f}s")
                self._last_error = "Handshake timeout"
                
            except ssl.SSLError as e:
                logger.error(f"[{self._connection_id}] SSL error during handshake: {e}")
                if "WRONG_VERSION_NUMBER" in str(e):
                    logger.error(f"[{self._connection_id}] Protocol version mismatch")
                    break  # Don't retry on protocol version mismatch
                self._last_error = f"SSL error: {str(e)}"
                
            except Exception as e:
                logger.error(f"[{self._connection_id}] Unexpected error during handshake: {e}")
                self._last_error = f"Unexpected error: {str(e)}"
            
            retry_count += 1
            if retry_count < self._max_retries:
                delay = self._retry_delay * (2 ** retry_count)
                logger.info(f"[{self._connection_id}] Retrying handshake in {delay:.1f}s")
                await asyncio.sleep(delay)
        
        logger.error(
            f"[{self._connection_id}] TLS handshake failed after {retry_count} attempts. "
            f"Last error: {self._last_error}"
        )
        return False

    def _create_ssl_contexts(self, cert_hostname: str) -> Tuple[ssl.SSLContext, ssl.SSLContext]:
        """Create SSL contexts for local and remote connections with OpenSSL 3.0 compatibility."""
        logger.debug(f"[{self._connection_id}] Creating SSL contexts for {cert_hostname}")

        # Create local context (server-side) with modern defaults
        local_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        local_ctx.verify_mode = ssl.CERT_NONE
        local_ctx.check_hostname = False
        
        # Set security level to 1 for broader compatibility
        local_ctx.set_ciphers('ALL@SECLEVEL=1')
        
        # Essential options for server side
        local_ctx.options = (
            ssl.OP_ALL |  # Enable bug workarounds
            ssl.OP_NO_SSLv2 | 
            ssl.OP_NO_SSLv3 |  # Disable very old protocols
            ssl.OP_CIPHER_SERVER_PREFERENCE |  # Server chooses cipher
            ssl.OP_SINGLE_DH_USE |  # Improve forward secrecy
            ssl.OP_NO_COMPRESSION  # Disable compression (CRIME attack)
        )
        
        if hasattr(ssl, 'OP_ENABLE_MIDDLEBOX_COMPAT'):
            local_ctx.options |= ssl.OP_ENABLE_MIDDLEBOX_COMPAT
            
        # Load certificate
        try:
            cert_path, key_path = cert_manager.get_cert_pair(cert_hostname)
            local_ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
            logger.debug(f"[{self._connection_id}] Loaded certificate for {cert_hostname}")
        except Exception as e:
            raise RuntimeError(f"Failed to load certificate: {e}")

        # Set ALPN protocols
        local_ctx.set_alpn_protocols(['h2', 'http/1.1'])

        # Create remote context (client-side) with flexible settings
        remote_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        remote_ctx.verify_mode = ssl.CERT_NONE  # Skip verification for MITM
        remote_ctx.check_hostname = False
        remote_ctx.set_alpn_protocols(['h2', 'http/1.1'])
        
        # Set security level to 1 and allow legacy protocols if needed
        remote_ctx.set_ciphers('ALL@SECLEVEL=1')
        
        # Configure options for client side
        remote_ctx.options = (
            ssl.OP_ALL |  # Enable bug workarounds
            ssl.OP_NO_SSLv2 |
            ssl.OP_NO_SSLv3 |
            ssl.OP_SINGLE_DH_USE
        )
        
        if hasattr(ssl, 'OP_ENABLE_MIDDLEBOX_COMPAT'):
            remote_ctx.options |= ssl.OP_ENABLE_MIDDLEBOX_COMPAT
            
        # Allow TLS 1.0+ for maximum compatibility
        if hasattr(remote_ctx, 'minimum_version'):
            remote_ctx.minimum_version = ssl.TLSVersion.TLSv1
            remote_ctx.maximum_version = ssl.TLSVersion.TLSv1_3

        logger.debug(f"[{self._connection_id}] Created SSL contexts with OpenSSL 3.0 compatibility")
        return local_ctx, remote_ctx
    
    async def _do_handshake(self, ssl_sock: ssl.SSLSocket, loop: asyncio.AbstractEventLoop) -> None:
        """Perform TLS handshake with enhanced error recovery."""
        start_time = time.monotonic()
        handshake_stats = {
            'duration': 0,
            'attempts': 0,
            'want_read_events': 0,
            'want_write_events': 0
        }

        try:
            # Get and validate raw socket
            raw_sock = self._get_raw_socket(ssl_sock)
            raw_sock.setblocking(False)

            while True:
                try:
                    ssl_sock.do_handshake()
                    break
                except ssl.SSLWantReadError:
                    handshake_stats['want_read_events'] += 1
                    r, _, x = select.select([raw_sock], [], [raw_sock], 0.1)
                    if x:
                        raise ssl.SSLError("Socket error during handshake")
                    if not r:
                        continue
                except ssl.SSLWantWriteError:
                    handshake_stats['want_write_events'] += 1
                    _, w, x = select.select([], [raw_sock], [raw_sock], 0.1)
                    if x:
                        raise ssl.SSLError("Socket error during handshake")
                    if not w:
                        continue
                except (socket.error, IOError) as e:
                    if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                        await asyncio.sleep(0.01)
                        continue
                    raise
                
                handshake_stats['attempts'] += 1
                if handshake_stats['attempts'] > 10:
                    raise ssl.SSLError("Handshake timeout")

            handshake_stats['duration'] = time.monotonic() - start_time
            logger.debug(f"[{self._connection_id}] Handshake completed successfully")
            logger.debug(f"[{self._connection_id}] Handshake stats: {handshake_stats}")
            logger.debug(f"[{self._connection_id}] Handshake completed in {handshake_stats['duration']:.3f}s")

        except Exception as e:
            logger.error(f"[{self._connection_id}] Unexpected handshake error: {e}")
            raise

    async def _establish_remote_connection(self, host: str, port: int, 
                                        remote_ctx: ssl.SSLContext) -> Optional[asyncio.Transport]:
        """Establish TLS connection with remote server."""
        last_error = None
        error_info = {}
        try:
            loop = asyncio.get_event_loop()
            
            async with timeout(15):
                # Create handler for remote connection
                remote_handler = ConnectHandler(
                    connection_id=f"{self._connection_id}_remote",
                    transport=None,
                    connect_timeout=10,
                    read_timeout=30
                )

                # Initialize variables
                transport = None
                sock = None
                ssl_sock = None
                
                try:
                    # Configure socket with optimized settings
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    # Set buffer sizes
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)  # 256KB
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)  # 256KB
                    sock.setblocking(False)

                    # Establish TCP connection
                    try:
                        await loop.sock_connect(sock, (host, port))
                    except Exception as e:
                        logger.error(f"[{self._connection_id}] TCP connection failed: {e}")
                        raise

                    # Enhanced TLS socket initialization with version-aware fallback
                    def create_ssl_socket(use_sni: bool, force_tls10: bool = False) -> ssl.SSLSocket:
                        try:
                            nonlocal remote_ctx
                            logger.debug(f"[{self._connection_id}] Creating TLS socket with SNI={'enabled' if use_sni else 'disabled'}, TLS1.0={'forced' if force_tls10 else 'auto'}")
                            
                            if force_tls10:
                                # Create new context for TLS 1.0
                                remote_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS)
                                remote_ctx.verify_mode = ssl.CERT_NONE
                                remote_ctx.check_hostname = False
                                remote_ctx.options = ssl.OP_ALL
                                
                                # Force TLS 1.0
                                if hasattr(remote_ctx, 'minimum_version'):
                                    remote_ctx.minimum_version = ssl.TLSVersion.TLSv1
                                    remote_ctx.maximum_version = ssl.TLSVersion.TLSv1
                                    
                                # Set legacy ciphers
                                remote_ctx.set_ciphers(self.CIPHER_SUITES['old'])
                                
                                # Enable legacy compatibility options
                                remote_ctx.options &= ~(
                                    getattr(ssl, 'OP_NO_TLSv1', 0) |
                                    getattr(ssl, 'OP_NO_COMPRESSION', 0))
                                if hasattr(ssl, 'OP_LEGACY_SERVER_CONNECT'):
                                    remote_ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT
                                if hasattr(ssl, 'OP_ENABLE_MIDDLEBOX_COMPAT'):
                                    remote_ctx.options |= ssl.OP_ENABLE_MIDDLEBOX_COMPAT

                            kwargs = {
                                "sock": sock,
                                "server_side": False,
                                "do_handshake_on_connect": False
                            }
                            if use_sni:
                                kwargs["server_hostname"] = self._state_manager.tls_state.get("sni_hostname", host)
                            return remote_ctx.wrap_socket(**kwargs)
                        except Exception as e:
                            logger.error(f"[{self._connection_id}] SSL socket creation failed: {e}")
                            raise

                    # Start TLS connection attempt
                    logger.debug(f"[{self._connection_id}] Attempting TLS handshake with {host}:{port}")
                    client_version = self._state_manager.tls_state.get("client_version")
                    
                    try:
                        # First attempt - use client's preferred version
                        logger.info(f"[{self._connection_id}] Attempting handshake with version: {client_version}")
                        ssl_sock = create_ssl_socket(use_sni=True)
                    except ssl.SSLError as e:
                        if client_version == "TLSv1.0" or "protocol_version" in str(e).lower():
                            logger.warning(f"[{self._connection_id}] Initial handshake failed, forcing TLS 1.0: {e}")
                            # Close existing socket
                            try:
                                sock.close()
                            except Exception:
                                pass
                                
                            # Create new socket for TLS 1.0
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                            sock.setblocking(False)
                            
                            # Reconnect
                            await loop.sock_connect(sock, (host, port))
                            
                            # Try with forced TLS 1.0
                            ssl_sock = create_ssl_socket(use_sni=False, force_tls10=True)
                        else:
                            logger.warning(f"[{self._connection_id}] TLS handshake failed with SNI, trying without: {e}")
                            # Try without SNI but keep version settings
                            ssl_sock = create_ssl_socket(use_sni=False)
                    
                    sock = None  # Transfer ownership
                    
                    # Enhanced handshake with retries and detailed error handling
                    logger.debug(f"[{self._connection_id}] Starting TLS handshake negotiation")
                    start_time = loop.time()
                    handshake_timeout = 15.0  # Increased timeout for legacy TLS
                    retry_count = 0
                    max_retries = 3
                    
                    while retry_count < max_retries:
                        try:
                            # Log detailed handshake attempt info
                            logger.debug(f"[{self._connection_id}] Handshake attempt {retry_count + 1}")
                            try:
                                blocking = not ssl_sock.getsockopt(socket.SOL_SOCKET, socket.SO_NONBLOCK)
                            except (AttributeError, socket.error):
                                blocking = 'unknown'
                            logger.debug(f"[{self._connection_id}] Socket state: fd={ssl_sock.fileno()}, blocking={blocking}")
                            
                            # Log TLS settings
                            ctx_info = {
                                'verify_mode': remote_ctx.verify_mode,
                                'check_hostname': remote_ctx.check_hostname,
                                'options': hex(remote_ctx.options),
                                'socket_family': sock.family if sock else 'unknown',
                                'socket_protocol': sock.proto if sock else 'unknown',
                                'blocking': not sock.getsockopt(socket.SOL_SOCKET, socket.SO_NONBLOCK) if sock else 'unknown',
                                'cipher': ssl_sock.cipher(),
                                'version': ssl_sock.version(),
                            }
                            logger.debug(f"[{self._connection_id}] TLS context settings: {ctx_info}")
                            
                            # Attempt handshake
                            ssl_sock.do_handshake()
                            
                            # Log successful handshake details
                            cipher = ssl_sock.cipher()
                            version = ssl_sock.version()
                            logger.info(f"[{self._connection_id}] Handshake successful:")
                            logger.info(f"  Version: {version}")
                            logger.info(f"  Cipher: {cipher}")
                            logger.info(f"  Shared ciphers: {ssl_sock.shared_ciphers() if hasattr(ssl_sock, 'shared_ciphers') else 'unknown'}")
                            logger.info(f"  TLS compression: {ssl_sock.compression()}")
                            logger.info(f"  Server hostname: {ssl_sock.server_hostname}")
                            break

                        except (ssl.SSLWantReadError, ssl.SSLWantWriteError) as e:
                            # Check both retry count and timeout
                            elapsed = loop.time() - start_time
                            if elapsed > handshake_timeout:
                                raise ssl.SSLError(f"Handshake timeout after {elapsed:.1f}s")
                            if isinstance(e, ssl.SSLWantReadError):
                                logger.debug(f"[{self._connection_id}] Waiting for read ({elapsed:.1f}s)")
                            else:
                                logger.debug(f"[{self._connection_id}] Waiting for write ({elapsed:.1f}s)")
                                
                            # Enhanced socket readiness check with stricter timeouts
                            want_read = isinstance(e, ssl.SSLWantReadError)
                            want_write = isinstance(e, ssl.SSLWantWriteError)
                            fileno = ssl_sock.fileno()
                            if fileno < 0:
                                raise ssl.SSLError("Invalid socket file descriptor")
                            
                            try:
                                async with timeout(min(1.0, handshake_timeout - elapsed)):
                                    readable = [fileno] if want_read else []
                                    writable = [fileno] if want_write else []
                                    
                                    # Enhanced select with error recovery and shorter timeouts
                                    def do_select() -> Tuple[List[int], List[int], List[int]]:
                                        try:
                                            # Use shorter select timeout to catch stalls
                                            r, w, x = select.select(readable, writable, [], 0.1)
                                            return r, w, x
                                        except Exception as se:
                                            if se.args[0] == errno.EINTR:  # Interrupted system call
                                                return [], [], []
                                            logger.error(f"[{self._connection_id}] Select error: {se}")
                                            raise ssl.SSLError(f"Socket select failed: {se}")
                                    
                                    r, w, _ = await loop.run_in_executor(None, do_select)
                                    if not (r if want_read else w):
                                        # Log stall and use progressive backoff
                                        logger.warning(f"[{self._connection_id}] Handshake stalled (attempt {retry_count + 1})")
                                        sleep_time = min(0.1 * (2 ** retry_count), 1.0)
                                        await asyncio.sleep(sleep_time)
                            except Exception as e:
                                logger.error(f"[{self._connection_id}] Handshake select error: {e}")
                                if isinstance(e, ssl.SSLError):
                                    raise
                                raise ssl.SSLError(f"Handshake error: {str(e)}")

                        except ssl.SSLError as e:
                            # Special handling for SSL_ERROR_SYSCALL
                            if "syscall" in str(e).lower():
                                logger.error(f"[{self._connection_id}] SSL_ERROR_SYSCALL during handshake: {e}")
                                # Recreate socket and try again
                                ssl_sock.close()
                                retry_count += 1
                                if retry_count < max_retries:
                                    logger.info(f"[{self._connection_id}] Attempting socket recreation after syscall error")
                                    # Create new socket with enhanced error handling
                                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                                    sock.setblocking(False)
                                    
                                    # Try connection with timeout
                                    try:
                                        await asyncio.wait_for(
                                            loop.sock_connect(sock, (host, port)),
                                            timeout=5.0
                                        )
                                        ssl_sock = create_ssl_socket(use_sni=False)  # Try without SNI
                                        continue
                                    except asyncio.TimeoutError:
                                        logger.error(f"[{self._connection_id}] Socket recreation timed out")
                                        raise ssl.SSLError("Socket recreation timeout")
                            else:
                                last_error = e
                                retry_count += 1
                                if retry_count < max_retries:
                                    logger.warning(f"[{self._connection_id}] Handshake attempt {retry_count} failed: {e}")
                                    await asyncio.sleep(0.5 * retry_count)  # Exponential backoff
                                    continue
                            logger.error(f"[{self._connection_id}] All handshake attempts failed: {e}")
                            raise

                        except Exception as e:
                            logger.error(f"[{self._connection_id}] Handshake error: {e}")
                            raise

                    # Configure transport with TLS version enforcement
                    logger.debug(f"[{self._connection_id}] Creating transport with TLS {ssl_sock.version()}")
                    
                    # Log negotiated TLS version
                    actual_version = ssl_sock.version()
                    logger.info(f"[{self._connection_id}] Successfully negotiated {actual_version} with remote server")

                    # Create transport with verified SSL socket
                    transport = None
                    try:
                        remote_handler.transport = None
                        transport = _SslTransport(
                            tls_handler=self,
                            loop=loop,
                            ssl_sock=ssl_sock,
                            protocol=remote_handler
                        )
                        remote_handler.transport = transport
                        
                        # Configure transport settings based on version
                        version = ssl_sock.version()
                        write_buffer_size = 16384 if version == "TLSv1" else 524288
                        transport.set_write_buffer_limits(high=write_buffer_size)
                        
                        # Wait for protocol initialization
                        await asyncio.sleep(0.1)
                        
                        if transport.is_closing():
                            raise RuntimeError("Transport closed during initialization")
                            
                    except Exception as e:
                        # Clean up on error
                        logger.error(f"[{self._connection_id}] Failed to create SSL transport: {e}")
                        if transport:
                            try:
                                transport.abort()
                            except Exception:
                                pass
                        # Ensure socket is closed
                        try:
                            ssl_sock.close()
                        except Exception:
                            pass
                        raise RuntimeError(f"Failed to create SSL transport: {e}")

                    version = self._state_manager.tls_state.get("client_version", "unknown")
                    logger.info(f"[{self._connection_id}] Remote TLS connection established using {version} settings")

                    # Get negotiated protocol version
                    ssl_object = transport.get_extra_info('ssl_object')
                    if ssl_object:
                        logger.info(f"[{self._connection_id}] Negotiated {ssl_object.version()} with remote server using cipher {ssl_object.cipher()}")

                except (ssl.SSLError, OSError) as e:
                    logger.error(f"[{self._connection_id}] SSL/Socket error with {host}:{port} - {str(e)}")
                    if isinstance(e, ssl.SSLError):
                        # Enhanced SSL error logging
                        error_info = {
                            'error_type': type(e).__name__,
                            'error_number': e.errno if hasattr(e, 'errno') else 'unknown',
                            'ssl_lib': e.library if hasattr(e, 'library') else 'unknown',
                            'reason': e.reason if hasattr(e, 'reason') else 'unknown',
                            'verify_code': e.verify_code if hasattr(e, 'verify_code') else 'unknown',
                            'verify_message': e.verify_message if hasattr(e, 'verify_message') else 'unknown'
                        }
                        logger.error(f"[{self._connection_id}] SSL error details:")
                        for key, value in error_info.items():
                            logger.error(f"  {key}: {value}")
                        
                        # Check if it's a syscall error
                        if "syscall" in str(e).lower():
                            sock_info = {
                                'local_address': sock.getsockname() if sock else 'unknown',
                                'remote_address': sock.getpeername() if sock else 'unknown',
                                'blocking': not sock.getsockopt(socket.SOL_SOCKET, socket.SO_NONBLOCK) if sock else 'unknown',
                                'type': sock.type if sock else 'unknown',
                                'error': sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR) if sock else 'unknown'
                            }
                            logger.error(f"[{self._connection_id}] Socket state at error:")
                            for key, value in sock_info.items():
                                logger.error(f"  {key}: {value}")
                    raise RuntimeError(f"Failed to establish remote TLS connection: {str(e)} (Details: {error_info})")
                else:
                    # Configure transport settings
                    if hasattr(transport, 'set_write_buffer_limits'):
                        transport.set_write_buffer_limits(high=524288)  # 512KB

                    # Never returns a tuple since we create transport directly
                    self._remote_transport = transport
                    return transport
                finally:
                    # Clean up any partial connections
                    if 'sock' in locals():
                        with suppress(Exception):
                            sock.close()
                    if 'ssl_sock' in locals() and 'sock' not in locals():
                        with suppress(Exception):
                            ssl_sock.close()

        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to establish remote connection: {e}")
            if last_error:
                logger.error(f"[{self._connection_id}] Previous error: {last_error}")
            raise

    async def _establish_local_tls(self, client_transport: asyncio.Transport,
                                 local_ctx: ssl.SSLContext,
                                 cert_hostname: str) -> Optional[asyncio.Transport]:
        """Establish TLS connection with client."""
        try:
            loop = asyncio.get_event_loop()
            
            async with timeout(30):
                max_retries = 3
                retry_count = 0
                last_error = None
                retry_delay = 0.1
                
                while retry_count < max_retries:
                    try:
                        logger.debug(f"[{self._connection_id}] Local handshake attempt {retry_count + 1}")
                        
                        # Extract raw socket from client transport
                        raw_socket = client_transport.get_extra_info('socket')
                        if raw_socket is None:
                            raise RuntimeError("Could not get socket from transport")
                        
                        # Create non-blocking SSL socket
                        ssl_sock = local_ctx.wrap_socket(
                            raw_socket,
                            server_side=True,
                            do_handshake_on_connect=False,
                            server_hostname=cert_hostname
                        )
                        ssl_sock.setblocking(False)
                        
                        # Perform handshake with timeout
                        try:
                            async with timeout(15):
                                logger.debug(f"[{self._connection_id}] Starting local handshake")
                                while True:
                                    try:
                                        ssl_sock.do_handshake()
                                        break
                                    except ssl.SSLWantReadError:
                                        await loop.sock_recv(ssl_sock, 0)
                                    except ssl.SSLWantWriteError:
                                        await loop.sock_sendall(ssl_sock, b'')
                                    except ssl.SSLError as e:
                                        if "renegotiation" in str(e).lower():
                                            continue
                                        if "alert" in str(e).lower():
                                            logger.error(f"[{self._connection_id}] TLS alert during handshake: {e}")
                                        else:
                                            logger.error(f"[{self._connection_id}] SSL error during handshake: {e}")
                                        raise
                                
                                logger.debug(f"[{self._connection_id}] Local handshake completed")
                                
                            # Create transport with successful SSL socket
                            proto = type("DummyProtocol", (), {
                                "connection_made": lambda s, t: None,
                                "data_received": lambda s, d: None,
                                "eof_received": lambda s: None
                            })()
                            tls_transport = _SslTransport(
                                tls_handler=self,
                                loop=loop,
                                ssl_sock=ssl_sock,
                                protocol=proto
                            )
                            
                            # Log TLS session info
                            ssl_object = ssl_sock
                            if ssl_object:
                                session_info = {
                                    'cipher': ssl_object.cipher(),
                                    'version': ssl_object.version(),
                                    'compression': ssl_object.compression(),
                                    'selected_alpn': ssl_object.selected_alpn_protocol()
                                }
                                logger.info(f"[{self._connection_id}] Local TLS session established: {session_info}")
                            
                            return tls_transport
                            
                        except asyncio.TimeoutError:
                            logger.error(f"[{self._connection_id}] Local handshake timed out")
                            raise ssl.SSLError("Handshake timeout")
                        except Exception as e:
                            logger.error(f"[{self._connection_id}] Local handshake error: {e}")
                            raise
                            
                    except ssl.SSLError as e:
                        if "alert" in str(e).lower():
                            logger.warning(f"[{self._connection_id}] TLS alert during handshake: {e}")
                            raise
                        else:
                            logger.error(f"[{self._connection_id}] SSL error during handshake: {e}")
                            last_error = e
                            retry_count += 1
                            if retry_count < max_retries:
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2
                                logger.debug(f"[{self._connection_id}] Retrying handshake in {retry_delay}s")
                                continue
                            raise

        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to establish local TLS: {e}")
            if last_error:
                raise last_error
            raise

    def _update_tls_state(self, transport: asyncio.Transport) -> None:
        """Update TLS state from established connection."""
        ssl_object = transport.get_extra_info('ssl_object')
        if ssl_object:
            self._state_manager.tls_state.update(
                handshake_complete=True,
                cipher=ssl_object.cipher(),
                version=ssl_object.version(),
                alpn_protocols=ssl_object.selected_alpn_protocol()
            )

    def cleanup(self) -> None:
        """Clean up all resources."""
        logger.debug(f"[{self._connection_id}] Starting TLS handler cleanup")
        
        # Close all transports
        for transport_id, transport in list(self._transports.items()):
            try:
                if transport and not transport.is_closing():
                    transport.close()
            except Exception as e:
                logger.debug(f"[{self._connection_id}] Error closing transport: {e}")
            self._transports.pop(transport_id, None)
            
        # Close all sockets
        for sock_id, socket in list(self._stored_sockets.items()):
            try:
                socket.close()
            except Exception as e:
                logger.debug(f"[{self._connection_id}] Error closing socket: {e}")
            self._stored_sockets.pop(sock_id, None)
            self._socket_states.pop(sock_id, None)
            
        # Clear connection stats
        self._connection_stats.clear()
        
        logger.debug(f"[{self._connection_id}] TLS handler cleanup complete")
