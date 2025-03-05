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

from async_timeout import timeout as async_timeout

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
        """Initialize SSL transport with robust socket access.
        
        Args:
            tls_handler: The TLS handler managing this transport
            loop: The event loop
            ssl_sock: The SSL socket to wrap
            protocol: The protocol to use
        """
        super().__init__()
        self._tls_handler = tls_handler
        self._loop = loop
        self._ssl_sock = ssl_sock
        self._closing = False
        self._write_buffer = bytearray()
        
        # Enhanced raw socket retrieval
        self._sock = None
        
        # First try direct socket access if it's already a raw socket
        if isinstance(ssl_sock, socket.socket) and not isinstance(ssl_sock, ssl.SSLSocket):
            self._sock = ssl_sock
        elif isinstance(ssl_sock, ssl.SSLSocket):
            # For SSLSocket, try to get the underlying socket
            if hasattr(ssl_sock, '_socket'):
                self._sock = ssl_sock._socket
            elif hasattr(ssl_sock, '_sock'):
                self._sock = ssl_sock._sock
            # Try unwrapping as last resort for SSLSocket
            if not self._sock:
                try:
                    self._sock = ssl_sock.unwrap()
                except Exception:
                    pass
        else:
            # For other types (like transports), try various methods
            for attr in ['_sock', '_socket', 'socket']:
                try:
                    sock = getattr(ssl_sock, attr, None)
                    if isinstance(sock, socket.socket):
                        self._sock = sock
                        break
                    # Handle TransportSocket wrapper
                    if str(type(sock)) == "<class 'asyncio.trsock.TransportSocket'>" and hasattr(sock, '_sock'):
                        self._sock = sock._sock
                        break
                except Exception:
                    continue
            
            # If still no socket, try transport method
            if not self._sock and hasattr(ssl_sock, 'get_extra_info'):
                try:
                    sock = ssl_sock.get_extra_info('socket')
                    if isinstance(sock, socket.socket):
                        self._sock = sock
                    elif hasattr(sock, '_sock'):
                        self._sock = sock._sock
                except Exception:
                    pass
                    
        if not self._sock:
            logger.error("Failed to get raw socket from SSL socket")
            logger.error(f"SSL socket type: {type(ssl_sock)}")
            logger.error(f"Available attributes: {dir(ssl_sock)}")
            raise RuntimeError("Could not get raw socket from SSL socket")
            
        # Validate socket
        try:
            fileno = self._sock.fileno()
            if fileno < 0:
                raise RuntimeError("Socket has invalid file descriptor")
                
            # Test socket operations
            self._sock.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
            self._sock.getpeername()  # Verify connection
            
            # Check socket error state
            error_code = self._sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if error_code != 0:
                raise RuntimeError(f"Socket has pending error: {error_code}")
        except Exception as e:
            logger.error(f"Socket validation failed: {e}")
            raise RuntimeError(f"Invalid socket: {e}")
            
        # Set socket to non-blocking mode
        self._sock.setblocking(False)
            
        self._protocol = protocol
        
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
        
    def _start_reading(self) -> None:
        """Start reading data from socket."""
        if not self._closing:
            asyncio.create_task(self._read_loop())
            
    async def _read_loop(self) -> None:
        """Continuously read data from socket with enhanced error recovery."""
        BUFFER_SIZE = 32768  # 32KB buffer size
        READ_TIMEOUT = 30.0  # Read timeout
        MAX_EMPTY_READS = 5  # Maximum consecutive empty reads
        empty_reads = 0
        last_activity = time.monotonic()

        while not self._closing:
            try:
                # Enhanced socket validation
                raw_sock = None
                # Enhanced socket validation with type checking
                try:
                    # First unwrap any SSLSocket
                    if isinstance(self._sock, ssl.SSLSocket):
                        try:
                            raw_sock = self._sock._sock
                        except AttributeError:
                            raw_sock = getattr(self._sock, '_socket', None)
                        
                        if not raw_sock:
                            logger.error("Could not get raw socket from SSLSocket")
                            break
                    else:
                        raw_sock = self._sock

                    # Validate raw socket
                    if not isinstance(raw_sock, socket.socket):
                        logger.error(f"Invalid socket type: {type(raw_sock)}")
                        break
                        
                    if raw_sock.fileno() < 0:
                        logger.error("Socket is closed")
                        break

                    # Test socket state
                    try:
                        raw_sock.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
                        raw_sock.getpeername()  # Verify connection
                    except (socket.error, IOError) as e:
                        logger.error(f"Socket validation failed: {e}")
                        break

                except Exception as e:
                    logger.error(f"Socket access error: {type(e).__name__}: {e}")
                    break

                # Safety check for SSL socket
                if not self._ssl_sock or getattr(self._ssl_sock, '_closed', True):
                    logger.error("SSL socket is closed or invalid")
                    break

                # Read data with timeout and state tracking
                try:
                    async with async_timeout(READ_TIMEOUT):
                        # Try non-blocking read
                        try:
                            import fcntl
                            flags = fcntl.fcntl(raw_sock.fileno(), fcntl.F_GETFL)
                            fcntl.fcntl(raw_sock.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
                            
                            data = await self._loop.sock_recv(raw_sock, BUFFER_SIZE)
                            
                            if data:
                                empty_reads = 0
                                last_activity = time.monotonic()
                            else:
                                empty_reads += 1
                                if empty_reads >= MAX_EMPTY_READS:
                                    logger.debug("Maximum consecutive empty reads - closing connection")
                                    break
                                await asyncio.sleep(0.01)  # Small delay
                                continue
                                
                            fcntl.fcntl(raw_sock, fcntl.F_SETFL, flags)  # Restore flags
                            
                        except (BlockingIOError, InterruptedError):
                            await asyncio.sleep(0.01)
                            continue
                            
                except asyncio.TimeoutError:
                    elapsed = time.monotonic() - last_activity
                    if elapsed > READ_TIMEOUT:
                        logger.warning(f"Read timeout after {elapsed:.1f}s inactivity")
                        break
                    continue
                except ConnectionError as e:
                    logger.error(f"Connection error during read: {e}")
                    break
                
                # Process SSL data
                try:
                    ssl_sock = self._ssl_sock
                    
                    # Use a dedicated decryption method
                    def decrypt_data(data: bytes) -> Optional[bytes]:
                        try:
                            if not data:
                                return None
                                
                            decrypted = ssl_sock.recv(len(data))
                            if not isinstance(decrypted, bytes):
                                logger.error(f"Invalid decrypted data type: {type(decrypted)}")
                                return None
                                
                            return decrypted
                        except ssl.SSLError as e:
                            if "renegotiation" in str(e).lower():
                                logger.debug("Handling TLS renegotiation")
                                ssl_sock.do_handshake()
                                return None
                            raise
                            
                    # Decrypt and process data
                    decrypted = decrypt_data(data)
                    if decrypted:
                        # Update stats
                        data_len = len(decrypted)
                        self._extra['bytes_received'] += data_len
                        logger.debug(f"Processed {data_len} bytes")
                        
                        # Monitor TLS state
                        try:
                            version = ssl_sock.version()
                            cipher = ssl_sock.cipher()
                            if cipher:
                                logger.debug(f"Using {version}, cipher: {cipher[0]}")
                        except Exception as e:
                            logger.debug(f"Could not get SSL info: {e}")
                            
                        # Forward data if protocol available
                        if self._protocol:
                            try:
                                self._protocol.data_received(decrypted)
                            except Exception as e:
                                logger.error(f"Protocol error: {e}")
                                break
                        else:
                            logger.error("No protocol available")
                            break
                except ssl.SSLWantReadError:
                    await asyncio.sleep(0.001)  # Small delay before retry
                    continue
                except ssl.SSLWantWriteError:
                    # Handle write blocking with timeout
                    try:
                        async with async_timeout(1.0):
                            await self.write_data(b'')  # Progress SSL state
                    except asyncio.TimeoutError:
                        logger.warning("SSL write operation timed out")
                    continue
                except ssl.SSLError as e:
                    if "renegotiation" in str(e).lower():
                        try:
                            # Handle TLS renegotiation with timeout
                            async with async_timeout(5.0):
                                logger.debug("Handling TLS renegotiation request")
                                self._ssl_sock.do_handshake()
                            continue
                        except Exception as re:
                            logger.error(f"Renegotiation failed: {re}")
                            break
                    logger.error(f"SSL error in read loop: {e}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error processing SSL data: {e}")
                    break
                    
            except Exception as e:
                if isinstance(e, (ConnectionError, ConnectionResetError)):
                    logger.debug(f"Connection closed: {e}")
                else:
                    logger.error(f"Error in read loop: {type(e).__name__}: {e}")
                break
                
        # Connection is closed
        logger.debug("Read loop ended")
        if not self._closing:
            self._closing = True
            self._protocol.connection_lost(None)

    def write(self, data: Union[bytes, bytearray, memoryview]) -> None:
        """Write data to transport."""
        if self._closing:
            return
            
        try:
            # Create task for async write
            asyncio.create_task(self.write_data(bytes(data)))
        except Exception as e:
            logger.error(f"Error creating write task: {e}")
            self.close()

    async def write_data(self, data: bytes) -> None:
        """Write data with TLS renegotiation handling."""
        WRITE_TIMEOUT = 30.0  # Write timeout in seconds
        CHUNK_SIZE = 16384  # 16KB chunks for better performance
        
        if not data:
            return
            
        try:
            # Validate socket and SSL state
            if not self._sock or self._sock.fileno() < 0:
                raise RuntimeError("Invalid socket in write operation")
            if not self._ssl_sock or getattr(self._ssl_sock, '_closed', True):
                raise RuntimeError("SSL socket is closed or invalid")
                
            # Process data in chunks
            remaining = data
            while remaining:
                chunk = remaining[:CHUNK_SIZE]
                chunk_size = len(chunk)
                
                try:
                    async with async_timeout(WRITE_TIMEOUT):
                        while chunk:
                            try:
                                # Attempt to send data
                                sent = self._ssl_sock.send(chunk)
                                if sent > 0:
                                    self._extra['bytes_sent'] += sent
                                    chunk = chunk[sent:]
                                else:
                                    # No progress - small delay before retry
                                    await asyncio.sleep(0.001)
                                    
                            except ssl.SSLWantReadError:
                                # Wait for read availability
                                await asyncio.sleep(0.001)
                                continue
                                
                            except ssl.SSLWantWriteError:
                                # Wait for write availability
                                await asyncio.sleep(0.001)
                                continue
                                
                            except ssl.SSLError as e:
                                if "renegotiation" in str(e).lower():
                                    try:
                                        # Handle TLS renegotiation with timeout
                                        async with async_timeout(5.0):
                                            logger.debug("Handling TLS renegotiation during write")
                                            self._ssl_sock.do_handshake()
                                        continue
                                    except Exception as re:
                                        logger.error(f"Write renegotiation failed: {re}")
                                        self.close()
                                        raise ssl.SSLError(f"Renegotiation failed: {re}")
                                raise
                                
                except asyncio.TimeoutError:
                    logger.error(f"Write operation timed out after {WRITE_TIMEOUT}s")
                    raise
                    
                # Move to next chunk
                remaining = remaining[chunk_size:]
        except Exception as e:
            logger.error(f"Error in write_data: {type(e).__name__}: {e}")
            self.close()
            if isinstance(e, ssl.SSLError):
                # Log detailed SSL error information
                error_info = {
                    'error_type': type(e).__name__,
                    'error_number': getattr(e, 'errno', 'unknown'),
                    'ssl_lib': getattr(e, 'library', 'unknown'),
                    'reason': getattr(e, 'reason', 'unknown')
                }
                logger.error(f"SSL error details: {error_info}")
            raise

    def close(self) -> None:
        """Close the transport."""
        if not self._closing:
            self._closing = True
            try:
                # Attempt graceful SSL shutdown
                self._ssl_sock.unwrap()
            except Exception:
                pass
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass  # Socket might be closed already
            try:
                self._sock.close()
            except Exception:
                pass
            self._protocol.connection_lost(None)

    def abort(self) -> None:
        """Abort the transport."""
        self.close()

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        """Get transport extra information."""
        return self._extra.get(name, default)

    def is_closing(self) -> bool:
        """Check if transport is closing."""
        return self._closing

    def pause_reading(self) -> None:
        """Pause reading from socket."""
        pass  # Not implemented for fallback mode

    def resume_reading(self) -> None:
        """Resume reading from socket."""
        pass  # Not implemented for fallback mode

    def set_write_buffer_limits(self, high: Optional[int] = None, low: Optional[int] = None) -> None:
        """Set write buffer limits."""
        pass  # Not implemented for fallback mode

    def get_write_buffer_size(self) -> int:
        """Get current write buffer size."""
        return len(self._write_buffer)  # Return actual buffer size

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
            if isinstance(ssl_sock, socket.socket):
                return ssl_sock

            # Case 2: SSL Socket handling
            if isinstance(ssl_sock, ssl.SSLSocket):
                # Try getting through _socket attribute (Python 3.10+)
                if hasattr(ssl_sock, '_socket'):
                    sock = getattr(ssl_sock, '_socket')
                    if isinstance(sock, socket.socket):
                        return sock
                
                # Try getting through the SSLObject's raw socket
                if hasattr(ssl_sock, '_sslobj') and ssl_sock._sslobj is not None:
                    raw_socket = getattr(ssl_sock._sslobj, '_sslobj', None)
                    if isinstance(raw_socket, socket.socket):
                        return raw_socket

                # Try unwrapping
                try:
                    return ssl_sock.unwrap()
                except (ssl.SSLError, OSError):
                    pass

                # Try getting through fileno
                if hasattr(ssl_sock, 'fileno'):
                    fileno = ssl_sock.fileno()
                    if fileno != -1:
                        return socket.fromfd(fileno, socket.AF_INET, socket.SOCK_STREAM)

            # Case 3: Transport object handling
            if hasattr(ssl_sock, 'get_extra_info'):
                sock = ssl_sock.get_extra_info('socket')
                if isinstance(sock, socket.socket):
                    return sock
                
            # Case 4: _SelectorSocketTransport handling
            transport_type = type(ssl_sock).__name__
            if transport_type == '_SelectorSocketTransport':
                # Try multiple known attributes
                for attr in ['_socket', '_sock', 'socket']:
                    if hasattr(ssl_sock, attr):
                        sock = getattr(ssl_sock, attr)
                        if isinstance(sock, socket.socket):
                            return sock
                
                # Try get_extra_info if available
                if hasattr(ssl_sock, 'get_extra_info'):
                    sock = ssl_sock.get_extra_info('socket')
                    if isinstance(sock, socket.socket):
                        return sock

            # Log available attributes for debugging
            logger.error(f"Failed to get raw socket from {type(ssl_sock)}")
            logger.error(f"Available attributes: {dir(ssl_sock)}")
            
            raise RuntimeError(f"Could not extract raw socket from {type(ssl_sock)}")
            
        except Exception as e:
            logger.error(f"Error extracting raw socket: {e}")
            raise RuntimeError(f"Could not extract raw socket from {type(ssl_sock)}: {str(e)}")

    def __init__(self, connection_id: str, state_manager: Any, error_handler: Any,
                 loop: Optional[asyncio.AbstractEventLoop] = None):
        """Initialize TLS handler with enhanced socket management.
        
        Args:
            connection_id: Unique connection identifier
            state_manager: State management instance
            error_handler: Error handling instance
            loop: Optional event loop
        """
        self._connection_id = connection_id
        self._state_manager = state_manager
        self._error_handler = error_handler
        self._loop = loop or asyncio.get_event_loop()
        self._connection_stats: Dict[str, Dict[str, Any]] = {}
        
        # Socket management
        self._stored_sockets: Dict[int, socket.socket] = {}
        self._socket_states: Dict[int, Dict[str, Any]] = {}
        
        logger.debug(f"[{connection_id}] Initialized TLS handler with socket tracking")
        
    async def wrap_client(self, protocol: TlsCapableProtocol,
                         server_hostname: str,
                         alpn_protocols: Optional[list[str]] = None) -> Tuple[asyncio.Transport, TlsCapableProtocol]:
        """Wrap client connection with TLS."""
        try:
            logger.debug(f"[{self._connection_id}] Wrapping client connection for {server_hostname}")
            # Get client context
            ssl_context = get_client_context(server_hostname)
            if alpn_protocols:
                ssl_context.set_alpn_protocols(alpn_protocols)
                
            # Create SSL transport
            transport = await self._create_ssl_transport(
                protocol,
                ssl_context,
                server_side=False,
                server_hostname=server_hostname
            )
            
            logger.debug(f"[{self._connection_id}] Client connection wrapped successfully")
            return transport, protocol
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to wrap client connection: {e}")
            raise

    async def wrap_server(self, protocol: TlsCapableProtocol,
                         server_hostname: Optional[str] = None,
                         alpn_protocols: Optional[list[str]] = None) -> Tuple[asyncio.Transport, TlsCapableProtocol]:
        """Wrap server connection with TLS."""
        try:
            logger.debug(f"[{self._connection_id}] Wrapping server connection")
            # Get server context
            ssl_context = get_server_context(server_hostname)
            if alpn_protocols:
                ssl_context.set_alpn_protocols(alpn_protocols)
                
            # Create SSL transport with server hostname
            transport = await self._create_ssl_transport(
                protocol,
                ssl_context,
                server_side=True,
                server_hostname=server_hostname
            )
            
            logger.debug(f"[{self._connection_id}] Server connection wrapped successfully")
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
        """Create SSL transport with optional delayed handshake for MITM debugging.
        
        Note: This intentionally uses behavior from Python <3.10.13 for MITM debugging.
        Do not use in production environments.
        """
        try:
            # Get the raw socket from the protocol's transport
            raw_socket = self._get_raw_socket(protocol.transport)
            
            # Set non-blocking mode
            raw_socket.setblocking(False)
            
            # Create SSL socket without immediate handshake
            ssl_sock = ssl_context.wrap_socket(
                raw_socket,
                server_side=server_side,
                server_hostname=server_hostname,
                do_handshake_on_connect=False  # Key for pre-3.10.13 behavior
            )
            
            # Create transport with the SSL socket
            transport = _SslTransport(self, self._loop, ssl_sock, protocol)
            
            # Store for cleanup
            self._transports[id(transport)] = transport
            
            return transport
            
        except Exception as e:
            logging.error(f"Failed to create SSL transport: {e}")
            if 'ssl_sock' in locals():
                ssl_sock.close()
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
        """Close and cleanup connection resources with improved error handling."""
        try:
            # Clean up socket tracking
            for sock_id, socket in list(self._stored_sockets.items()):
                try:
                    # Check if socket is still valid
                    if socket.fileno() >= 0:
                        socket.shutdown(socket.SHUT_RDWR)
                    socket.close()
                except (socket.error, IOError):
                    pass  # Ignore errors during cleanup
                finally:
                    self._stored_sockets.pop(sock_id, None)
                    self._socket_states.pop(sock_id, None)

            # Clean up connection stats
            self._connection_stats.pop(connection_id, None)
            logger.debug(f"[{connection_id}] Connection resources cleaned up")

        except Exception as e:
            logger.error(f"[{connection_id}] Error during connection cleanup: {e}")

    async def establish_tls_tunnel(self, host: str, port: int, 
                                 client_transport: asyncio.Transport,
                                 connection_id: str) -> Tuple[bool, Optional[asyncio.Transport]]:
        """Establish TLS tunnel with server and client using MITM handshake."""
        try:
            # Create connection handler
            remote_handler = ConnectionHandler(f"{connection_id}_remote")
            
            # Connect to remote server
            transport = await self._loop.create_connection(
                lambda: remote_handler,
                host=host,
                port=port
            )
            
            if not transport or not transport[0]:
                raise RuntimeError("Failed to establish remote connection")
                
            return True, transport[0]
            
        except Exception as e:
            logger.error(f"Failed to establish TLS tunnel: {e}")
            return False, None

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
                    if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EINPROGRESS):
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
            
            async with async_timeout(15):
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
                                async with async_timeout(min(1.0, handshake_timeout - elapsed)):
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
            
            async with async_timeout(30):
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
                            async with async_timeout(15):
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
