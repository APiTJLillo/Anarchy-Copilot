"""HTTPS interception protocol."""
import asyncio
import contextlib
import logging
import os
import ssl
import time
from typing import Optional, Callable, ClassVar, Dict, Any, TYPE_CHECKING, Tuple, Union, Awaitable, cast
from uuid import uuid4
import aiofiles
from aiofiles import os as aio_os
import collections
import socket

from async_timeout import timeout

if TYPE_CHECKING:
    from ..certificates import CertificateAuthority
    from ..handlers.connect_factory import ConnectConfig

from .base import BaseProxyProtocol 
from .error_handler import ErrorHandler
from .buffer_manager import BufferManager
from .state_manager import StateManager
from .tls_handler import TlsHandler
from .types import Request
from ..tls_helper import cert_manager, CertificateManager
from ..handlers.http import HttpRequestHandler
from proxy.interceptors.database import DatabaseInterceptor
from proxy.interceptor import InterceptedRequest, InterceptedResponse
from database import AsyncSessionLocal
from proxy.models import ProxySessionData
from sqlalchemy import text, select
from proxy.session import get_active_sessions
from ..custom_protocol import TunnelProtocol

logger = logging.getLogger("proxy.core")
logger.setLevel(logging.DEBUG)

# Also enable debug logging for SSL/TLS operations
logging.getLogger("ssl").setLevel(logging.DEBUG)

class ServerProtocol(asyncio.Protocol):
    """Protocol for handling server-side connection."""
    
    def __init__(self, state_manager):
        super().__init__()
        self._state_manager = state_manager
        self._transport = None
        self._data_queue = asyncio.Queue()
        self._data_processor = None
        self._handshake_complete = asyncio.Event()
        self._handshake_timeout = 10.0  # 10 seconds timeout
        self._ssl_context = None
        self._ssl_object = None
        self._target_transport = None
        self._target_protocol = None
        self._logger = logging.getLogger(__name__)

    def connection_made(self, transport):
        self._transport = transport
        self._data_processor = asyncio.create_task(self._process_data())
        self._logger.debug("Connection established")

    async def _setup_tls(self, hostname):
        try:
            self._logger.debug(f"Setting up TLS for {hostname}")
            
            # Initialize certificate manager if needed
            if not cert_manager.is_running():
                self._logger.debug("Starting certificate manager")
                await cert_manager.start()
                self._logger.debug("Certificate manager started")
            
            # Get certificate pair
            cert_path, key_path = await cert_manager.get_cert_pair(hostname)
            self._logger.debug(f"Got certificate pair for {hostname}")
            
            # Create SSL context
            self._ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self._ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path)
            self._ssl_context.set_alpn_protocols(['http/1.1'])
            self._ssl_context.options |= (
                ssl.OP_NO_COMPRESSION |
                ssl.OP_NO_SSLv2 |
                ssl.OP_NO_SSLv3 |
                ssl.OP_NO_TLSv1 |
                ssl.OP_NO_TLSv1_1
            )
            self._ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')
            
            # Create SSL object
            self._ssl_object = self._ssl_context.wrap_socket(
                socket.socket(),
                server_side=True,
                do_handshake_on_connect=False
            )
            self._logger.debug("TLS setup complete")
            return True
        except Exception as e:
            self._logger.error(f"Error setting up TLS: {e}")
            return False

    async def _process_data(self):
        try:
            while True:
                try:
                    data = await self._data_queue.get()
                    if not data:
                        break

                    if not self._handshake_complete.is_set():
                        try:
                            if not self._ssl_object:
                                # First data chunk, try to parse CONNECT request
                                request = HTTPParser.parse_request(data)
                                if request and request.method == 'CONNECT':
                                    hostname = request.target.split(':')[0]
                                    if await self._setup_tls(hostname):
                                        self._transport.write(b'HTTP/1.1 200 Connection Established\r\n\r\n')
                                        continue
                                    else:
                                        self._transport.close()
                                        return

                            # Process TLS handshake
                            self._ssl_object.write(data)
                            handshake_data = self._ssl_object.read()
                            if handshake_data:
                                self._transport.write(handshake_data)
                                self._handshake_complete.set()
                                self._logger.debug("TLS handshake completed successfully")
                        except ssl.SSLError as e:
                            if e.args[0] == ssl.SSL_ERROR_WANT_READ:
                                continue
                            self._logger.error(f"SSL error during handshake: {e}")
                            self._transport.close()
                            return
                        except Exception as e:
                            self._logger.error(f"Error during handshake: {e}")
                            self._transport.close()
                            return
                    else:
                        # Normal data processing after handshake
                        try:
                            decrypted_data = self._ssl_object.unwrap(data)
                            if self._target_transport:
                                self._target_transport.write(decrypted_data)
                        except Exception as e:
                            self._logger.error(f"Error processing data: {e}")
                            self._transport.close()
                            return
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self._logger.error(f"Error in data processing loop: {e}")
                    if self._transport and not self._transport.is_closing():
                        self._transport.close()
                    break
        except Exception as e:
            self._logger.error(f"Fatal error in data processor: {e}")
            if self._transport and not self._transport.is_closing():
                self._transport.close()
        finally:
            self._logger.debug("Data processor task ended")

    def data_received(self, data):
        self._data_queue.put_nowait(data)

    def connection_lost(self, exc):
        if self._data_processor:
            self._data_processor.cancel()
        if self._target_transport:
            self._target_transport.close()
        if self._ssl_object:
            self._ssl_object.unwrap()
        super().connection_lost(exc)

class HttpsInterceptProtocol(BaseProxyProtocol):
    """Protocol for intercepting HTTPS traffic."""

    _cleanup_tasks: ClassVar[Dict[str, asyncio.Task]] = {}
    _ca_initialized: ClassVar[bool] = False
    _initialization_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self, *, connection_id: str, target_host: str, target_port: int,
                 ca: Any, cert_manager: Any, db_interceptor: Any,
                 client_transport: Optional[asyncio.Transport] = None):
        """Initialize HTTPS intercept protocol.
        
        Args:
            connection_id: Unique connection identifier
            target_host: Target host to connect to
            target_port: Target port to connect to
            ca: Certificate authority instance
            cert_manager: Certificate manager instance
            db_interceptor: Database interceptor instance
            client_transport: Optional client transport
        """
        super().__init__(connection_id)
        self.connection_id = connection_id
        self._target_host = target_host
        self._target_port = target_port
        self._ca_instance = ca  # Store CA instance
        self.cert_manager = cert_manager
        self.db_interceptor = db_interceptor
        self._client_context = None
        self._server_context = None
        self._client_ssl = None
        self._server_writer = None
        self._server_protocol = None
        self._tunnel_established = False
        self._client_hello_received = False
        self._server_hello_received = False
        self._handshake_complete = False
        self._closed = False
        self._buffer = bytearray()
        self.transport = client_transport
        self.target_host = target_host
        self.target_port = target_port
        self.loop = asyncio.get_running_loop()
        self.logger = logging.getLogger("proxy.core")
        self._first_tls_record = None
        self._target_transport = None
        self._target_protocol = None
        self._connect_request_complete = False
        
        # Initialize state management
        self.state_manager = StateManager(connection_id)
        self.error_handler = ErrorHandler(connection_id)
        self.buffer_manager = BufferManager()
        self.tls_handler = TlsHandler(
            connection_id=self.connection_id,
            state_handler=self.state_manager,
            error_handler=self.error_handler
        )
        
        # Initialize handlers
        self._connect_handler = None
        self._http_handler = None
        
        self.logger.debug(f"[{self.connection_id}] TLS handler initialized")

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle new connection."""
        super().connection_made(transport)
        self.transport = transport
        self._tunnel_established = False
        self._client_handshake_complete = False
        self._target_handshake_complete = False
        self._client_hello_received = False
        self._client_ssl = None
        self._target_ssl = None
        self._client_in_bio = None
        self._client_out_bio = None
        self._target_in_bio = None
        self._target_out_bio = None
        self._server_context = None
        self._client_context = None
        self._buffer = bytearray()
        logger.debug(f"[{self.connection_id}] Connection established")

    def data_received(self, data: bytes) -> None:
        """Handle received data."""
        try:
            if not self._connect_request_complete:
                # Buffer data until we process the CONNECT request
                self._buffer.extend(data)
                
                # Check if we have a complete CONNECT request
                if b"\r\n\r\n" in self._buffer:
                    # Process CONNECT request
                    request_data = bytes(self._buffer)
                    self._buffer.clear()
                    
                    # Find the end of headers
                    headers_end = request_data.find(b"\r\n\r\n")
                    if headers_end != -1:
                        # Extract request line and headers
                        headers_data = request_data[:headers_end]
                        leftover_data = request_data[headers_end + 4:]
                        
                        # Parse request line
                        request_line = headers_data.split(b"\r\n")[0].decode()
                        method, target, version = request_line.split(" ")
                        
                        if method == "CONNECT":
                            try:
                                # Parse target host and port
                                host, port = target.split(':')
                                port = int(port)
                                
                                # Store target information
                                self.target_host = host
                                self.target_port = port
                                
                                # Initialize state manager
                                self.state_manager = StateManager(
                                    connection_id=self.connection_id
                                )
                                self.logger.debug(f"[{self.connection_id}] StateManager initialized")
                                
                                # Initialize TLS handler
                                self.tls_handler = TlsHandler(
                                    connection_id=self.connection_id,
                                    state_handler=self.state_manager,
                                    error_handler=self.error_handler
                                )
                                self.logger.debug(f"[{self.connection_id}] TLS handler initialized")
                                
                                # Store target information
                                self.state_manager.set_metadata("target_host", host)
                                self.state_manager.set_metadata("target_port", port)
                                
                                # Create connection to target
                                asyncio.create_task(self._establish_tunnel_async(host, port, leftover_data))
                            except Exception as e:
                                self.logger.error(f"[{self.connection_id}] Error handling CONNECT request: {e}")
                                if self.transport and not self.transport.is_closing():
                                    self.transport.write(b"HTTP/1.1 500 Internal Server Error\r\n\r\n")
                                    self.transport.close()
                return

            # If we're waiting for the first TLS record, set the future
            if self._first_tls_record and not self._first_tls_record.done():
                self.logger.debug(f"[{self.connection_id}] Received first TLS record ({len(data)} bytes)")
                self._first_tls_record.set_result(data)
                return

            # Process TLS data
            self._process_tls_data(data)
                
        except Exception as e:
            self.logger.error(f"[{self.connection_id}] Error in data_received: {e}")
            if self.transport and not self.transport.is_closing():
                self.transport.close()

    def _process_tls_data(self, data: bytes) -> None:
        """Process TLS data after handshake is established."""
        try:
            # Write data to client input BIO
            self.logger.debug(f"[{self.connection_id}] Processing {len(data)} bytes from client")
            self._client_in_bio.write(data)
            
            # Try to complete client handshake if not done
            if not self._client_handshake_complete:
                try:
                    self._client_ssl.do_handshake()
                    self._client_handshake_complete = True
                    self.logger.debug(f"[{self.connection_id}] Client handshake completed")
                    
                    # Get negotiated protocol version and cipher
                    try:
                        version = self._client_ssl.version()
                        cipher = self._client_ssl.cipher()
                        self.logger.debug(f"[{self.connection_id}] Client TLS: {version}, {cipher}")
                    except Exception as e:
                        self.logger.warning(f"[{self.connection_id}] Error getting client TLS info: {e}")
                    
                    # Flush any pending handshake data
                    self._flush_client_bio()
                    return
                        
                except ssl.SSLWantReadError:
                    # Need more data, send any pending output
                    self.logger.debug(f"[{self.connection_id}] Client needs more handshake data")
                    self._flush_client_bio()
                    return
                except ssl.SSLError as e:
                    self.logger.error(f"[{self.connection_id}] Client handshake failed: {e}")
                    if self.transport and not self.transport.is_closing():
                        self.transport.close()
                    return

            # If we get here, client handshake is complete
            # Read decrypted data from client
            try:
                while True:
                    decrypted = self._client_ssl.read(8192)
                    if not decrypted:
                        break

                    self.logger.debug(f"[{self.connection_id}] Decrypted {len(decrypted)} bytes from client")

                    # Write decrypted client data to target SSL if available
                    if self._target_ssl and self._target_handshake_complete:
                        self._target_ssl.write(decrypted)
                    else:
                        self.logger.debug(f"[{self.connection_id}] Buffering {len(decrypted)} bytes until target connection ready")
                        # Buffer the data until target connection is ready
                        if not hasattr(self, '_pending_client_data'):
                            self._pending_client_data = []
                        self._pending_client_data.append(decrypted)

            except ssl.SSLWantReadError:
                # Need more data
                pass
            except ssl.SSLError as e:
                self.logger.error(f"[{self.connection_id}] Error processing client data: {e}")
                return
            
            # Flush any pending output
            self._flush_client_bio()
            
        except Exception as e:
            self.logger.error(f"[{self.connection_id}] Error processing TLS data: {e}")
            if self.transport and not self.transport.is_closing():
                self.transport.close()

    def _flush_client_bio(self) -> None:
        """Flush and send any pending data in the client output BIO."""
        try:
            while True:
                out_data = self._client_out_bio.read()
                if not out_data:
                    break
                if self.transport and not self.transport.is_closing():
                    self.transport.write(out_data)
        except Exception as e:
            self.logger.error(f"[{self.connection_id}] Error flushing client BIO: {e}")

    def _process_decrypted_data(self) -> None:
        """Process decrypted data from client SSL."""
        try:
            while True:
                try:
                    decrypted = self._client_ssl.read()
                    if not decrypted:
                        break
                    
                    # Write decrypted data to target SSL
                    self._target_ssl.write(decrypted)
                    
                    # Try to complete target handshake if not done
                    if not self._target_handshake_complete:
                        try:
                            self._target_ssl.do_handshake()
                            self._target_handshake_complete = True
                            self.logger.debug(f"[{self.connection_id}] Target handshake completed")
                            
                            # Get negotiated protocol version and cipher
                            try:
                                version = self._target_ssl.version()
                                cipher = self._target_ssl.cipher()
                                self.logger.debug(f"[{self.connection_id}] Target TLS: {version}, {cipher}")
                                self.state_manager.tls_state.update(
                                    target_version=version,
                                    target_cipher=cipher,
                                    target_handshake_complete=True,
                                    connection_state="target_handshake_complete"
                                )
                            except Exception as e:
                                self.logger.warning(f"[{self.connection_id}] Error getting target TLS info: {e}")
                                
                        except ssl.SSLWantReadError:
                            # Send any pending target output data
                            self._flush_target_bio()
                            return
                        except ssl.SSLError as e:
                            self.logger.error(f"[{self.connection_id}] Target handshake failed: {e}")
                            return
                    
                    # Read and forward encrypted data to target
                    self._flush_target_bio()
                    
                except ssl.SSLWantReadError:
                    break
                except ssl.SSLError as e:
                    self.logger.error(f"[{self.connection_id}] Error processing client data: {e}")
                    return
                
        except Exception as e:
            self.logger.error(f"[{self.connection_id}] Error processing decrypted data: {e}")

    def _flush_target_bio(self) -> None:
        """Flush and send any pending data in the target output BIO."""
        try:
            while True:
                out_data = self._target_out_bio.read()
                if not out_data:
                    break
                if self._target_transport and not self._target_transport.is_closing():
                    self._target_transport.write(out_data)
        except Exception as e:
            self.logger.error(f"[{self.connection_id}] Error flushing target BIO: {e}")

    async def _establish_tunnel_async(self, host: str, port: int, leftover_data: bytes) -> None:
        """Establish tunnel asynchronously after CONNECT."""
        try:
            # Set up database interceptor first
            await self._setup_database_interceptor()
            if not self.db_interceptor:
                self.logger.error(f"[{self.connection_id}] Failed to initialize database interceptor")
                if self.transport and not self.transport.is_closing():
                    self.transport.write(b"HTTP/1.1 500 Internal Server Error\r\n\r\n")
                    self.transport.close()
                return

            # Set up TLS contexts and get certificates
            await self._setup_tls(host)

            # Create server-side TLS protocol
            class ServerSideTlsProtocol(asyncio.Protocol):
                def __init__(self, proxy):
                    self.proxy = proxy
                    self.transport = None
                    self._handshake_in_progress = True
                    self._relay_task = None
                    self._closed = False
                    self._server_hello_sent = False
                    self._request_buffer = bytearray()  # Buffer for accumulating request data
                    self._response_buffer = bytearray()  # Buffer for accumulating response data

                def connection_made(self, transport):
                    self.transport = transport
                    self.proxy.logger.debug(f"[{self.proxy.connection_id}] Server connection established")
                    # Start server TLS handshake immediately
                    try:
                        # Send initial ClientHello
                        self.proxy._target_ssl.do_handshake()
                        self._flush_target_bio()
                    except ssl.SSLWantReadError:
                        # This is expected - need server's response
                        self._flush_target_bio()
                    except Exception as e:
                        self.proxy.logger.error(f"[{self.proxy.connection_id}] Server handshake init error: {e}")

                def _flush_target_bio(self):
                    """Helper to flush target BIO buffer to transport."""
                    try:
                        while True:
                            data = self.proxy._target_out_bio.read()
                            if not data:
                                break
                            if self.transport and not self.transport.is_closing():
                                self.transport.write(data)
                                self.proxy.logger.debug(f"[{self.proxy.connection_id}] Flushed {len(data)} bytes to server")
                    except Exception as e:
                        self.proxy.logger.error(f"[{self.proxy.connection_id}] Error flushing target BIO: {e}")

                def data_received(self, data):
                    try:
                        # Feed encrypted data from server into target SSL
                        self.proxy.logger.debug(f"[{self.proxy.connection_id}] Got {len(data)} bytes from server")
                        self.proxy._target_in_bio.write(data)

                        # Keep trying handshake until complete
                        if self._handshake_in_progress:
                            try:
                                self.proxy._target_ssl.do_handshake()
                                self._handshake_in_progress = False
                                self.proxy._target_handshake_complete = True
                                self.proxy.logger.debug(f"[{self.proxy.connection_id}] Server handshake completed")

                                # Get negotiated protocol version and cipher
                                try:
                                    version = self.proxy._target_ssl.version()
                                    cipher = self.proxy._target_ssl.cipher()
                                    self.proxy.logger.debug(f"[{self.proxy.connection_id}] Server TLS: {version}, {cipher}")

                                    # Store TLS info in database
                                    if self.proxy.db_interceptor:
                                        asyncio.create_task(self.proxy.db_interceptor.store_tls_info(
                                            "server",
                                            {
                                                "version": version,
                                                "cipher": cipher,
                                                "host": self.proxy.target_host
                                            }
                                        ))

                                except Exception as e:
                                    self.proxy.logger.warning(f"[{self.proxy.connection_id}] Error getting server TLS info: {e}")

                                # Send 200 Connection Established only after server handshake completes
                                if not self._server_hello_sent and self.proxy.transport and not self.proxy.transport.is_closing():
                                    self.proxy.transport.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                                    self._server_hello_sent = True

                                # Flush any final handshake data
                                self._flush_target_bio()

                                # Start continuous relay after handshake completes
                                self._start_relay()

                            except ssl.SSLWantReadError:
                                # Need more data, flush any pending output
                                self.proxy.logger.debug(f"[{self.proxy.connection_id}] Server needs more handshake data")
                                self._flush_target_bio()
                                return
                            except ssl.SSLError as e:
                                self.proxy.logger.error(f"[{self.proxy.connection_id}] Server handshake failed: {e}")
                                if self.proxy.transport and not self.proxy.transport.is_closing():
                                    self.proxy.transport.close()
                                return

                        # Once handshake is complete, read decrypted data from server
                        self._relay_server_to_client()

                    except Exception as e:
                        self.proxy.logger.error(f"[{self.proxy.connection_id}] Error in server data_received: {e}")
                        if self.proxy.transport and not self.proxy.transport.is_closing():
                            self.proxy.transport.close()

                def _relay_server_to_client(self):
                    """Relay decrypted data from server to client."""
                    try:
                        while True:
                            try:
                                decrypted = self.proxy._target_ssl.read(8192)
                                if not decrypted:
                                    break

                                self.proxy.logger.debug(f"[{self.proxy.connection_id}] Decrypted {len(decrypted)} bytes from server")

                                # Parse and store decrypted server response in database
                                if self.proxy.db_interceptor:
                                    try:
                                        # Try to parse HTTP response
                                        response_lines = decrypted.split(b'\r\n')
                                        if response_lines:
                                            status_line = response_lines[0].decode('utf-8')
                                            if status_line.startswith('HTTP/'):
                                                # Parse headers
                                                headers = {}
                                                i = 1
                                                while i < len(response_lines):
                                                    line = response_lines[i].decode('utf-8').strip()
                                                    if not line:
                                                        break
                                                    name, value = line.split(':', 1)
                                                    headers[name.strip()] = value.strip()
                                                    i += 1
                                                
                                                # Get body
                                                body = b'\r\n'.join(response_lines[i+1:]) if i+1 < len(response_lines) else None
                                                
                                                # Store parsed response
                                                asyncio.create_task(self.proxy.db_interceptor.store_response(decrypted))
                                            else:
                                                # Not an HTTP response, store raw data
                                                asyncio.create_task(self.proxy.db_interceptor.store_raw_data(
                                                    "target->client",
                                                    decrypted,
                                                    is_encrypted=False
                                                ))
                                        else:
                                            # No response lines, store raw data
                                            asyncio.create_task(self.proxy.db_interceptor.store_raw_data(
                                                "target->client",
                                                decrypted,
                                                is_encrypted=False
                                            ))
                                    except Exception as e:
                                        self.proxy.logger.error(f"[{self.proxy.connection_id}] Error storing response: {e}")
                                        # Store as raw data on error
                                        asyncio.create_task(self.proxy.db_interceptor.store_raw_data(
                                            "target->client",
                                            decrypted,
                                            is_encrypted=False
                                        ))

                                # Write decrypted server data to client SSL
                                self.proxy._client_ssl.write(decrypted)

                                # Flush re-encrypted data to client
                                self.proxy._flush_client_bio()

                            except ssl.SSLWantReadError:
                                # Need more data from server
                                break
                            except ssl.SSLError as e:
                                self.proxy.logger.error(f"[{self.proxy.connection_id}] Error reading from server: {e}")
                                return

                        # Flush any pending output to server
                        self._flush_target_bio()

                    except Exception as e:
                        self.proxy.logger.error(f"[{self.proxy.connection_id}] Error in server to client relay: {e}")
                        if self.proxy.transport and not self.proxy.transport.is_closing():
                            self.proxy.transport.close()

                def _start_relay(self):
                    """Start continuous data relay after handshake completion."""
                    if not self._relay_task:
                        self._relay_task = asyncio.create_task(self._continuous_relay())
                        self.proxy.logger.debug(f"[{self.proxy.connection_id}] Started continuous relay task")

                async def _continuous_relay(self):
                    """Continuously relay data between client and server."""
                    try:
                        while not self._closed:
                            # Process any pending client data first
                            if hasattr(self.proxy, '_pending_client_data'):
                                for data in self.proxy._pending_client_data:
                                    # Parse and store decrypted client request in database
                                    if self.proxy.db_interceptor:
                                        try:
                                            # Try to parse HTTP request
                                            request_lines = data.split(b'\r\n')
                                            if request_lines:
                                                request_line = request_lines[0].decode('utf-8')
                                                if ' ' in request_line:  # Looks like an HTTP request
                                                    # Parse headers
                                                    headers = {}
                                                    i = 1
                                                    while i < len(request_lines):
                                                        line = request_lines[i].decode('utf-8').strip()
                                                        if not line:
                                                            break
                                                        name, value = line.split(':', 1)
                                                        headers[name.strip()] = value.strip()
                                                        i += 1
                                                    
                                                    # Get body
                                                    body = b'\r\n'.join(request_lines[i+1:]) if i+1 < len(request_lines) else None
                                                    
                                                    # Store parsed request
                                                    asyncio.create_task(self.proxy.db_interceptor.store_request(data))
                                                else:
                                                    # Not an HTTP request, store raw data
                                                    asyncio.create_task(self.proxy.db_interceptor.store_raw_data(
                                                        "client->target",
                                                        data,
                                                        is_encrypted=False
                                                    ))
                                            else:
                                                # No request lines, store raw data
                                                asyncio.create_task(self.proxy.db_interceptor.store_raw_data(
                                                    "client->target",
                                                    data,
                                                    is_encrypted=False
                                                ))
                                        except Exception as e:
                                            self.proxy.logger.error(f"[{self.proxy.connection_id}] Error storing request: {e}")
                                            # Store as raw data on error
                                            asyncio.create_task(self.proxy.db_interceptor.store_raw_data(
                                                "client->target",
                                                data,
                                                is_encrypted=False
                                            ))
                                    
                                    self.proxy._target_ssl.write(data)
                                    self._flush_target_bio()
                                delattr(self.proxy, '_pending_client_data')

                            # Relay client to server
                            try:
                                while True:
                                    try:
                                        decrypted = self.proxy._client_ssl.read(8192)
                                        if not decrypted:
                                            break

                                        self.proxy.logger.debug(f"[{self.proxy.connection_id}] Relaying {len(decrypted)} bytes from client to server")

                                        # Parse and store decrypted client request in database
                                        if self.proxy.db_interceptor:
                                            try:
                                                # Try to parse HTTP request
                                                request_lines = decrypted.split(b'\r\n')
                                                if request_lines:
                                                    request_line = request_lines[0].decode('utf-8')
                                                    if ' ' in request_line:  # Looks like an HTTP request
                                                        # Parse headers
                                                        headers = {}
                                                        i = 1
                                                        while i < len(request_lines):
                                                            line = request_lines[i].decode('utf-8').strip()
                                                            if not line:
                                                                break
                                                            name, value = line.split(':', 1)
                                                            headers[name.strip()] = value.strip()
                                                            i += 1
                                                        
                                                        # Get body
                                                        body = b'\r\n'.join(request_lines[i+1:]) if i+1 < len(request_lines) else None
                                                        
                                                        # Store parsed request
                                                        asyncio.create_task(self.proxy.db_interceptor.store_request(decrypted))
                                                    else:
                                                        # Not an HTTP request, store raw data
                                                        asyncio.create_task(self.proxy.db_interceptor.store_raw_data(
                                                            "client->target",
                                                            decrypted,
                                                            is_encrypted=False
                                                        ))
                                                else:
                                                    # No request lines, store raw data
                                                    asyncio.create_task(self.proxy.db_interceptor.store_raw_data(
                                                        "client->target",
                                                        decrypted,
                                                        is_encrypted=False
                                                    ))
                                            except Exception as e:
                                                self.proxy.logger.error(f"[{self.proxy.connection_id}] Error storing request: {e}")
                                                # Store as raw data on error
                                                asyncio.create_task(self.proxy.db_interceptor.store_raw_data(
                                                    "client->target",
                                                    decrypted,
                                                    is_encrypted=False
                                                ))

                                        # Write decrypted client data to server SSL
                                        self.proxy._target_ssl.write(decrypted)
                                        self._flush_target_bio()

                                    except ssl.SSLWantReadError:
                                        break
                                    except ssl.SSLError as e:
                                        self.proxy.logger.error(f"[{self.proxy.connection_id}] Error reading from client: {e}")
                                        return

                            except Exception as e:
                                self.proxy.logger.error(f"[{self.proxy.connection_id}] Error in client to server relay: {e}")
                                break

                            await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning

                    except asyncio.CancelledError:
                        self.proxy.logger.debug(f"[{self.proxy.connection_id}] Relay task cancelled")
                    except Exception as e:
                        self.proxy.logger.error(f"[{self.proxy.connection_id}] Error in continuous relay: {e}")
                    finally:
                        self._closed = True

                def connection_lost(self, exc):
                    self._closed = True
                    if self._relay_task:
                        self._relay_task.cancel()
                    if exc:
                        self.proxy.logger.error(f"[{self.proxy.connection_id}] Server connection lost with error: {exc}")
                    else:
                        self.proxy.logger.debug(f"[{self.proxy.connection_id}] Server connection closed cleanly")

            # Create connection with server-side TLS protocol
            self.logger.debug(f"[{self.connection_id}] Connecting to {host}:{port}")
            loop = asyncio.get_running_loop()
            transport, protocol = await loop.create_connection(
                lambda: ServerSideTlsProtocol(self),
                host,
                port
            )
            
            # Store target transport
            self._target_transport = transport
            self._target_protocol = protocol
            
            # Set flags
            self._connect_request_complete = True
            self._tunnel_established = True
            
            # Create a future to wait for the first TLS record
            self._first_tls_record = asyncio.Future()
            
            # If we have leftover data from the CONNECT request, process it
            if leftover_data:
                self.logger.debug(f"[{self.connection_id}] Processing {len(leftover_data)} bytes of post-CONNECT data")
                self.data_received(leftover_data)
            
            # Wait for first TLS record with timeout
            try:
                async with timeout(10.0) as cm:  # 10 second timeout for first TLS record
                    data = await self._first_tls_record
                    self.logger.debug(f"[{self.connection_id}] Received first TLS record")
                    
                    # Process the first TLS record
                    self._process_tls_data(data)
                    
            except asyncio.TimeoutError:
                self.logger.error(f"[{self.connection_id}] Timeout waiting for TLS handshake")
                if self.transport and not self.transport.is_closing():
                    self.transport.close()
                return
            except Exception as e:
                self.logger.error(f"[{self.connection_id}] Error waiting for TLS handshake: {e}")
                if self.transport and not self.transport.is_closing():
                    self.transport.close()
                return
            
            self.logger.debug(f"[{self.connection_id}] Tunnel established to {host}:{port}")
            
        except Exception as e:
            self.logger.error(f"[{self.connection_id}] Error establishing tunnel: {e}")
            if self.transport and not self.transport.is_closing():
                self.transport.close()

    async def _setup_tls(self, target_host: str) -> None:
        """Set up TLS contexts and get certificates."""
        try:
            self.logger.debug(f"Setting up TLS for {target_host}")
            
            # Ensure CA is initialized
            if not self._ca_instance:
                self.logger.error("No CA instance available")
                raise RuntimeError("No CA instance available")
            
            # Initialize certificate manager if needed
            if not cert_manager.is_running():
                self.logger.debug("Starting certificate manager")
                await cert_manager.start()
                self.logger.debug("Certificate manager started")
            
            # Get certificate pair
            cert_path, key_path = await cert_manager.get_cert_pair(target_host)
            self.logger.debug(f"Got certificate pair for {target_host}")
            
            # Create server context (for client connection)
            self._server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self._server_context.load_cert_chain(cert_path, key_path)
            self._server_context.check_hostname = False
            self._server_context.verify_mode = ssl.CERT_NONE
            self._server_context.minimum_version = ssl.TLSVersion.TLSv1_2
            self._server_context.maximum_version = ssl.TLSVersion.TLSv1_3
            self._server_context.set_alpn_protocols(['http/1.1'])
            self._server_context.options |= (
                ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 |  # Disable SSL2/3
                ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1 |  # Disable TLS 1.0/1.1
                ssl.OP_SINGLE_DH_USE | ssl.OP_SINGLE_ECDH_USE |  # Improve forward secrecy
                ssl.OP_NO_COMPRESSION  # Disable compression
            )
            self._server_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')
            
            # Create client context (for target connection)
            client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            client_context.check_hostname = False  # We're intercepting, can't verify
            client_context.verify_mode = ssl.CERT_NONE
            client_context.minimum_version = ssl.TLSVersion.TLSv1_2
            client_context.maximum_version = ssl.TLSVersion.TLSv1_3
            client_context.set_alpn_protocols(['http/1.1'])
            client_context.options |= (
                ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 |  # Disable SSL2/3
                ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1 |  # Disable TLS 1.0/1.1
                ssl.OP_SINGLE_DH_USE | ssl.OP_SINGLE_ECDH_USE |  # Improve forward secrecy
                ssl.OP_NO_COMPRESSION  # Disable compression
            )
            client_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')
            
            # Create memory BIOs and wrap with SSL
            self._client_in_bio = ssl.MemoryBIO()
            self._client_out_bio = ssl.MemoryBIO()
            self._client_ssl = self._server_context.wrap_bio(
                self._client_in_bio,
                self._client_out_bio,
                server_side=True
            )
            
            self._target_in_bio = ssl.MemoryBIO()
            self._target_out_bio = ssl.MemoryBIO()
            self._target_ssl = client_context.wrap_bio(
                self._target_in_bio,
                self._target_out_bio,
                server_hostname=target_host
            )
            
            self.logger.debug(f"[{self.connection_id}] TLS setup complete for {target_host}")
            self.logger.debug(f"[{self.connection_id}] Server context version: min={self._server_context.minimum_version}, max={self._server_context.maximum_version}")
            self.logger.debug(f"[{self.connection_id}] Client context version: min={client_context.minimum_version}, max={client_context.maximum_version}")
            
        except Exception as e:
            self.logger.error(f"[{self.connection_id}] Error setting up TLS: {e}")
            if self.transport and not self.transport.is_closing():
                self.transport.close()
            raise

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection lost."""
        try:
            if exc:
                logger.debug(f"[{self.connection_id}] Connection closed with error: {exc}")
            else:
                logger.debug(f"[{self.connection_id}] Connection closed cleanly")
            
            # Clean up target transport
            if self._target_transport and not self._target_transport.is_closing():
                self._target_transport.close()
            
            # Clean up SSL objects
            self._client_ssl = None
            self._target_ssl = None
            self._client_in_bio = None
            self._client_out_bio = None
            self._target_in_bio = None
            self._target_out_bio = None
            self._server_context = None
            self._client_context = None
            
            # Clean up state
            self._tunnel_established = False
            self._client_handshake_complete = False
            self._target_handshake_complete = False
            self._client_hello_received = False
            self._buffer.clear()
            
            # Log final state
            logger.debug(f"[{self.connection_id}] Final connection state - Transport closing: {self.transport.is_closing() if self.transport else True}, Tunnel exists: {bool(self._target_transport)}, Tunnel closing: {self._target_transport.is_closing() if self._target_transport else True}")
            
            # Clear state manager
            self.state_manager.clear_state()
            logger.debug(f"[{self.connection_id}] State cleared")
            
            # Call parent
            super().connection_lost(exc)
            
            logger.debug(f"[{self.connection_id}] Connection closed")
            
        except Exception as e:
            logger.error(f"[{self.connection_id}] Error in connection_lost: {e}")
            # Ensure parent is called even if we have an error
            super().connection_lost(exc)

    @classmethod
    async def _cleanup_files(cls, cert_path: str, key_path: str) -> None:
        """Clean up certificate files."""
        task_key = f"cleanup_{cert_path}_{key_path}"
        
        try:
            loop = asyncio.get_running_loop()
            for path, desc in [(cert_path, "cert"), (key_path, "key")]:
                try:
                    exists = await loop.run_in_executor(None, os.path.exists, path)
                    if exists:
                        logger.debug(f"Removing {desc} file: {path}")
                        await aio_os.remove(path)
                        logger.debug(f"Successfully removed {desc} file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {desc} file {path}: {e}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            if task_key in cls._cleanup_tasks:
                cls._cleanup_tasks.pop(task_key, None)

    @classmethod
    def _handle_task_done(cls, task_key: str, task: asyncio.Task) -> None:
        """Handle completion of task and clean up."""
        try:
            if task.cancelled():
                logger.debug(f"Cleanup task {task_key} was cancelled")
            elif exc := task.exception():
                logger.error(f"Cleanup task {task_key} failed: {exc}")
            else:
                logger.debug(f"Cleanup task {task_key} completed successfully")
        except Exception as e:
            logger.error(f"Error handling task completion: {e}")
        finally:
            cls._cleanup_tasks.pop(task_key, None)

    @classmethod
    def create_cleanup_task(cls, cert_path: str, key_path: str, *, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Create and register a cleanup task for certificate files."""
        try:
            # Get event loop if not provided
            if loop is None:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    logger.warning("No running event loop, cleanup may be delayed")
                    return

            task_key = cls.get_cleanup_key(cert_path, key_path)

            # Cancel any existing cleanup
            if task_key in cls._cleanup_tasks:
                old_task = cls._cleanup_tasks[task_key]
                if not old_task.done():
                    old_task.cancel()
                cls._cleanup_tasks.pop(task_key)

            # Create cleanup coroutine and task
            cleanup_coro = cls._cleanup_files(cert_path, key_path)
            task = loop.create_task(cleanup_coro)
            cls._cleanup_tasks[task_key] = task

            def cleanup_callback(t: asyncio.Task) -> None:
                try:
                    if t.cancelled():
                        logger.debug(f"Cleanup task cancelled for {cert_path}")
                    elif exc := t.exception():
                        logger.error(f"Cleanup task failed for {cert_path}: {exc}")
                    else:
                        logger.debug(f"Cleanup task completed for {cert_path}")
                finally:
                    if task_key in cls._cleanup_tasks:
                        cls._cleanup_tasks.pop(task_key, None)

            task.add_done_callback(cleanup_callback)
            logger.debug(f"Created cleanup task for {cert_path} and {key_path}")
        except Exception as e:
            logger.error(f"Failed to create cleanup task: {e}")

    @classmethod
    def get_cleanup_key(cls, cert_path: str, key_path: str) -> str:
        """Get the task key for cleanup tasks."""
        return f"cleanup_{cert_path}_{key_path}"

    @classmethod
    def cancel_cleanup(cls, cert_path: str, key_path: str) -> None:
        """Cancel any existing cleanup task."""
        task_key = cls.get_cleanup_key(cert_path, key_path)
        if task_key in cls._cleanup_tasks:
            old_task = cls._cleanup_tasks[task_key]
            if not old_task.done():
                old_task.cancel()
            cls._cleanup_tasks.pop(task_key)

    @classmethod
    async def _initialize_ca(cls) -> bool:
        """Internal CA initialization method."""
        if cls._ca_initialized:
            return True
            
        try:
            if cls._ca_instance:
                logger.debug("CA already initialized")
                cls._ca_initialized = True
                return True

            if cert_manager.ca:
                cls._ca_instance = cert_manager.ca
                cls._ca_initialized = True
                logger.info("Protocol CA initialized from existing cert_manager")
                return True

            if not cert_manager.is_running():
                try:
                    logger.info("Starting cert_manager...")
                    await cert_manager.start()
                    logger.info("cert_manager started successfully")
                except Exception as e:
                    logger.error(f"Failed to start cert_manager: {e}")
                    raise RuntimeError(f"Failed to start cert_manager: {e}")

            retries = 15
            retry_delay = 2
            while retries > 0:
                if cert_manager.ca:
                    health = cert_manager.get_health()
                    if health.details.get("ca_initialized", False):
                        cls._ca_instance = cert_manager.ca
                        try:
                            test_cert_path, test_key_path = await cls._ca_instance.get_certificate("test.local")
                            if os.path.exists(test_cert_path):
                                logger.info("Protocol CA initialized and verified successfully")
                                cls._ca_initialized = True
                                # Get current loop and schedule cleanup
                                loop = asyncio.get_running_loop()
                                cls.create_cleanup_task(test_cert_path, test_key_path, loop=loop)
                                return True
                        except Exception as e:
                            logger.warning(f"CA verification failed: {e}")
                    else:
                        logger.debug("CA exists but not fully initialized yet")

                logger.debug(f"Waiting for CA initialization (retries left: {retries})")
                await asyncio.sleep(retry_delay)
                retries -= 1

            logger.error("CA initialization failed or timed out")
            return False

        except Exception as e:
            logger.error(f"Failed to initialize protocol CA: {e}", exc_info=True)
            return False

    @classmethod
    async def _wait_for_initialization(cls) -> bool:
        """Wait for any ongoing initialization to complete."""
        if not cls._initialization_task:
            return False

        try:
            # If task is not done, await it
            if not cls._initialization_task.done():
                return await cls._initialization_task
            # If task is done, get result
            return cls._initialization_task.result()
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            # Clear the failed task
            cls._initialization_task = None
            return False

    @classmethod
    async def init_ca(cls) -> bool:
        """Initialize the Certificate Authority for the protocol.
        
        Returns:
            bool: True if initialization was successful
        """
        # Early return if already initialized
        if cls._ca_initialized:
            return True

        async with cls._initialization_lock:
            # Check again after acquiring lock
            if cls._ca_initialized:
                return True

            # Wait for any existing initialization
            if await cls._wait_for_initialization():
                return True

            # Start new initialization
            try:
                coro = cls._initialize_ca()
                cls._initialization_task = asyncio.create_task(coro)
                result = await cls._initialization_task
                if result:
                    cls._ca_initialized = True
                return result
            except Exception as e:
                logger.error(f"CA initialization failed: {e}")
                if cls._initialization_task is not None:
                    cls._initialization_task.cancel()
                    cls._initialization_task = None
                return False

    @classmethod
    async def ensure_ca_initialized(cls) -> None:
        """Ensure CA is initialized, waiting if necessary."""
        if not await cls.init_ca():
            raise RuntimeError("Failed to initialize CA")

    @classmethod
    def create_protocol_factory(cls) -> Callable[..., 'HttpsInterceptProtocol']:
        """Create a protocol factory."""
        def protocol_factory(*args, **kwargs) -> 'HttpsInterceptProtocol':
            if not cls._ca_initialized:
                raise RuntimeError("CA must be initialized before creating protocol instances")
            protocol = cls(*args, **kwargs)
            protocol.state_manager.set_intercept_enabled(True)
            asyncio.create_task(protocol._setup_database_interceptor())
            connection_id = protocol.connection_id
            logger.info(f"Created HTTPS intercept protocol {connection_id}")
            return protocol

        return protocol_factory

    @classmethod
    async def ensure_initialized_factory(cls) -> Callable[..., 'HttpsInterceptProtocol']:
        """Initialize CA and return protocol factory.
        
        This is the recommended way to get a protocol factory as it ensures CA initialization.
        """
        await cls.ensure_ca_initialized()
        return cls.create_protocol_factory()

    async def _setup_database_interceptor(self) -> None:
        """Initialize the database interceptor with the active session."""
        if self.db_interceptor:
            logger.debug(f"[{self.connection_id}] Database interceptor already initialized")
            return

        try:
            logger.debug(f"[{self.connection_id}] Setting up database interceptor")
            async with AsyncSessionLocal() as db:
                # Get active session
                result = await db.execute(
                    text("SELECT * FROM proxy_sessions WHERE is_active = true ORDER BY start_time DESC LIMIT 1")
                )
                active_session = result.first()
                
                if active_session:
                    logger.debug(f"[{self.connection_id}] Found active session {active_session.id}")
                    self.db_interceptor = DatabaseInterceptor(self.connection_id)
                    # Store session ID in interceptor
                    self.db_interceptor._session_id = active_session.id
                    logger.info(f"[{self.connection_id}] Initialized database interceptor for session {active_session.id}")
                else:
                    logger.warning(f"[{self.connection_id}] No active session found for database interceptor")
                    # Create a new session
                    try:
                        logger.info(f"[{self.connection_id}] Creating new session")
                        await db.execute(
                            text("""
                                INSERT INTO proxy_sessions (
                                    name, settings, is_active, start_time
                                ) VALUES (
                                    'Auto-created Session',
                                    '{"intercept_requests": true, "intercept_responses": true}',
                                    true,
                                    CURRENT_TIMESTAMP
                                )
                            """)
                        )
                        await db.commit()
                        
                        # Get the new session
                        result = await db.execute(
                            text("SELECT * FROM proxy_sessions WHERE is_active = true ORDER BY start_time DESC LIMIT 1")
                        )
                        new_session = result.first()
                        
                        if new_session:
                            self.db_interceptor = DatabaseInterceptor(self.connection_id)
                            self.db_interceptor._session_id = new_session.id
                            logger.info(f"[{self.connection_id}] Created and initialized new session {new_session.id}")
                    except Exception as e2:
                        logger.error(f"[{self.connection_id}] Failed to create new session: {e2}", exc_info=True)
        except Exception as e:
            logger.error(f"[{self.connection_id}] Failed to setup database interceptor: {e}", exc_info=True)

    def _setup_initial_state(self) -> None:
        """Set up initial protocol state."""
        self.state_manager.set_intercept_enabled(bool(self._ca_instance))
        logger.debug(f"[{self.connection_id}] TLS interception enabled: {bool(self._ca_instance)}")

    async def handle_request(self, request: Request) -> None:
        """Handle HTTPS interception request."""
        try:
            # Convert method to string if it's bytes
            method = request.method.decode() if isinstance(request.method, bytes) else request.method
            target = request.target.decode() if isinstance(request.target, bytes) else request.target
            
            logger.debug(f"[{self.connection_id}] Handling request: {method} {target}")
            
            if method != "CONNECT":
                # For non-CONNECT requests, delegate to HTTP handler and database interceptor
                logger.debug(f"[{self.connection_id}] Handling non-CONNECT request: {method} {target}")
                
                # Ensure database interceptor is set up
                if not self._database_interceptor:
                    await self._setup_database_interceptor()
                
                if self._database_interceptor:
                    # Create intercepted request object
                    intercepted_request = InterceptedRequest(
                        method=method,
                        url=target,
                        headers=request.headers,
                        body=request.body if isinstance(request.body, bytes) else str(request.body).encode('utf-8'),
                        connection_id=self.connection_id
                    )
                    logger.debug(f"[{self.connection_id}] Created intercepted request object")
                    
                    # Let database interceptor process request
                    try:
                        modified_request = await self._database_interceptor.intercept(intercepted_request)
                        logger.debug(f"[{self.connection_id}] Successfully intercepted request")
                        
                        # Update request with any modifications from interceptor
                        # Convert string method/url to bytes if needed
                        method = modified_request.method
                        target = modified_request.url
                        request.method = method.encode() if isinstance(method, str) else method
                        request.target = target.encode() if isinstance(target, str) else target
                        request.headers = modified_request.headers
                        request.body = modified_request.body
                    except Exception as e:
                        logger.error(f"[{self.connection_id}] Failed to intercept request: {e}", exc_info=True)
                else:
                    logger.warning(f"[{self.connection_id}] No database interceptor available for request")
                    
                # Continue with normal handling
                response = await self._http_handler.handle_request(request)
                
                # Intercept response if we have an interceptor
                if self._database_interceptor and response:
                    try:
                        intercepted_response = InterceptedResponse(
                            status_code=response.status_code,
                            headers=response.headers,
                            body=response.body if isinstance(response.body, bytes) else str(response.body).encode('utf-8'),
                            connection_id=self.connection_id
                        )
                        await self._database_interceptor.intercept(intercepted_response, intercepted_request)
                        logger.debug(f"[{self.connection_id}] Successfully intercepted response")
                    except Exception as e:
                        logger.error(f"[{self.connection_id}] Failed to intercept response: {e}", exc_info=True)
                
                return

            # Parse target from request
            host, port = self._parse_authority(target)
            
            # Log request details and state
            logger.debug(f"[{self.connection_id}] Handling CONNECT request for {host}:{port}")
            logger.debug(f"[{self.connection_id}] Current transport state: {self.transport is not None and not self.transport.is_closing()}")
            logger.debug(f"[{self.connection_id}] Current tunnel state: {self._tunnel is not None and not self._tunnel.is_closing() if self._tunnel else False}")
            
            # Handle CONNECT request
            await self.handle_connect(request)
            
        except Exception as e:
            logger.error(f"[{self.connection_id}] Error handling request: {e}", exc_info=True)
            if self.transport and not self.transport.is_closing():
                self.transport.close()

    def _parse_authority(self, authority: str) -> Tuple[str, int]:
        """Parse host and port from authority string."""
        try:
            if ':' in authority:
                host, port = authority.rsplit(':', 1)
                return host, int(port)
            return authority, 443  # Default HTTPS port
        except Exception as e:
            logger.error(f"[{self.connection_id}] Failed to parse authority '{authority}': {e}")
            raise ValueError(f"Invalid authority format: {authority}")

    async def _cleanup(self, error: Optional[str] = None) -> None:
        """Clean up connection resources."""
        try:
            logger.debug(f"[{self.connection_id}] Starting cleanup, error: {error}")
            
            # Update state
            await self.state_manager.update_status("closing", error=error)
            
            # Close handlers
            try:
                if hasattr(self, '_connect_handler') and self._connect_handler:
                    self._connect_handler.close()
                if hasattr(self, '_http_handler') and self._http_handler:
                    self._http_handler.close()
                if self.db_interceptor:
                    await self.db_interceptor.close()
            except Exception as e:
                logger.warning(f"[{self.connection_id}] Error during handler cleanup: {e}")
            
            # Clear buffers
            self.buffer_manager.clear()
            
            # Final state update and cleanup
            await self.state_manager.update_status("closed")
            
        except Exception as e:
            logger.error(f"[{self.connection_id}] Cleanup error: {e}")

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
                logger.debug(f"[{self.connection_id}] Sent error response: {status_code} {message}")
        except Exception as e:
            logger.error(f"[{self.connection_id}] Failed to send error response: {e}")

    def set_tunnel(self, tunnel: asyncio.Transport) -> None:
        """Set the tunnel transport for bidirectional forwarding."""
        self._tunnel = tunnel
        self._tunnel_established = True
        logger.debug(f"[{self.connection_id}] Set tunnel transport")
        
        # Forward any pending data
        if self._buffer and self._tunnel and not self._tunnel.is_closing():
            logger.debug(f"[{self.connection_id}] Forwarding {len(self._buffer)} buffered chunks after tunnel setup")
            for data in self._buffer:
                self._tunnel.write(data)
            self._buffer.clear()

    async def handle_connect(self, request: Request) -> None:
        """Handle CONNECT request."""
        # CONNECT is now handled entirely in data_received
        # This method remains only for compatibility with the protocol interface
        pass
