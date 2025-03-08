"""Main proxy server implementation."""
import asyncio
import logging
import socket
import ssl
import signal
import os
import psutil
import sys
import uuid
import uvicorn
from contextlib import AsyncExitStack
from pathlib import Path
from async_timeout import timeout as async_timeout
from typing import Dict, List, Optional, Any, Tuple, Callable, TYPE_CHECKING
from urllib.parse import urlparse, unquote
from uuid import uuid4
from sqlalchemy import text
from datetime import datetime

# Force use of standard asyncio event loop
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

from database import AsyncSessionLocal
from ..config import ProxyConfig
from .handlers import ProxyResponse

if TYPE_CHECKING:
    from api.proxy.database_models import ProxyHistoryEntry, ProxySession as DBProxySession

from .custom_protocol import TunnelProtocol
from .protocol import HttpsInterceptProtocol
from ..interceptor import RequestInterceptor, ResponseInterceptor, InterceptedRequest, InterceptedResponse
from ..encoding import ContentEncodingInterceptor
from ..session import ProxySession
from ..websocket import WebSocketManager, DebugInterceptor
from ..analysis.analyzer import TrafficAnalyzer
from .certificates import CertificateAuthority
from .tls_helper import cert_manager, CertificateManager
from .handlers.asgi_handler import ASGIHandler

logger = logging.getLogger("proxy.core")

_instance = None

class ProxyServer:
    """Main proxy server implementation."""
    
    def __init__(self, config: ProxyConfig, add_default_interceptors: bool = False):
        """Initialize the proxy server."""
        self._config = config
        self._add_default_interceptors = add_default_interceptors
        self._port = config.port
        self._host = config.host
        self._request_interceptors = []
        self._response_interceptors = []
        self._traffic_analyzer = None
        self._is_running = False
        self._server = None
        self._asgi_handler = ASGIHandler(self)
        self._setup_interceptors()
        
        logger.debug(f"Proxy server initialized with port {self._port}")
        
        self._loop = None
        self._sessions: Dict[str, ProxySession] = {}
        self._db_initialized = False
        self._ssl_contexts: Dict[str, ssl.SSLContext] = {}
        
        # Track pending requests/responses
        self._pending_requests: Dict[str, Any] = {}
        self._pending_responses: Dict[str, Any] = {}

        # Initialize CA
        self._ca = None
        try:
            if config.ca_cert_path and config.ca_key_path:
                try:
                    # Initialize CA with absolute paths
                    ca_cert_path = config.ca_cert_path.resolve()
                    ca_key_path = config.ca_key_path.resolve()
                    
                    # Verify certificate files exist and are readable
                    if not ca_cert_path.exists():
                        logger.error(f"CA certificate not found: {ca_cert_path}")
                        raise FileNotFoundError(f"CA certificate not found: {ca_cert_path}")
                    if not ca_key_path.exists():
                        logger.error(f"CA key not found: {ca_key_path}")
                        raise FileNotFoundError(f"CA key not found: {ca_key_path}")
                    
                    # Check file permissions
                    try:
                        with open(ca_cert_path, 'rb') as f:
                            cert_data = f.read()
                        with open(ca_key_path, 'rb') as f:
                            key_data = f.read()
                    except PermissionError as e:
                        logger.error(f"Permission denied accessing CA files: {e}")
                        raise PermissionError(f"Cannot access CA files: {e}")
                    except Exception as e:
                        logger.error(f"Error reading CA files: {e}")
                        raise
                    
                    # Validate certificate format
                    try:
                        from cryptography import x509
                        from cryptography.hazmat.primitives import serialization
                        x509.load_pem_x509_certificate(cert_data)
                        serialization.load_pem_private_key(key_data, password=None)
                    except Exception as e:
                        logger.error(f"Invalid CA certificate or key format: {e}")
                        raise ValueError(f"Invalid CA certificate or key format: {e}")
                    
                    logger.info(f"Initializing CA with cert={ca_cert_path}, key={ca_key_path}")
                    self._ca = CertificateAuthority(ca_cert_path, ca_key_path)
                    
                    # Configure cert manager with CA
                    cert_manager.set_ca(self._ca)
                    logger.info("Certificate Authority initialized successfully")
                    
                except FileNotFoundError as e:
                    logger.error(f"CA file(s) not found: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Certificate Authority initialization failed: {e}")
                    raise RuntimeError(f"Failed to initialize CA: {e}")
            else:
                logger.info("No CA configuration provided - running without HTTPS interception")
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

    @classmethod
    def get_instance(cls) -> Optional['ProxyServer']:
        """Get the global proxy server instance."""
        return _instance

    @classmethod
    def configure(cls, config_dict: dict) -> None:
        """Configure or reconfigure the proxy server instance."""
        global _instance

        config = ProxyConfig.from_dict(config_dict)
        
        if _instance is None:
            _instance = cls(config)
            logger.info("Created new proxy server instance with config: %s", config_dict)
        else:
            # Update existing instance
            _instance._config.update(config_dict)
            # Update key attributes
            _instance._host = config.host
            _instance._port = config.port
            logger.info("Updated existing proxy server instance with config: %s", config_dict)

    @property
    def is_running(self) -> bool:
        """Check if the proxy server is running."""
        return self._is_running

    @property
    def port(self) -> int:
        """Get the port number the proxy server is running on."""
        return self._port

    @property
    def config(self) -> ProxyConfig:
        return self._config

    @property
    def session(self) -> ProxySession:
        """Get the proxy session."""
        if not hasattr(self, '_session'):
            self._session = ProxySession()
        return self._session

    async def _force_cleanup_port(self) -> None:
        """Force cleanup any processes using the port."""
        def is_current_process(pid: int) -> bool:
            try:
                return pid == os.getpid() or (
                    psutil.Process(pid).ppid() == os.getpid()
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return False

        def get_processes_using_port(port: int) -> List[int]:
            pids = []
            try:
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        connections = []
                        try:
                            connections = proc.connections('inet')
                        except (psutil.AccessDenied, psutil.NoSuchProcess):
                            continue
                            
                        for conn in connections:
                            try:
                                if hasattr(conn, 'laddr') and hasattr(conn.laddr, 'port'):
                                    if conn.laddr.port == port and not is_current_process(proc.pid):
                                        pids.append(proc.pid)
                            except (AttributeError, TypeError):
                                continue
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
            except Exception as e:
                logger.warning(f"Error checking processes: {e}")
            return list(set(pids))  # Remove duplicates

        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Find processes using the port
                pids = get_processes_using_port(self._port)
                if not pids:
                    logger.debug(f"No processes found using port {self._port}")
                    break

                logger.warning(f"Found processes {pids} using port {self._port}")
                
                # First try graceful termination
                for pid in pids:
                    try:
                        proc = psutil.Process(pid)
                        proc.terminate()
                    except:
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except ProcessLookupError:
                            continue

                # Wait briefly for processes to terminate
                await asyncio.sleep(retry_delay)

                # Check if processes are still running and force kill if necessary
                remaining_pids = get_processes_using_port(self._port)
                if remaining_pids:
                    logger.warning(f"Force killing remaining processes: {remaining_pids}")
                    for pid in remaining_pids:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            continue

                # Wait again and verify port is free
                await asyncio.sleep(retry_delay)
                
                # Try to bind to the port to verify it's free
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                    sock.bind((self._host, self._port))
                    sock.close()
                    logger.info(f"Successfully cleaned up port {self._port}")
                    return
                except Exception as e:
                    sock.close()
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Port {self._port} is still in use after cleanup attempts")
                    logger.warning(f"Port {self._port} still in use after cleanup attempt {attempt + 1}")
                    await asyncio.sleep(retry_delay)

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to clean up port {self._port}: {e}")
                    raise RuntimeError(f"Failed to clean up port {self._port}: {e}")
                logger.warning(f"Cleanup attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(retry_delay)

    async def start(self) -> None:
        """Start the proxy server."""
        self._loop = asyncio.get_event_loop()
        cert_manager.start()  # Start certificate manager with event loop
        
        self._server = await asyncio.start_server(
            self._handle_client,
            self._host,
            self._port,
            ssl=None,
            reuse_address=True,
            reuse_port=True,
        )
        
        addr = self._server.sockets[0].getsockname()
        logger.info(f"Proxy server started on {addr[0]}:{addr[1]}")

    async def stop(self) -> bool:
        """Stop the proxy server and cleanup all resources."""
        if not self._is_running:
            return False

        try:
            logger.info("Stopping proxy server...")
            self._is_running = False

            # Clean up pending operations
            self._pending_requests.clear()
            self._pending_responses.clear()

            # Gracefully shutdown server
            if self._server:
                try:
                    # Signal shutdown
                    self._server.should_exit = True
                    
                    # Version-compatible timeout handling
                    async with AsyncExitStack() as stack:
                        try:
                            if sys.version_info >= (3, 11):
                                await stack.enter_async_context(asyncio.timeout(3.0))
                            else:
                                await stack.enter_async_context(async_timeout(3.0))

                            # Force close any remaining connections
                            if hasattr(self._server, 'servers'):
                                for server in self._server.servers:
                                    server.close()
                                    await server.wait_closed()

                            # Final shutdown
                            await self._server.shutdown()
                        except asyncio.TimeoutError:
                            logger.warning("Server shutdown timed out, forcing cleanup")
                            if hasattr(self._server, 'servers'):
                                for server in self._server.servers:
                                    server.abort()
                except Exception as e:
                    logger.warning(f"Error during server shutdown: {e}")
                finally:
                    self._server = None

            # Close the proxy session
            await self.session.close()
            
            await self._force_cleanup_port()
            logger.info("Proxy server stopped")
            return True

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False
            
    def create_asgi_app(self):
        """Create ASGI application handler."""
        handler = ASGIHandler(self)
        
        async def asgi_app(scope, receive, send):
            await handler(scope, receive, send)
            
        return asgi_app

    def get_analyzer(self) -> Optional[TrafficAnalyzer]:
        """Get the traffic analyzer instance."""
        return self._traffic_analyzer if self._is_running else None

    async def handle_request(self, scope: dict, receive: Callable, send: Callable) -> Optional[ProxyResponse]:
        """Handle an incoming request."""
        start_time = datetime.utcnow()
        
        if not self._is_running:
            return ProxyResponse(
                status_code=503,
                headers={'Content-Type': 'text/plain'},
                body=b'Proxy server is not running'
            )

        method = scope.get("method", "")
        # Skip if this is a CONNECT request - let ASGIHandler route it
        if method == "CONNECT":
            return None

        try:
            # Extract info for regular HTTP request
            method, host, path = self._extract_request_info(scope)
            if not host:
                return ProxyResponse(
                    status_code=400,
                    headers={'Content-Type': 'text/plain'},
                    body=b'Missing Host header'
                )

            # Check host restrictions if configured
            if self._config.allowed_hosts and host not in self._config.allowed_hosts:
                return ProxyResponse(
                    status_code=403,
                    headers={'Content-Type': 'text/plain'},
                    body=b'Host not allowed'
                )
            
            if self._config.excluded_hosts and host in self._config.excluded_hosts:
                return ProxyResponse(
                    status_code=403,
                    headers={'Content-Type': 'text/plain'},
                    body=b'Host is excluded'
                )

            request_id = str(uuid4())
            raw_headers = scope.get("headers", [])
            headers = {key.decode(): value.decode() for key, value in raw_headers}

            # Create request object
            request = InterceptedRequest(
                id=request_id,
                method=method,
                url=f"http://{host}{path}",
                headers=headers,
                body=b""
            )
            self._pending_requests[request_id] = request

            try:
                # Apply request interceptors if enabled
                if self._config.intercept_requests:
                    for interceptor in self._request_interceptors:
                        request = await interceptor.intercept(request)

                # Forward request
                async with self.session.request(request) as aiohttp_response:
                    response_body = await aiohttp_response.read()
                    
                    # Create intercepted response object
                    response = InterceptedResponse(
                        status_code=aiohttp_response.status,
                        headers=dict(aiohttp_response.headers),
                        body=response_body
                    )

                    # Apply response interceptors if enabled
                    if self._config.intercept_responses:
                        for interceptor in self._response_interceptors:
                            response = await interceptor.intercept(response, request)

                    # Store history in database if we have an active session
                    try:
                        async with AsyncSessionLocal() as db:
                            result = await db.execute(
                                text("SELECT * FROM proxy_sessions WHERE is_active = true ORDER BY start_time DESC LIMIT 1")
                            )
                            active_session = result.first()
                            if active_session:
                                from api.proxy.database_models import ProxyHistoryEntry
                                
                                end_time = datetime.utcnow()
                                duration = (end_time - start_time).total_seconds()
                                
                                history_entry = ProxyHistoryEntry(
                                    session_id=active_session.id,
                                    timestamp=start_time,
                                    method=request.method,
                                    url=request.url,
                                    request_headers=request.headers,
                                    request_body=request.body.decode('utf-8', errors='ignore') if request.body else None,
                                    response_status=response.status_code,
                                    response_headers=response.headers,
                                    response_body=response.body.hex() if response.body else None,
                                    duration=duration,
                                    is_intercepted=bool(self._request_interceptors or self._response_interceptors),
                                    tags=[],
                                    notes=None
                                )
                                
                                db.add(history_entry)
                                await db.commit()
                    except Exception as e:
                        logger.error(f"Failed to store proxy history: {e}", exc_info=True)
                    
                    # Send response through ASGI
                    return ProxyResponse(
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        body=response.body
                    )

            finally:
                # Cleanup
                if request_id in self._pending_requests:
                    del self._pending_requests[request_id]

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            return ProxyResponse(
                status_code=502,
                headers={'Content-Type': 'text/plain'},
                body=str(e).encode()
            )

    def _extract_request_info(self, scope: dict) -> Tuple[str, str, Optional[str]]:
        """Extract method, host and path from ASGI scope."""
        method = scope.get("method", "").upper()
        
        # Get headers from raw bytes
        headers = dict(scope.get("headers", []))
        host = headers.get(b"host", b"").decode()
        
        # Get path, ensuring it starts with /
        path = scope.get("path", "")
        if not path.startswith("/"):
            path = "/" + path
            
        return method, host, path

    async def _send_error(self, send: Callable, status: int, message: str) -> None:
        """Send an error response through the ASGI interface."""
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain"),
                (b"connection", b"close")
            ]
        })
        await send({
            "type": "http.response.body",
            "body": message.encode(),
        })

    async def handle_connect(self, scope: dict, receive: Callable, send: Callable) -> None:
        """Handle CONNECT tunnel request."""
        client_addr = scope.get('client', ('unknown', 0))[0]
        connection_id = str(uuid.uuid4())
        log_prefix = f"[{client_addr}] [{connection_id}]"
        target_socket = None

        try:
            # Parse target from scope
            logger.debug(f"{log_prefix} Raw target from scope: {scope['path']}")
            target = unquote(scope['path'])
            logger.debug(f"{log_prefix} URL-decoded target: {target}")

            try:
                host, port_str = target.split(':')
                logger.debug(f"{log_prefix} Split target into host: {host}, port_str: {port_str}")
                port = int(port_str)
                logger.debug(f"{log_prefix} Parsed port number: {port}")
            except ValueError as e:
                logger.error(f"{log_prefix} Invalid target format: {e}")
                await send({
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [(b"content-type", b"text/plain")]
                })
                await send({
                    "type": "http.response.body",
                    "body": b"Invalid target format",
                    "more_body": False
                })
                return

            logger.info(f"{log_prefix} Establishing tunnel to {host}:{port}")

            try:
                # Connect to target server
                target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                target_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                target_socket.setblocking(False)
                await asyncio.get_event_loop().sock_connect(target_socket, (host, port))
                logger.info(f"{log_prefix} Connected to {host}:{port}")

                # Get the raw socket from the ASGI scope
                transport = scope.get('extensions', {}).get('transport', None)
                if not transport:
                    raise RuntimeError("No transport found in ASGI scope")
                
                client_socket = transport.get_extra_info('socket')
                if not client_socket:
                    raise RuntimeError("Could not get client socket from transport")

                # Send 200 Connection Established
                logger.debug(f"{log_prefix} Sending 200 Connection Established")
                response = (
                    b"HTTP/1.1 200 Connection Established\r\n"
                    b"Connection: keep-alive\r\n"
                    b"Proxy-Agent: AnarchyProxy\r\n\r\n"
                )
                await asyncio.get_event_loop().sock_sendall(client_socket, response)

                # Set both sockets to non-blocking mode
                client_socket.setblocking(False)
                target_socket.setblocking(False)

                # Create tasks for bidirectional forwarding
                forward_client_task = asyncio.create_task(
                    self._forward_socket(client_socket, target_socket, 8192, f"{log_prefix} [client->target]")
                )
                forward_target_task = asyncio.create_task(
                    self._forward_socket(target_socket, client_socket, 8192, f"{log_prefix} [target->client]")
                )

                # Wait for either direction to complete
                done, pending = await asyncio.wait(
                    [forward_client_task, forward_target_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            except Exception as e:
                logger.error(f"{log_prefix} Error establishing tunnel: {e}")
                if not scope.get("response_started", False):
                    await send({
                        "type": "http.response.start",
                        "status": 502,
                        "headers": [(b"content-type", b"text/plain")]
                    })
                    await send({
                        "type": "http.response.body",
                        "body": str(e).encode(),
                        "more_body": False
                    })
                return

        except Exception as e:
            logger.error(f"{log_prefix} Error in handle_connect: {e}")
            if not scope.get("response_started", False):
                try:
                    await send({
                        "type": "http.response.start",
                        "status": 500,
                        "headers": [(b"content-type", b"text/plain")]
                    })
                    await send({
                        "type": "http.response.body",
                        "body": str(e).encode(),
                        "more_body": False
                    })
                except Exception as e2:
                    logger.error(f"{log_prefix} Failed to send error response: {e2}")

        finally:
            # Clean up resources
            if target_socket:
                try:
                    target_socket.close()
                except:
                    pass

    async def _forward_socket(self, src: socket.socket, dst: socket.socket, chunk_size: int, log_prefix: str) -> None:
        """Forward data between sockets."""
        loop = asyncio.get_event_loop()
        try:
            while True:
                try:
                    data = await loop.sock_recv(src, chunk_size)
                    if not data:
                        logger.debug(f"{log_prefix} Connection closed by sender")
                        break

                    await loop.sock_sendall(dst, data)
                    logger.debug(f"{log_prefix} Forwarded {len(data)} bytes")

                except ConnectionError as e:
                    logger.debug(f"{log_prefix} Connection error: {e}")
                    break

        except Exception as e:
            logger.error(f"{log_prefix} Error in socket forwarding: {e}")
            raise
        finally:
            # Shutdown the destination socket's write side
            try:
                dst.shutdown(socket.SHUT_WR)
            except:
                pass

    def _setup_interceptors(self) -> None:
        """Set up request and response interceptors."""
        if self._add_default_interceptors:
            # Add content encoding interceptor for handling compression
            self._response_interceptors.append(ContentEncodingInterceptor())
            
            # Add debug interceptor if debug mode is enabled
            if self._config.debug:
                debug_interceptor = DebugInterceptor(self._websocket_manager)
                self._request_interceptors.append(debug_interceptor)
                self._response_interceptors.append(debug_interceptor)
                
        # Add traffic analyzer interceptor
        self._request_interceptors.append(self._traffic_analyzer)
        self._response_interceptors.append(self._traffic_analyzer)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle incoming client connection."""
        client_addr = writer.get_extra_info('peername')
        connection_id = str(uuid.uuid4())
        log_prefix = f"[{client_addr}] [{connection_id}]"

        try:
            # Read the initial request line
            request_line = await reader.readline()
            if not request_line:
                logger.warning(f"{log_prefix} Empty request received")
                return

            # Parse the request line
            try:
                method, target, version = request_line.decode().strip().split(' ')
            except ValueError:
                logger.error(f"{log_prefix} Invalid request line: {request_line}")
                writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                await writer.drain()
                return

            # Read headers
            headers = {}
            while True:
                line = await reader.readline()
                if line == b'\r\n' or not line:
                    break
                try:
                    name, value = line.decode().strip().split(':', 1)
                    headers[name.strip().lower()] = value.strip()
                except ValueError:
                    logger.warning(f"{log_prefix} Invalid header line: {line}")
                    continue

            # Handle CONNECT requests differently
            if method == 'CONNECT':
                await self._handle_connect_request(target, reader, writer, headers, log_prefix)
            else:
                await self._handle_http_request(method, target, version, headers, reader, writer, log_prefix)

        except Exception as e:
            logger.error(f"{log_prefix} Error handling client: {e}", exc_info=True)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass

    async def _handle_connect_request(self, target: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, headers: dict, log_prefix: str) -> None:
        """Handle CONNECT tunnel request."""
        try:
            host, port_str = target.split(':')
            port = int(port_str)
        except ValueError:
            logger.error(f"{log_prefix} Invalid CONNECT target: {target}")
            writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            await writer.drain()
            return

        try:
            # Connect to target server
            target_reader, target_writer = await asyncio.open_connection(host, port)
            logger.info(f"{log_prefix} Connected to {host}:{port}")

            # Send 200 Connection Established
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()

            # Create tasks for bidirectional forwarding
            client_to_target = asyncio.create_task(self._forward_stream(reader, target_writer, f"{log_prefix} [client->target]"))
            target_to_client = asyncio.create_task(self._forward_stream(target_reader, writer, f"{log_prefix} [target->client]"))

            # Wait for either direction to complete
            done, pending = await asyncio.wait(
                [client_to_target, target_to_client],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.error(f"{log_prefix} Error in CONNECT tunnel: {e}")
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await writer.drain()

    async def _handle_http_request(self, method: str, target: str, version: str, headers: dict, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, log_prefix: str) -> None:
        """Handle regular HTTP request."""
        try:
            # Parse target URL
            if not target.startswith('http://'):
                target = f"http://{headers.get('host', '')}{target}"

            parsed = urlparse(target)
            if not parsed.netloc:
                logger.error(f"{log_prefix} Invalid target URL: {target}")
                writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                await writer.drain()
                return

            # Read request body if present
            body = b""
            content_length = headers.get('content-length')
            if content_length:
                try:
                    body = await reader.read(int(content_length))
                except ValueError:
                    logger.warning(f"{log_prefix} Invalid content-length: {content_length}")

            # Create request object
            request = InterceptedRequest(
                id=str(uuid.uuid4()),
                method=method,
                url=target,
                headers=headers,
                body=body
            )

            # Apply request interceptors
            if self._config.intercept_requests:
                for interceptor in self._request_interceptors:
                    request = await interceptor.intercept(request)

            # Forward request to target
            async with self.session.request(request) as response:
                # Create response object
                response_obj = InterceptedResponse(
                    status_code=response.status,
                    headers=dict(response.headers),
                    body=await response.read()
                )

                # Apply response interceptors
                if self._config.intercept_responses:
                    for interceptor in self._response_interceptors:
                        response_obj = await interceptor.intercept(response_obj, request)

                # Send response to client
                status_line = f"HTTP/1.1 {response_obj.status_code} {response.reason}\r\n"
                writer.write(status_line.encode())
                
                for name, value in response_obj.headers.items():
                    header_line = f"{name}: {value}\r\n"
                    writer.write(header_line.encode())
                
                writer.write(b"\r\n")
                writer.write(response_obj.body)
                await writer.drain()

        except Exception as e:
            logger.error(f"{log_prefix} Error handling HTTP request: {e}")
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await writer.drain()

    async def _forward_stream(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, log_prefix: str) -> None:
        """Forward data between streams."""
        try:
            while True:
                data = await reader.read(8192)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
                logger.debug(f"{log_prefix} Forwarded {len(data)} bytes")
        except Exception as e:
            logger.error(f"{log_prefix} Error in stream forwarding: {e}")
            raise
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass
