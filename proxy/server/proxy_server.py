"""Main proxy server implementation."""
import asyncio
import logging
import socket
import ssl
import signal
import os
import time
import psutil
import sys
import uuid
import uvicorn
from contextlib import AsyncExitStack
from pathlib import Path
from async_timeout import timeout
from typing import Dict, List, Optional, Any, Tuple, Callable, TYPE_CHECKING, cast, NamedTuple
from typing_extensions import TypedDict, Protocol
from urllib.parse import urlparse, unquote
from uuid import uuid4
from sqlalchemy import text
from datetime import datetime
from proxy.session import get_active_sessions

# Force use of standard asyncio event loop
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

from sqlalchemy.ext.asyncio import AsyncSession
from database import AsyncSessionLocal
from ..config import ProxyConfig
from .handlers import ProxyResponse

if TYPE_CHECKING:
    from api.proxy.database_models import ProxyHistoryEntry, ProxySession as DBProxySession

from .custom_protocol import TunnelProtocol
from .protocol.https_intercept import HttpsInterceptProtocol
from .protocol import HttpsInterceptProtocol
from ..interceptor import RequestInterceptor, ResponseInterceptor, InterceptedRequest, InterceptedResponse
from ..encoding import ContentEncodingInterceptor
from ..session import ProxySession
from ..websocket import WebSocketManager, DebugInterceptor
from ..analysis.analyzer import TrafficAnalyzer
from .certificates import CertificateAuthority
from .tls_helper import cert_manager, CertificateManager
from .handlers.asgi_handler import ASGIHandler
from typing import Dict, List, Optional, Any, Tuple, Callable, TYPE_CHECKING, cast
from typing_extensions import TypedDict
from proxy.interceptors.database import DatabaseInterceptor
from .flow_control import FlowControl

class MemoryStats(TypedDict):
    """Memory statistics type."""
    samples: List[float]
    timestamps: List[float]
    total: int
    used: int
    free: int

class ProxyStats(TypedDict):
    """Proxy statistics type."""
    total_requests: int
    active_connections: int
    bytes_sent: int
    bytes_received: int
    memory_usage: float
    cpu_usage: float

logger = logging.getLogger("proxy.core")

_instance = None

# Module-level configure function to fix initialization error
def configure(config_dict: dict) -> None:
    """Configure or reconfigure the proxy server instance at the module level."""
    ProxyServer.configure(config_dict)

class SocketAddress(NamedTuple):
    """Socket address tuple type."""
    host: str
    port: int

class CertificateAuthorityProtocol(Protocol):
    """Protocol defining certificate authority interface."""
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def close(self) -> None: ...

class BaseStats(TypedDict, total=False):
    """Base statistics type."""
    memory: MemoryStats
    proxy: ProxyStats

class ProxyServer:
    """Proxy server implementation."""

    def __init__(self, config: ProxyConfig, ca_instance=None):
        """Initialize proxy server with configuration and optional CA instance."""
        self._config = config
        self.ca = ca_instance
        self.loop = None
        self.server = None
        self.port = None
        self._active_connections = {}
        self._interceptors = []
        self._tasks = set()
        self._is_running = False
        self._session = None
        self._serve_task = None
        self._stats = {
            'total_requests': 0,
            'active_connections': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0
        }
        self._memory = {
            'samples': [],
            'timestamps': [],
            'total': 0,
            'used': 0,
            'free': 0
        }
        self._base_stats = {
            'memory': self._memory,
            'proxy': self._stats
        }
        self._setup_logging()
        self._setup_interceptors()

    def _setup_interceptors(self):
        """Set up proxy interceptors."""
        logger.debug("Setting up interceptors")
        try:
            # Add global interceptors
            self._interceptors.append(ContentEncodingInterceptor())
            logger.debug("Added ContentEncodingInterceptor")
            
            # Add database interceptor
            from proxy.interceptors.database import DatabaseInterceptor
            self._interceptors.append(DatabaseInterceptor(str(uuid.uuid4())))
            logger.debug("Added DatabaseInterceptor")
        except Exception as e:
            logger.error(f"Error setting up interceptors: {e}", exc_info=True)
            raise

    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming connection."""
        connection_id = str(uuid.uuid4())
        logger.debug(f"New connection {connection_id}")
        
        try:
            # Read the initial request line
            request_line = await reader.readline()
            if not request_line:
                logger.warning(f"[{connection_id}] Empty request received")
                writer.close()
                return

            # Parse request line
            try:
                method, target, version = request_line.decode().strip().split(' ')
                logger.debug(f"[{connection_id}] {method} {target} {version}")
            except Exception as e:
                logger.error(f"[{connection_id}] Failed to parse request line: {e}")
                writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                writer.close()
                return

            # For CONNECT requests, use HTTPS interception protocol
            if method == 'CONNECT':
                try:
                    # Parse target host and port
                    target_host, target_port = target.split(':')
                    target_port = int(target_port)
                    
                    # Create protocol instance with connection-specific settings
                    protocol = HttpsInterceptProtocol(
                        connection_id=connection_id,
                        target_host=target_host,
                        target_port=target_port,
                        ca=self.ca,
                        cert_manager=cert_manager,
                        db_interceptor=None,  # Will be created when needed
                        client_transport=writer.transport
                    )
                    
                    # Store connection info
                    self._active_connections[connection_id] = {
                        'reader': reader,
                        'writer': writer,
                        'protocol': protocol,
                        'start_time': time.time()
                    }
                    
                    # Set up the transport for the protocol
                    protocol.connection_made(writer.transport)
                    
                    # Process the CONNECT request
                    protocol.data_received(request_line)
                    
                except Exception as e:
                    logger.error(f"[{connection_id}] Error handling CONNECT request: {e}")
                    writer.write(b"HTTP/1.1 500 Internal Server Error\r\n\r\n")
                    writer.close()
                    return
                    
            else:
                # For non-CONNECT requests, use regular HTTP handling
                try:
                    # Read headers
                    headers = {}
                    while True:
                        line = await reader.readline()
                        if line in (b'\r\n', b''):
                            break
                        try:
                            name, value = line.decode().split(':', 1)
                            headers[name.strip().lower()] = value.strip()
                        except ValueError:
                            continue

                    # Create protocol instance
                    protocol = TunnelProtocol(
                        connection_id=connection_id,
                        client_transport=writer.transport
                    )
                    
                    # Store connection info
                    self._active_connections[connection_id] = {
                        'reader': reader,
                        'writer': writer,
                        'protocol': protocol,
                        'start_time': time.time()
                    }
                    
                    # Process the request
                    await self._handle_regular_request(method, target, version, headers, reader, writer, f"[{connection_id}]")
                    
                except Exception as e:
                    logger.error(f"[{connection_id}] Error handling regular request: {e}")
                    writer.write(b"HTTP/1.1 500 Internal Server Error\r\n\r\n")
                    writer.close()
                    return
            
            # Process the connection
            await self._process_connection(connection_id, reader, writer, protocol)
            
        except Exception as e:
            logger.error(f"[{connection_id}] Error handling connection: {e}")
            writer.close()
            await writer.wait_closed()
            
        finally:
            if connection_id in self._active_connections:
                del self._active_connections[connection_id]

    def _setup_logging(self):
        """Set up logging configuration."""
        # Configure root logger if not already configured
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
            root_logger.setLevel(logging.INFO)

        # Configure proxy logger
        proxy_logger = logging.getLogger("proxy.core")
        proxy_logger.setLevel(logging.DEBUG)

    async def _process_connection(self, connection_id: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, protocol: Any) -> None:
        """Process a client connection."""
        try:
            # Set up protocol
            protocol.transport = writer.transport
            
            # Process data
            while True:
                try:
                    data = await reader.read(8192)  # 8KB chunks
                    if not data:
                        logger.debug(f"Connection {connection_id} closed by peer")
                        break
                        
                    # Process data through protocol
                    protocol.data_received(data)
                    
                except ConnectionError as e:
                    logger.debug(f"Connection {connection_id} error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Error processing connection {connection_id}: {e}", exc_info=True)
            
        finally:
            # Clean up
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass
            
            if connection_id in self._active_connections:
                del self._active_connections[connection_id]
                
            logger.debug(f"Connection {connection_id} closed")

    @classmethod
    def configure(cls, config_dict: dict) -> None:
        """Configure the proxy server with the given settings."""
        logger.debug("Configuring proxy server")
        logger.debug(f"Configuration: {config_dict}")
        global _instance
        try:
            if (_instance is None):
                logger.debug("Creating new ProxyServer instance")
                try:
                    logger.debug("Creating ProxyConfig from config_dict")
                    config = ProxyConfig(**config_dict)
                    logger.debug(f"Created ProxyConfig: {config}")
                    
                    logger.debug("Creating ProxyServer instance")
                    _instance = cls(config)
                    logger.debug("Successfully created ProxyServer instance")
                except Exception as e:
                    logger.error(f"Failed to create ProxyServer instance: {e}", exc_info=True)
                    raise
            else:
                logger.debug("Updating existing ProxyServer instance configuration")
                try:
                    logger.debug("Creating new ProxyConfig from config_dict")
                    config = ProxyConfig(**config_dict)
                    logger.debug(f"Created ProxyConfig: {config}")
                    
                    logger.debug("Updating ProxyServer configuration")
                    _instance._config = config
                    logger.debug("Successfully updated ProxyServer configuration")
                except Exception as e:
                    logger.error(f"Failed to update ProxyServer configuration: {e}", exc_info=True)
                    raise
        except Exception as e:
            logger.error(f"Error in configure method: {e}", exc_info=True)
            raise

    @property
    def config(self) -> ProxyConfig:
        """Get the current configuration."""
        return self._config
    
    @config.setter 
    def config(self, value: ProxyConfig) -> None:
        """Set new configuration."""
        self._config = value

    def create_task(self, coro) -> asyncio.Task:
        """Create a task in the server's event loop."""
        if not self.loop:
            raise RuntimeError("Server not started - no event loop available")
            
        task = self.loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def start(self) -> None:
        """Start the proxy server."""
        try:
            # Check for active sessions first
            active_sessions = await get_active_sessions()
            if not active_sessions:
                logger.warning("No active proxy sessions found. Some functionality may be limited.")
            else:
                logger.info(f"Found {len(active_sessions)} active proxy sessions")

            logger.debug("Setting up server components")
            # Set up interceptors
            try:
                logger.debug("Initializing interceptors")
                self._setup_interceptors()
                logger.debug("Interceptors initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize interceptors: {e}", exc_info=True)
                raise

            # Verify CA is available
            # Initialize CA and certificate manager
            if not self.ca:
                logger.error("No Certificate Authority provided. HTTPS interception will be disabled.")
                raise RuntimeError("Certificate Authority is required for proxy operation")

            logger.info("Starting Certificate Authority initialization...")
            
            # Initialize certificate manager first
            if cert_manager.is_running():
                logger.info("Stopping existing certificate manager...")
                await cert_manager.stop()

            # Set CA in certificate manager
            logger.info("Configuring certificate manager with CA...")
            cert_manager.set_ca(self.ca)

            # Start certificate manager - this will also initialize the CA
            logger.info("Starting certificate manager...")
            max_retries = 3
            retry_delay = 1.0
            last_error = None

            for attempt in range(max_retries):
                try:
                    await cert_manager.start()
                    
                    # Verify initialization
                    health = cert_manager.get_health()
                    if health.status != "healthy":
                        raise RuntimeError(f"Certificate manager health check failed: {health.status}\nDetails: {health.details}")
                    
                    logger.info("Certificate manager started successfully")
                    break

                except Exception as e:
                    last_error = e
                    logger.error(f"Initialization attempt {attempt + 1} failed: {e}")
                    
                    if cert_manager.is_running():
                        await cert_manager.stop()
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.error("All initialization attempts failed")
                        self.ca = None
                        raise RuntimeError(f"Failed to initialize certificate handling after {max_retries} attempts: {last_error}")

            logger.info("Certificate handling initialization completed successfully")

            # Create server
            self.server = await asyncio.start_server(
                self.handle_connection,
                self._config.host,
                self._config.port,
                reuse_address=True,
                reuse_port=True
            )
            
            # Store port for cleanup
            self.port = self._config.port
            
            logger.info(f"Proxy server started on {self._config.host}:{self._config.port}")
            
            # Get or create event loop
            self.loop = asyncio.get_running_loop()
            
            # Create and monitor server task
            serve_task = self.loop.create_task(self.server.serve_forever())
            
            def _on_server_task_done(task: asyncio.Task) -> None:
                self._tasks.discard(task)
                try:
                    # Re-raise any exception that occurred
                    exc = task.exception()
                    if exc is not None and not isinstance(exc, asyncio.CancelledError):
                        logger.error(f"Server task failed: {exc}")
                        # Stop the server if task failed
                        asyncio.create_task(self.stop())
                except asyncio.CancelledError:
                    pass
            
            serve_task.add_done_callback(_on_server_task_done)
            self._tasks.add(serve_task)
            
            # Store server task for cleanup
            self._serve_task = serve_task
            
            # Set up signal handlers for graceful shutdown
            def make_signal_handler(sig: signal.Signals) -> Callable[[], None]:
                def handler() -> None:
                    logger.info(f"Received signal {sig.name}, initiating shutdown...")
                    asyncio.create_task(self._handle_signal(sig))
                return handler
                
            for sig in (signal.SIGTERM, signal.SIGINT):
                self.loop.add_signal_handler(sig, make_signal_handler(sig))
            
            # Set running flag
            self._is_running = True
            
            logger.info("Proxy server is running in background")
            
        except Exception as e:
            logger.error(f"Failed to start proxy server: {e}")
            raise

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle incoming client connection."""
        client_addr = writer.get_extra_info('peername')
        connection_id = str(uuid.uuid4())
        log_prefix = f"[{client_addr}] [{connection_id}]"

        try:
            # Log initial connection state
            logger.debug(f"{log_prefix} New client connection")
            logger.debug(f"{log_prefix} CA instance available: {self.ca is not None}")
            if self.ca:
                logger.debug(f"{log_prefix} CA certificate path: {self.ca.ca_cert_path}")
                logger.debug(f"{log_prefix} CA key path: {self.ca.ca_key_path}")
                logger.debug(f"{log_prefix} CA certificate exists: {self.ca.ca_cert_path.exists()}")
                logger.debug(f"{log_prefix} CA key exists: {self.ca.ca_key_path.exists()}")
            
            self._stats['active_connections'] += 1
            
            request_line = await reader.readline()
            if not request_line:
                logger.warning(f"{log_prefix} Empty request received")
                return

            method, target, version = request_line.decode().strip().split(' ')
            logger.info(f"{log_prefix} {method} {target} {version}")

            headers = {}
            while True:
                header_line = await reader.readline()
                if header_line in (b'\r\n', b''): 
                    break
                
                try:
                    name, value = header_line.decode().split(':', 1)
                    headers[name.strip().lower()] = value.strip()
                except ValueError:
                    continue

            # Log request details
            logger.debug(f"{log_prefix} Request headers: {headers}")
            logger.debug(f"{log_prefix} Certificate manager running: {cert_manager.is_running()}")
            if cert_manager.is_running():
                health = cert_manager.get_health()
                logger.debug(f"{log_prefix} Certificate manager health: {health.status}")
                logger.debug(f"{log_prefix} Certificate manager details: {health.details}")

            # Handle CONNECT tunneling differently
            if method == 'CONNECT':
                logger.debug(f"{log_prefix} Processing CONNECT request")
                await self._handle_connect_tunnel(request_line, reader, writer)
            else:
                logger.debug(f"{log_prefix} Processing regular request")
                await self._handle_regular_request(method, target, version, headers, reader, writer, log_prefix)

        except Exception as e:
            logger.error(f"{log_prefix} Request error: {e}", exc_info=True)
            if writer and not writer.is_closing():
                writer.close()
                await writer.wait_closed()
        finally:
            self._stats['active_connections'] -= 1

    async def _handle_connect_tunnel(self, request_line: bytes, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle CONNECT tunnel request with TLS interception."""
        try:
            # Parse target host and port
            target = request_line.decode().split()[1]
            target_host, target_port = target.split(':')
            target_port = int(target_port)
            
            # Generate unique connection ID
            connection_id = str(uuid.uuid4())
            logger.debug(f"New connection {connection_id}")
            
            # Create database interceptor
            db_interceptor = DatabaseInterceptor(connection_id)
            
            # Create protocol
            protocol = HttpsInterceptProtocol(
                connection_id=connection_id,
                target_host=target_host,
                target_port=target_port,
                ca=self.ca,
                cert_manager=cert_manager,
                db_interceptor=db_interceptor,
                client_transport=writer.transport
            )
            
            # Store connection info
            self._active_connections[connection_id] = {
                'reader': reader,
                'writer': writer,
                'protocol': protocol
            }
            
            # Send 200 Connection Established
            writer.write(b'HTTP/1.1 200 Connection Established\r\n\r\n')
            await writer.drain()
            
            # Establish tunnel and set up TLS
            await protocol.establish_tunnel(target_host, target_port)
            
            # Process connection
            await self._process_connection(connection_id, reader, writer, protocol)
            
        except Exception as e:
            logger.error(f"Error handling CONNECT tunnel: {e}", exc_info=True)
            writer.write(b'HTTP/1.1 502 Bad Gateway\r\n\r\n')
            await writer.drain()
            writer.close()

    async def _handle_regular_request(self, method: str, target: str, version: str,
                                    headers: dict, reader: asyncio.StreamReader,
                                    writer: asyncio.StreamWriter, log_prefix: str) -> None:
        """Handle regular HTTP requests."""
        try:
            # Build full URL if needed
            if not target.startswith(('http://', 'https://')):
                host = headers.get('host', '')
                target = f"http://{host}{target}"

            # Read request body if present
            body = b''
            if 'content-length' in headers:
                content_length = int(headers['content-length'])
                body = await reader.read(content_length)

            # Update stats
            self._stats['total_requests'] += 1
            self._stats['bytes_sent'] += len(body)

            # Create request object
            request = InterceptedRequest(
                id=str(uuid.uuid4()),
                method=method,
                url=target,
                headers=headers,
                body=body
            )

            # Create database interceptor for this request
            connection_id = str(uuid.uuid4())
            db_interceptor = DatabaseInterceptor(connection_id)

            # Run request through interceptors
            try:
                await db_interceptor.intercept_request(request)
            except Exception as e:
                logger.error(f"{log_prefix} Request interceptor error: {str(e)}", exc_info=True)

            # Forward request and get response
            async with self.session.request(request) as response:
                response_body = await response.read()
                
                # Create intercepted response
                intercepted_response = InterceptedResponse(
                    status_code=response.status,
                    headers=dict(response.headers),
                    body=response_body
                )

                # Run response through interceptors
                try:
                    await db_interceptor.intercept_response(intercepted_response, request)
                except Exception as e:
                    logger.error(f"{log_prefix} Response interceptor error: {str(e)}", exc_info=True)

                # Clean up interceptor
                await db_interceptor.close()
                
                # Send response back to client
                status_line = f"HTTP/1.1 {response.status} {response.reason}\r\n"
                writer.write(status_line.encode())
                
                for name, value in response.headers.items():
                    writer.write(f"{name}: {value}\r\n".encode())
                
                writer.write(b"\r\n")
                if response_body:
                    writer.write(response_body)
                    self._stats['bytes_received'] += len(response_body)
                
                await writer.drain()

        except Exception as e:
            logger.error(f"{log_prefix} Request error: {str(e)}", exc_info=True)
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await writer.drain()

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current server statistics.
        
        Returns:
            Dict[str, Any]: Current server statistics
        """
        return dict(self._stats)  # Convert TypedDict to regular dict

    @property
    def session(self) -> ProxySession:
        """Get or create proxy session."""
        if self._session is None:
            self._session = ProxySession()
        return self._session

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signals gracefully.
        
        Args:
            sig: The signal received (SIGTERM or SIGINT)
        """
        logger.info(f"Received signal {sig.name}, initiating graceful shutdown...")
        try:
            await self.stop()
        except Exception as e:
            logger.error(f"Error during signal-triggered shutdown: {e}")
            
    async def _force_cleanup_port(self) -> None:
        """Force cleanup any processes using the proxy port."""
        if not self.port:
            return

        async def kill_process(pid: int, force: bool = False) -> None:
            try:
                proc = psutil.Process(pid)
                if force:
                    proc.kill()
                else:
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Extract port number from connections
        def get_port_number(conn: Any) -> Optional[int]:
            try:
                if hasattr(conn, 'laddr'):
                    addr = getattr(conn.laddr, '_asdict', lambda: {'port': None})()
                    return addr.get('port')
                return None
            except Exception:
                return None

        # Find and kill processes using the port
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.connections():
                    port = get_port_number(conn)
                    if port == self.port:
                        await kill_process(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    async def _stop_certificate_authority(self) -> None:
        """Stop the certificate authority."""
        # Stop CA with graceful fallback
        if self.ca:
            try:
                # Try cleanup/stop methods in sequence
                for method in ['cleanup_old_certs', 'stop', 'close']:
                    if hasattr(self.ca, method):
                        try:
                            await getattr(self.ca, method)()
                        except Exception as e:
                            logger.warning(f"CA {method}() failed: {e}")
            except Exception as e:
                logger.error(f"Error stopping CA: {e}")
            finally:
                self.ca = None

        # Cancel all tasks
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error cancelling task: {e}")
        self._tasks.clear()

    async def stop(self) -> None:
        """Stop the proxy server and cleanup resources."""
        if not self._is_running:
                return

        try:
            logger.info("Stopping proxy server...")
            self._is_running = False

            # Stop servers
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                self.server = None

            # Stop certificate manager
            if cert_manager and cert_manager.is_running():
                await cert_manager.stop()

            # Stop CA with graceful fallback
            if self.ca:
                try:
                    # Try cleanup/stop methods in sequence
                    for method in ['cleanup_old_certs', 'stop', 'close']:
                        if hasattr(self.ca, method):
                            try:
                                await getattr(self.ca, method)()
                            except Exception as e:
                                logger.warning(f"CA {method}() failed: {e}")
                except Exception as e:
                    logger.error(f"Error stopping CA: {e}")
                self.ca = None

            # Cancel all tasks
            for task in self._tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self._tasks.clear()

            # Close session
            if self._session:
                await self._session.close()
                self._session = None

            # Clean up port
            await self._force_cleanup_port()
            
            logger.info("Proxy server stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping proxy server: {e}")
            raise
        finally:
            self._is_running = False

    def get_proxy_stats(self) -> ProxyStats:
        """Get proxy-specific statistics."""
        stats = self._stats.copy()
        stats['memory_usage'] = float(psutil.virtual_memory().percent)
        stats['cpu_usage'] = float(psutil.cpu_percent())
        return stats

    def get_memory_stats(self) -> MemoryStats:
        """Get memory statistics."""
        return self._memory.copy()

    def get_base_stats(self) -> BaseStats:
        """Get complete statistics including memory and proxy stats."""
        stats = self._base_stats.copy()
        stats['proxy'] = self.get_proxy_stats()
        stats['memory'] = self.get_memory_stats()
        return stats

    def get_stats(self) -> Dict[str, Any]:
        """Get legacy statistics format."""
        return {
            'proxy': self.get_proxy_stats(),
            'memory': self.get_memory_stats()
        }


    def add_memory_sample(self, sample: float) -> None:
        """Add a memory usage sample."""
        self._memory['samples'].append(sample)
        self._memory['timestamps'].append(time.time())

    def get_memory_deltas(self, window_size: int = 10) -> List[float]:
        """Get memory usage deltas over specified window."""
        samples = self._memory['samples']
        if len(samples) < 2:
            return []

        window = samples[-window_size:] if window_size > 0 else samples
        return [window[i] - window[i-1] for i in range(1, len(window))]

    def _create_tunnel_protocol(self, client_transport: asyncio.Transport, connection_id: Optional[str] = None) -> TunnelProtocol:
        """Create a new tunnel protocol instance."""
        # Always ensure we have a connection_id
        if connection_id is None:
            connection_id = str(uuid4())
            logger.debug(f"Generated new connection ID: {connection_id}")
            
        flow_control = FlowControl(client_transport)
        return TunnelProtocol(
            connection_id=connection_id,  # Now we're guaranteed to have a value
            client_transport=client_transport,
            flow_control=flow_control,
            interceptor_class=DatabaseInterceptor,
            buffer_size=self._config.buffer_size,
            metrics_interval=self._config.metrics_interval,
            write_limit=self._config.write_limit,
            write_interval=self._config.write_interval
        )
