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
    """HTTPS-capable proxy server with certificate management."""

    def __init__(self, config: Optional[ProxyConfig] = None, ca_instance: Optional[CertificateAuthority] = None):
        """Initialize proxy server.
        
        Args:
            config: Optional proxy configuration
            ca_instance: Optional pre-initialized Certificate Authority instance
        """
        self._config = config or ProxyConfig()
        logger.info(f"Initializing proxy server with config: {self._config}")
        
        # Initialize server components
        self._server = None
        self._is_running = False
        self._lock = asyncio.Lock()
        self._ca = ca_instance
        self._tasks: set[asyncio.Task] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._port: Optional[int] = None
        
        # Initialize request/response tracking
        self._pending_requests: Dict[str, Any] = {}
        self._pending_responses: Dict[str, Any] = {}
        
        # Initialize interceptors
        self._request_interceptors: List[RequestInterceptor] = []
        self._response_interceptors: List[ResponseInterceptor] = []
        self._analyzer: Optional[TrafficAnalyzer] = None
        self._add_default_interceptors = True
        
        # Initialize session
        self._session: Optional[ProxySession] = None
        
        # Initialize memory tracking
        self._memory: MemoryStats = {
            'samples': [],
            'timestamps': [],
            'total': 0,
            'used': 0,
            'free': 0
        }
        
        # Initialize stats
        self._stats: ProxyStats = {
            'total_requests': 0,
            'active_connections': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0
        }
        
        # Initialize base stats
        self._base_stats: BaseStats = {
            'memory': self._memory,
            'proxy': self._stats
        }

    @classmethod
    def configure(cls, config_dict: dict) -> None:
        """Configure the proxy server with the given settings."""
        global _instance
        if (_instance is None):
            _instance = cls(ProxyConfig(**config_dict))
        else:
            _instance.config = ProxyConfig(**config_dict)

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
        if not self._loop:
            raise RuntimeError("Server not started - no event loop available")
            
        task = self._loop.create_task(coro)
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

            # Set up interceptors
            self._setup_interceptors()

            # Verify CA is available
            # Initialize CA and certificate manager
            if not self._ca:
                logger.error("No Certificate Authority provided. HTTPS interception will be disabled.")
                raise RuntimeError("Certificate Authority is required for proxy operation")

            logger.info("Starting Certificate Authority initialization...")
            
            # Initialize certificate manager first
            if cert_manager.is_running():
                logger.info("Stopping existing certificate manager...")
                await cert_manager.stop()

            # Set CA in certificate manager
            logger.info("Configuring certificate manager with CA...")
            cert_manager.set_ca(self._ca)

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
                        self._ca = None
                        raise RuntimeError(f"Failed to initialize certificate handling after {max_retries} attempts: {last_error}")

            logger.info("Certificate handling initialization completed successfully")

            # Create server
            self._server = await asyncio.start_server(
                self._handle_client,
                self.config.host,
                self.config.port,
                reuse_address=True,
                reuse_port=True
            )
            
            # Store port for cleanup
            self._port = self.config.port
            
            logger.info(f"Proxy server started on {self.config.host}:{self.config.port}")
            
            # Get or create event loop
            self._loop = asyncio.get_running_loop()
            
            # Create and monitor server task
            serve_task = self._loop.create_task(self._server.serve_forever())
            
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
                self._loop.add_signal_handler(sig, make_signal_handler(sig))
            
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
            logger.debug(f"{log_prefix} CA instance available: {self._ca is not None}")
            if self._ca:
                logger.debug(f"{log_prefix} CA certificate path: {self._ca.ca_cert_path}")
                logger.debug(f"{log_prefix} CA key path: {self._ca.ca_key_path}")
                logger.debug(f"{log_prefix} CA certificate exists: {self._ca.ca_cert_path.exists()}")
                logger.debug(f"{log_prefix} CA key exists: {self._ca.ca_key_path.exists()}")
            
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
                await self._handle_connect_tunnel(target, reader, writer, headers, log_prefix)
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

    async def _handle_connect_tunnel(self, target: str, reader: asyncio.StreamReader, 
                                   writer: asyncio.StreamWriter, headers: dict,
                                   log_prefix: str) -> None:
        """Handle CONNECT tunnel requests."""
        try:
            host, port_str = target.split(':')
            port = int(port_str)
            logger.debug(f"{log_prefix} Parsed CONNECT target - Host: {host}, Port: {port}")
        except ValueError as e:
            logger.error(f"{log_prefix} Invalid CONNECT target: {target}")
            writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            await writer.drain()
            return

        try:
            # Log connection attempt
            logger.debug(f"{log_prefix} Attempting to connect to {host}:{port}")
            logger.debug(f"{log_prefix} Using CA: {self._ca is not None}")
            if self._ca:
                logger.debug(f"{log_prefix} CA certificate status: {self._ca.ca_cert_path.exists()}")
                logger.debug(f"{log_prefix} CA key status: {self._ca.ca_key_path.exists()}")
            
            # Connect to target
            target_reader, target_writer = await asyncio.open_connection(host, port)
            logger.debug(f"{log_prefix} Successfully connected to target")

            # Send 200 Connection Established
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            logger.debug(f"{log_prefix} Sent Connection Established response")

            # Create database interceptor for this connection
            connection_id = str(uuid.uuid4())
            db_interceptor = DatabaseInterceptor(connection_id)

            # Create bidirectional tunnel
            logger.debug(f"{log_prefix} Creating bidirectional tunnel")
            await self._create_tunnel(
                (reader, writer),
                (target_reader, target_writer),
                log_prefix,
                db_interceptor
            )
            logger.debug(f"{log_prefix} Tunnel closed")
            
            # Clean up interceptor
            await db_interceptor.close()
            
        except Exception as e:
            logger.error(f"{log_prefix} Tunnel error: {str(e)}", exc_info=True)
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await writer.drain()

    async def _create_tunnel(self, client: Tuple[asyncio.StreamReader, asyncio.StreamWriter],
                           target: Tuple[asyncio.StreamReader, asyncio.StreamWriter],
                           log_prefix: str,
                           db_interceptor: Optional[DatabaseInterceptor] = None) -> None:
        """Create bidirectional tunnel between client and target."""
        client_reader, client_writer = client
        target_reader, target_writer = target
        logger.debug(f"{log_prefix} Setting up tunnel forwarding")

        async def forward(reader: asyncio.StreamReader, 
                        writer: asyncio.StreamWriter,
                        direction: str) -> None:
            """Forward data between endpoints."""
            try:
                logger.debug(f"{log_prefix} Starting {direction} forwarding")
                bytes_forwarded = 0
                while True:
                    data = await reader.read(8192)
                    if not data:
                        logger.debug(f"{log_prefix} [{direction}] End of stream")
                        break

                    # Store data in database if interceptor is available
                    if db_interceptor:
                        try:
                            await db_interceptor.store_raw_data(direction, data)
                        except Exception as e:
                            logger.error(f"{log_prefix} [{direction}] Error storing data: {e}")

                    await writer.drain()
                    writer.write(data)
                    bytes_forwarded += len(data)
                    
                    # Update stats
                    if direction == "client->target":
                        self._stats['bytes_sent'] += len(data)
                    else:
                        self._stats['bytes_received'] += len(data)
                        
                    # Log progress periodically
                    if bytes_forwarded % (1024 * 1024) == 0:  # Log every MB
                        logger.debug(f"{log_prefix} [{direction}] Forwarded {bytes_forwarded/1024/1024:.2f} MB")
                        
            except Exception as e:
                logger.error(f"{log_prefix} [{direction}] Forward error: {str(e)}", exc_info=True)
            finally:
                try:
                    logger.debug(f"{log_prefix} [{direction}] Closing writer")
                    writer.close()
                    await writer.wait_closed()
                except:
                    pass

        # Create tasks for both directions
        logger.debug(f"{log_prefix} Creating forwarding tasks")
        forward_client = asyncio.create_task(
            forward(client_reader, target_writer, "client->target")
        )
        forward_target = asyncio.create_task(
            forward(target_reader, client_writer, "target->client")
        )

        # Wait for either direction to complete
        logger.debug(f"{log_prefix} Waiting for tunnel completion")
        done, pending = await asyncio.wait(
            [forward_client, forward_target],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Log completion status
        for task in done:
            try:
                exc = task.exception()
                if exc:
                    logger.error(f"{log_prefix} Task failed: {exc}", exc_info=exc)
            except asyncio.CancelledError:
                logger.debug(f"{log_prefix} Task was cancelled")
                
        # Cancel any pending tasks
        for task in pending:
            logger.debug(f"{log_prefix} Cancelling pending task")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
        logger.debug(f"{log_prefix} Tunnel forwarding completed")

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

    def _setup_interceptors(self) -> None:
        """Set up default interceptors."""
        if self._add_default_interceptors:
            # Add content encoding interceptor
            self._response_interceptors.append(ContentEncodingInterceptor())
            
            # Add debug interceptor if in debug mode
            if getattr(self._config, 'debug', False):
                debug_interceptor = DebugInterceptor()
                if isinstance(debug_interceptor, RequestInterceptor):
                    self._request_interceptors.append(debug_interceptor)
                if isinstance(debug_interceptor, ResponseInterceptor):
                    self._response_interceptors.append(debug_interceptor)
                
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
        if not self._port:
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
                    if port == self._port:
                        await kill_process(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    async def _stop_certificate_authority(self) -> None:
        """Stop the certificate authority."""
        # Stop CA with graceful fallback
        if self._ca:
            try:
                # Try cleanup/stop methods in sequence
                for method in ['cleanup_old_certs', 'stop', 'close']:
                    if hasattr(self._ca, method):
                        try:
                            await getattr(self._ca, method)()
                        except Exception as e:
                            logger.warning(f"CA {method}() failed: {e}")
            except Exception as e:
                logger.error(f"Error stopping CA: {e}")
            finally:
                self._ca = None

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
            if self._server:
                self._server.close()
                await self._server.wait_closed()
                self._server = None

            # Stop certificate manager
            if cert_manager and cert_manager.is_running():
                await cert_manager.stop()

            # Stop CA with graceful fallback
            if self._ca:
                try:
                    # Try cleanup/stop methods in sequence
                    for method in ['cleanup_old_certs', 'stop', 'close']:
                        if hasattr(self._ca, method):
                            try:
                                await getattr(self._ca, method)()
                            except Exception as e:
                                logger.warning(f"CA {method}() failed: {e}")
                except Exception as e:
                    logger.error(f"Error stopping CA: {e}")
                self._ca = None

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

    def _create_tunnel_protocol(self, client_transport: asyncio.Transport) -> TunnelProtocol:
        """Create a new tunnel protocol instance."""
        connection_id = str(uuid.uuid4())
        flow_control = FlowControl(client_transport)
        return TunnelProtocol(
            client_transport=client_transport,
            flow_control=flow_control,
            connection_id=connection_id,
            interceptor_class=DatabaseInterceptor,
            buffer_size=self.config.buffer_size,
            metrics_interval=self.config.metrics_interval,
            write_limit=self.config.write_limit,
            write_interval=self.config.write_interval
        )
