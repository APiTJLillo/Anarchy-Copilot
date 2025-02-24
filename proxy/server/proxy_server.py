"""Main proxy server implementation."""
import asyncio
import logging
import socket
import ssl
import signal
import os
import psutil
import sys
import uvicorn
from contextlib import AsyncExitStack
from pathlib import Path
from async_timeout import timeout as async_timeout
from typing import Dict, List, Optional, Any, Tuple, Callable, TYPE_CHECKING
from urllib.parse import urlparse
from uuid import uuid4
from sqlalchemy import text
from datetime import datetime

from database import AsyncSessionLocal
from ..config import ProxyConfig

if TYPE_CHECKING:
    from api.proxy.database_models import ProxyHistoryEntry, ProxySession as DBProxySession

from .custom_protocol import TunnelProtocol
from .https_intercept_protocol import HttpsInterceptProtocol
from ..interceptor import RequestInterceptor, ResponseInterceptor, InterceptedRequest, InterceptedResponse
from ..encoding import ContentEncodingInterceptor
from ..session import ProxySession
from ..websocket import WebSocketManager, DebugInterceptor
from ..analysis.analyzer import TrafficAnalyzer
from .certificates import CertificateAuthority
from .tls_helper import cert_manager, CertificateManager
from .handlers import ProxyResponse

logger = logging.getLogger("proxy.core")

_instance = None

class ProxyServer:
    """Main proxy server implementation."""
    
    def __init__(self, config: ProxyConfig, add_default_interceptors: bool = False):
        """Initialize the proxy server."""
        self.config = config
        self._host = config.host
        self._port = config.port
        self.session = ProxySession(max_history=config.history_size)
        self._request_interceptors: List[RequestInterceptor] = []
        self._response_interceptors: List[ResponseInterceptor] = []
        
        if add_default_interceptors:
            content_encoding = ContentEncodingInterceptor()
            self._request_interceptors.append(content_encoding)
            self._response_interceptors.append(content_encoding)

        self._db_initialized = False
        self._server = None
        self._ssl_contexts: Dict[str, ssl.SSLContext] = {}
        self._is_running: bool = False
        
        # Initialize components
        self._ws_manager = WebSocketManager()
        if config.websocket_support:
            self._ws_manager.add_interceptor(DebugInterceptor())

        self._analyzer = TrafficAnalyzer()
        
        # Track pending requests/responses
        self._pending_requests: Dict[str, Any] = {}
        self._pending_responses: Dict[str, Any] = {}

        # Initialize CA
        self._ca = None
        try:
            if config.ca_cert_path and config.ca_key_path:
                try:
                    # Initialize CA
                    self._ca = CertificateAuthority(
                        config.ca_cert_path,
                        config.ca_key_path
                    )
                    # Configure cert manager with CA
                    cert_manager.set_ca(self._ca)
                    logger.info("Certificate Authority initialized successfully")
                except Exception as e:
                    logger.warning(f"Certificate Authority initialization failed: {e}")
                    logger.warning("Proxy will run without HTTPS interception capability")
                    self._ca = None
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
            _instance.config.update(config_dict)
            # Update key attributes
            _instance._host = config.host
            _instance._port = config.port
            _instance.session.max_history = config.history_size
            logger.info("Updated existing proxy server instance with config: %s", config_dict)

    @property
    def is_running(self) -> bool:
        """Check if the proxy server is running."""
        return self._is_running

    async def _force_cleanup_port(self) -> None:
        """Force cleanup any processes using the port."""
        def is_current_process(pid: int) -> bool:
            return pid == os.getpid() or (
                psutil.Process(pid).ppid() == os.getpid()
            )

        def get_processes_using_port(port: int) -> List[int]:
            pids = []
            try:
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        # Separately get connections to handle cases where attribute is unavailable
                        if hasattr(proc, 'connections'):
                            for conn in proc.connections():
                                try:
                                    if conn.laddr.port == port and not is_current_process(proc.pid):
                                        pids.append(proc.pid)
                                except (AttributeError, TypeError):
                                    continue
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
            except Exception as e:
                logger.warning(f"Error checking processes: {e}")
            return pids

        try:
            # Find and terminate processes using the port
            pids = get_processes_using_port(self._port)
            if pids:
                logger.warning(f"Found processes {pids} using port {self._port}")
                for pid in pids:
                    try:
                        proc = psutil.Process(pid)
                        proc.terminate()
                    except:
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except:
                            pass

                # Wait for processes to terminate
                await asyncio.sleep(1)

                # Check if processes are still running
                for pid in pids:
                    try:
                        proc = psutil.Process(pid)
                        if proc.is_running():
                            logger.warning(f"Force killing process {pid}")
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except:
                                pass
                    except psutil.NoSuchProcess:
                        pass

            # Ensure the port is free
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            try:
                sock.bind((self._host, self._port))
                sock.close()
            except Exception as e:
                logger.error(f"Port {self._port} is still in use after cleanup: {e}")
                raise RuntimeError(f"Port {self._port} is still in use and cannot be cleaned up")

        except Exception as e:
            logger.error(f"Error during port cleanup: {e}")
            raise

    async def start(self, port: Optional[int] = None, host: Optional[str] = None) -> None:
        """Start the proxy server."""
        logger.info("Starting proxy server...")
        if self._is_running:
            raise RuntimeError("Proxy server is already running")

        if port:
            self._port = port
        if host:
            self._host = host

        # Initialize database
        try:
            logger.info("Initializing database connection...")
            async with AsyncSessionLocal() as session:
                # First check if tables exist
                result = await session.execute(
                    text("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='proxy_history'")
                )
                tables_count = result.scalar()
                if not tables_count:
                    logger.error("Database tables not found - migrations may not have been run")
                    raise RuntimeError("Database tables not found - please run migrations first")

                # Test querying the table
                result = await session.execute(text("SELECT COUNT(*) FROM proxy_history"))
                count = result.scalar()
                logger.info("Database connection test successful - found %d history entries", count)
            self._db_initialized = True
            logger.info("Database initialization completed")
        except Exception as e:
            logger.error("Database initialization failed: %s", str(e), exc_info=True)
            raise RuntimeError("Failed to initialize database connection") from e

        try:
            # Attempt to clean up any existing processes using the port
            await self._force_cleanup_port()
            await asyncio.sleep(1)  # Brief pause to allow cleanup

            # Configure server
            config = uvicorn.Config(
                app=self.create_asgi_app(),
                host=self._host,
                port=self._port,
                log_level="debug",
                server_header=False,
                proxy_headers=False,
                forwarded_allow_ips='*',
                http=TunnelProtocol if not self._ca else HttpsInterceptProtocol.create_protocol_factory(),
                timeout_keep_alive=30,  # Fixed timeout for better tunnel stability
                timeout_notify=1,
                timeout_graceful_shutdown=3,
                access_log=False,
                use_colors=False
            )

            # Create and start server
            try:
                server = uvicorn.Server(config)
                server.config.reload = False  # Ensure reload is disabled
                self._server = server
                self._is_running = True
                logger.info(f"Starting proxy server on {self._host}:{self._port}")
                await server.serve()
            except OSError as e:
                if e.errno == 98:  # Address already in use
                    logger.error(f"Port {self._port} is still in use after cleanup")
                    await self._force_cleanup_port()  # Try one more time
                    await asyncio.sleep(0.5)
                    # Try starting again
                    server = uvicorn.Server(config)
                    self._server = server
                    await server.serve()
                else:
                    raise
            except Exception as e:
                self._is_running = False
                self._server = None
                raise RuntimeError(f"Server startup failed: {e}") from e

        except Exception as e:
            self._is_running = False
            self._server = None
            logger.error(f"Failed to start proxy server: {e}")
            raise RuntimeError(f"Failed to start proxy server: {e}") from e

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
        from .asgi import ASGIHandler
        return ASGIHandler(self)

    def get_analyzer(self) -> Optional[TrafficAnalyzer]:
        """Get the traffic analyzer instance."""
        return self._analyzer if self._is_running else None

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
            if self.config.allowed_hosts and host not in self.config.allowed_hosts:
                return ProxyResponse(
                    status_code=403,
                    headers={'Content-Type': 'text/plain'},
                    body=b'Host not allowed'
                )
            
            if self.config.excluded_hosts and host in self.config.excluded_hosts:
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
                if self.config.intercept_requests:
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
                    if self.config.intercept_responses:
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
