"""Main proxy server implementation."""
import asyncio
import logging
import socket
import ssl
import signal
import os
import psutil
import uvicorn
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from urllib.parse import urlparse
from uuid import uuid4
from sqlalchemy import text

from database import AsyncSessionLocal

from ..config import ProxyConfig
from .custom_protocol import TunnelProtocol
from ..interceptor import RequestInterceptor, ResponseInterceptor, InterceptedRequest, InterceptedResponse
from ..encoding import ContentEncodingInterceptor
from ..session import ProxySession
from ..websocket import WebSocketManager, DebugInterceptor
from ..analysis.analyzer import TrafficAnalyzer
from .certificates import CertificateAuthority
from .handlers import ProxyResponse

logger = logging.getLogger("proxy.core")

class ProxyServer:
    """Main proxy server implementation."""
    
    def __init__(self, config: ProxyConfig, add_default_interceptors: bool = False):
        """Initialize the proxy server."""
        self.config = config
        self._host = config.host or "127.0.0.1"
        self._port = config.port or 8080
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

        try:
            if config.ca_cert_path and config.ca_key_path:
                self._ca = CertificateAuthority(
                    config.ca_cert_path,
                    config.ca_key_path
                )
            else:
                self._ca = None
        except Exception as e:
            logger.warning(f"Certificate Authority initialization failed: {e}")
            logger.warning("Proxy will run without HTTPS interception capability")
            self._ca = None

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
        if not self._db_initialized:
            try:
                async_session = AsyncSessionLocal()
                async with async_session as session:
                    await session.execute(text("SELECT 1"))
                    logger.info("Database connection established")
                    self._db_initialized = True
                    await session.close()
            except Exception as e:
                logger.warning(f"Database initialization failed: {e}", exc_info=True)
                logger.warning("Proxy will run without database functionality")

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
                http=TunnelProtocol,
                timeout_keep_alive=30,
                timeout_notify=1,
                timeout_graceful_shutdown=3,
                access_log=False,
                use_colors=False
            )

            # Kill parent process first if it exists
            parent = psutil.Process(os.getppid())
            if any(conn.laddr.port == self._port for conn in parent.connections()):
                logger.warning("Found parent process using port, terminating")
                parent.terminate()
                await asyncio.sleep(1)

            # Clean up the port
            await self._force_cleanup_port()

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
                    
                    # Add timeout for shutdown
                    try:
                        async with asyncio.timeout(3.0):  # 3 second timeout
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

    async def handle_request(self, scope: dict, receive: Callable, send: Callable) -> Optional[ProxyResponse]:
        """Handle an incoming request."""
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
            # Process regular HTTP request
            logger.debug(f"ProxyServer handling HTTP {method} request")

            # Extract info for regular HTTP request
            method, host, path = self._extract_request_info(scope)
            if not host:
                return ProxyResponse(
                    status_code=400,
                    headers={'Content-Type': 'text/plain'},
                    body=b'Missing Host header'
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
                for interceptor in self._request_interceptors:
                    request = await interceptor.intercept(request)

                # Forward regular HTTP request
                async with self.session.request(request) as aiohttp_response:
                    # Create intercepted response object
                    response = InterceptedResponse(
                        status_code=aiohttp_response.status,
                        headers=dict(aiohttp_response.headers),
                        body=await aiohttp_response.read()
                    )

                    # Run response interceptors
                    for interceptor in self._response_interceptors:
                        response = await interceptor.intercept(response, request)
                    
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
