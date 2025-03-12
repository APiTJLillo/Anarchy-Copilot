"""HTTPS interception protocol implementation."""
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
from ..tls_helper import cert_manager
from ..handlers.http import HttpRequestHandler
from proxy.interceptors.database import DatabaseInterceptor
from proxy.interceptor import InterceptedRequest, InterceptedResponse
from database import AsyncSessionLocal
from proxy.models import ProxySessionData
from sqlalchemy import text, select
from proxy.session import get_active_sessions

logger = logging.getLogger("proxy.core")
logger.setLevel(logging.DEBUG)

# Also enable debug logging for SSL/TLS operations
logging.getLogger("ssl").setLevel(logging.DEBUG)

class HttpsInterceptProtocol(BaseProxyProtocol):
    """Protocol for intercepting HTTPS traffic using modular components."""

    # Class level settings
    _ca_instance: ClassVar[Optional['CertificateAuthority']] = None
    _transport_retry_attempts: int = 3
    _transport_retry_delay: float = 0.5
    _database_interceptor: Optional[DatabaseInterceptor] = None
    _ca_initialized: bool = False
    _init_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _initialization_task: ClassVar[Optional[asyncio.Task[bool]]] = None
    _cleanup_tasks: ClassVar[Dict[str, asyncio.Task]] = {}

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

        async with cls._init_lock:
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize managers and handlers
        self._state_manager = StateManager(self._connection_id)
        self._error_handler = ErrorHandler(self._connection_id, self.transport)
        self._buffer_manager = BufferManager(self._connection_id, self.transport)
        self._tls_handler = TlsHandler(
            self._connection_id,
            self._state_manager,
            self._error_handler
        )

        # Initialize handlers
        self._connect_handler = None  # Will be initialized in connection_made
        self._http_handler = HttpRequestHandler(self._connection_id)
        self._database_interceptor = None  # Will be initialized in _setup_database_interceptor

        # Initialize protocol state
        self._remote_transport: Optional[asyncio.Transport] = None
        self._tunnel: Optional[asyncio.Transport] = None
        self._pending_data: list[bytes] = []  # Buffer for data received before tunnel setup
        self._tunnel_established = False
        self._setup_initial_state()
        
        logger.debug(f"[{self._connection_id}] HttpsInterceptProtocol initialized")

    @classmethod
    def create_protocol_factory(cls) -> Callable[..., 'HttpsInterceptProtocol']:
        """Create a protocol factory."""
        def protocol_factory(*args, **kwargs) -> 'HttpsInterceptProtocol':
            if not cls._ca_initialized:
                raise RuntimeError("CA must be initialized before creating protocol instances")
            protocol = cls(*args, **kwargs)
            protocol._state_manager.set_intercept_enabled(True)
            asyncio.create_task(protocol._setup_database_interceptor())
            connection_id = protocol._connection_id
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
        if self._database_interceptor:
            logger.debug(f"[{self._connection_id}] Database interceptor already initialized")
            return

        try:
            logger.debug(f"[{self._connection_id}] Setting up database interceptor")
            sessions = await get_active_sessions()
            if sessions:
                logger.debug(f"[{self._connection_id}] Found active session {sessions[0]['id']}")
                self._database_interceptor = DatabaseInterceptor(self._connection_id)
                logger.info(f"[{self._connection_id}] Initialized database interceptor for session {sessions[0]['id']}")
            else:
                logger.warning(f"[{self._connection_id}] No active session found for database interceptor")
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to setup database interceptor: {e}", exc_info=True)
            # Try to recover by creating a new session
            try:
                logger.info(f"[{self._connection_id}] Attempting to create new session")
                async with AsyncSessionLocal() as session:
                    from api.proxy.database_models import ProxySession
                    from proxy.models.proxy_types import ProxySessionData
                    
                    new_session = ProxySession(
                        name="Auto-created Session",
                        settings={"intercept_requests": True, "intercept_responses": True},
                        is_active=True
                    )
                    session.add(new_session)
                    await session.commit()
                    await session.refresh(new_session)
                    
                    # Convert to proxy session data
                    session_data = ProxySessionData.from_db(new_session)
                    logger.info(f"[{self._connection_id}] Created new session {session_data.id}")
                    self._database_interceptor = DatabaseInterceptor(self._connection_id)
            except Exception as e2:
                logger.error(f"[{self._connection_id}] Failed to create new session: {e2}", exc_info=True)

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle new connection."""
        super().connection_made(transport)
        
        from ..handlers.connect_factory import create_connect_handler, ConnectConfig
        
        # Initialize ConnectHandler using factory
        config = ConnectConfig(
            connection_id=self._connection_id,
            transport=transport,
            connect_timeout=30,
            read_timeout=60
        )
        self._connect_handler = create_connect_handler(
            config,
            self._state_manager,
            self._error_handler,
            tls_handler=self._tls_handler
        )
        
        # Setup database interceptor
        asyncio.create_task(self._setup_database_interceptor())
        logger.debug(f"[{self._connection_id}] Connection established")

    def _setup_initial_state(self) -> None:
        """Set up initial protocol state."""
        self._state_manager.set_intercept_enabled(bool(self._ca_instance))
        logger.debug(f"[{self._connection_id}] TLS interception enabled: {bool(self._ca_instance)}")

    async def handle_request(self, request: Request) -> None:
        """Handle HTTPS interception request."""
        try:
            # Convert method to string if it's bytes
            method = request.method.decode() if isinstance(request.method, bytes) else request.method
            target = request.target.decode() if isinstance(request.target, bytes) else request.target
            
            logger.debug(f"[{self._connection_id}] Handling request: {method} {target}")
            
            if method != "CONNECT":
                # For non-CONNECT requests, delegate to HTTP handler and database interceptor
                logger.debug(f"[{self._connection_id}] Handling non-CONNECT request: {method} {target}")
                
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
                        connection_id=self._connection_id
                    )
                    logger.debug(f"[{self._connection_id}] Created intercepted request object")
                    
                    # Let database interceptor process request
                    try:
                        modified_request = await self._database_interceptor.intercept(intercepted_request)
                        logger.debug(f"[{self._connection_id}] Successfully intercepted request")
                        
                        # Update request with any modifications from interceptor
                        request.method = modified_request.method.encode() if isinstance(modified_request.method, str) else modified_request.method
                        request.target = modified_request.url.encode() if isinstance(modified_request.url, str) else modified_request.url
                        request.headers = modified_request.headers
                        request.body = modified_request.body
                    except Exception as e:
                        logger.error(f"[{self._connection_id}] Failed to intercept request: {e}", exc_info=True)
                else:
                    logger.warning(f"[{self._connection_id}] No database interceptor available for request")
                    
                # Continue with normal handling
                response = await self._http_handler.handle_request(request)
                
                # Intercept response if we have an interceptor
                if self._database_interceptor and response:
                    try:
                        intercepted_response = InterceptedResponse(
                            status_code=response.status_code,
                            headers=response.headers,
                            body=response.body if isinstance(response.body, bytes) else str(response.body).encode('utf-8'),
                            connection_id=self._connection_id
                        )
                        await self._database_interceptor.intercept(intercepted_response, intercepted_request)
                        logger.debug(f"[{self._connection_id}] Successfully intercepted response")
                    except Exception as e:
                        logger.error(f"[{self._connection_id}] Failed to intercept response: {e}", exc_info=True)
                
                return

            # Parse target from request
            host, port = self._parse_authority(target)
            
            # Log request details and state
            logger.debug(f"[{self._connection_id}] Handling CONNECT request for {host}:{port}")
            logger.debug(f"[{self._connection_id}] Current transport state: {self.transport is not None and not self.transport.is_closing()}")
            logger.debug(f"[{self._connection_id}] Current tunnel state: {self._tunnel is not None and not self._tunnel.is_closing() if self._tunnel else False}")
            
            # Handle CONNECT request
            await self._handle_connect(host, port)
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error handling request: {e}", exc_info=True)
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
            logger.error(f"[{self._connection_id}] Failed to parse authority '{authority}': {e}")
            raise ValueError(f"Invalid authority format: {authority}")

    async def _cleanup(self, error: Optional[str] = None) -> None:
        """Clean up connection resources."""
        try:
            logger.debug(f"[{self._connection_id}] Starting cleanup, error: {error}")
            
            # Update state
            await self._state_manager.update_status("closing", error=error)
            
            # Close handlers
            try:
                if self._connect_handler:
                    self._connect_handler.close()
                if self._http_handler:
                    self._http_handler.close()
                if self._database_interceptor:
                    await self._database_interceptor.close()
            except Exception as e:
                logger.warning(f"[{self._connection_id}] Error during handler cleanup: {e}")
            
            # Clear buffers
            self._buffer_manager.clear_buffers()
            
            # Final state update and cleanup
            await self._state_manager.update_status("closed")
            await super()._cleanup(error=error)
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Cleanup error: {e}")

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
                logger.debug(f"[{self._connection_id}] Sent error response: {status_code} {message}")
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to send error response: {e}")

    def set_tunnel(self, tunnel: asyncio.Transport) -> None:
        """Set the tunnel transport for bidirectional forwarding."""
        self._tunnel = tunnel
        self._tunnel_established = True
        logger.debug(f"[{self._connection_id}] Set tunnel transport")
        
        # Forward any pending data
        if self._pending_data and self._tunnel and not self._tunnel.is_closing():
            logger.debug(f"[{self._connection_id}] Forwarding {len(self._pending_data)} buffered chunks after tunnel setup")
            for data in self._pending_data:
                self._tunnel.write(data)
            self._pending_data.clear()

    def data_received(self, data: bytes) -> None:
        """Schedule data handling in event loop."""
        if not self._is_closing and self.transport and not self.transport.is_closing():
            # Create task for async handling with error callback
            task = asyncio.create_task(self._handle_data(data))
            task.add_done_callback(self._handle_data_error)
        
    def _handle_data_error(self, task: asyncio.Task[None]) -> None:
        """Handle any errors from async data processing."""
        try:
            exc = task.exception()
            if exc:
                logger.error(f"[{self._connection_id}] Error handling data: {exc}", exc_info=exc)
                asyncio.create_task(self._cleanup(str(exc)))
        except asyncio.CancelledError:
            pass

    async def _handle_data(self, data: bytes) -> None:
        """Handle received data with detailed logging and buffering."""
        try:
            if not self._tunnel_established:
                # Buffer data if tunnel not ready
                logger.debug(f"[{self._connection_id}] Buffering {len(data)} bytes until tunnel is established")
                self._pending_data.append(data)
                return

            if self._tunnel and not self._tunnel.is_closing():
                # First send any pending data
                if self._pending_data:
                    logger.debug(f"[{self._connection_id}] Forwarding {len(self._pending_data)} buffered chunks")
                    for buffered in self._pending_data:
                        await self._process_data_chunk(buffered) # Fixed: Added await here
                        self._tunnel.write(buffered)
                    self._pending_data.clear()
                
                # Then handle current data
                logger.debug(f"[{self._connection_id}] Processing {len(data)} bytes")
                await self._process_data_chunk(data) # Fixed: Added await here
                self._tunnel.write(data)
                logger.debug(f"{self._connection_id} Forwarded {len(data)} bytes")
            else:
                logger.warning(
                    f"[{self._connection_id}] Cannot forward data - "
                    f"Tunnel exists: {self._tunnel is not None}, "
                    f"Tunnel closing: {self._tunnel.is_closing() if self._tunnel else True}"
                )
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error handling data: {e}")
            
    async def _process_data_chunk(self, data: bytes) -> None:
        """Process a chunk of data through interceptors."""
        if self._database_interceptor is None:
            logger.warning(f"[{self._connection_id}] Database interceptor not initialized")
            # Try to initialize it if we failed before
            await self._setup_database_interceptor()
            if self._database_interceptor is None:
                logger.error(f"[{self._connection_id}] Failed to initialize database interceptor after retry")
                return
            
        try:
            # Try to parse as HTTP message
            first_line = data.split(b'\r\n')[0].decode('utf-8', errors='ignore')
            
            if ' ' not in first_line:
                logger.debug(f"[{self._connection_id}] Not an HTTP message: {first_line[:50]}")
                return
                
            try:
                if first_line.startswith(('GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH')):
                    # Parse HTTP request
                    logger.debug(f"[{self._connection_id}] Detected HTTP request, parsing...")
                    method, target, *_ = first_line.split(' ')
                    
                    # Parse headers
                    headers = {}
                    body = None
                    if b'\r\n\r\n' in data:
                        header_section, body = data.split(b'\r\n\r\n', 1)
                        header_lines = header_section.split(b'\r\n')[1:]
                        logger.debug(f"[{self._connection_id}] Parsing {len(header_lines)} header lines")
                        
                        for line in header_lines:
                            try:
                                line = line.decode('utf-8', errors='ignore')
                                if ': ' in line:
                                    name, value = line.split(': ', 1)
                                    headers[name] = value
                            except Exception as e:
                                logger.debug(f"[{self._connection_id}] Error parsing header line: {e}")
                                continue
                    
                    # Create and store intercepted request
                    intercepted_request = InterceptedRequest(
                        method=method,
                        url=target,
                        headers=headers,
                        body=body,
                        connection_id=self._connection_id
                    )
                    
                    logger.debug(f"[{self._connection_id}] Intercepting request: {method} {target}")
                    await self._database_interceptor.intercept(intercepted_request)
                    logger.info(f"[{self._connection_id}] Successfully intercepted and stored HTTP request: {method} {target}")
                    
                    # Store for matching with response
                    self._last_request = intercepted_request
                    
                elif first_line.startswith('HTTP/'):
                    # Parse HTTP response
                    logger.debug(f"[{self._connection_id}] Detected HTTP response, parsing...")
                    version, status_code, *reason = first_line.split(' ')
                    status_code = int(status_code)
                    
                    # Parse headers
                    headers = {}
                    body = None
                    if b'\r\n\r\n' in data:
                        header_section, body = data.split(b'\r\n\r\n', 1)
                        header_lines = header_section.split(b'\r\n')[1:]
                        logger.debug(f"[{self._connection_id}] Parsing {len(header_lines)} header lines")
                        
                        for line in header_lines:
                            try:
                                line = line.decode('utf-8', errors='ignore')
                                if ': ' in line:
                                    name, value = line.split(': ', 1)
                                    headers[name] = value
                            except Exception as e:
                                logger.debug(f"[{self._connection_id}] Error parsing header line: {e}")
                                continue
                    
                    # Create and store intercepted response
                    logger.debug(f"[{self._connection_id}] Creating InterceptedResponse object")
                    intercepted_response = InterceptedResponse(
                        status_code=status_code,
                        headers=headers,
                        body=body,
                        connection_id=self._connection_id
                    )
                    
                    # Get the last request if available
                    last_request = None
                    if hasattr(self, '_last_request'):
                        last_request = self._last_request
                    
                    logger.debug(f"[{self._connection_id}] Intercepting response: {status_code}")
                    await self._database_interceptor.intercept(intercepted_response, last_request)
                    logger.info(f"[{self._connection_id}] Successfully intercepted and stored HTTP response: {status_code}")
                    
                else:
                    logger.debug(f"[{self._connection_id}] Data does not appear to be an HTTP message: {first_line}")
                    
            except Exception as e:
                # This is expected for non-HTTP data
                logger.debug(f"[{self._connection_id}] Could not parse as HTTP message: {str(e)}")
                
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error processing data chunk: {str(e)}", exc_info=True)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection lost event with proper cleanup and logging."""
        try:
            if exc:
                logger.error(f"[{self._connection_id}] Connection lost with error: {exc}")
            else:
                logger.debug(f"[{self._connection_id}] Connection closed cleanly")
            
            # Log final connection state
            logger.debug(
                f"[{self._connection_id}] Final connection state - "
                f"Transport closing: {self.transport.is_closing() if self.transport else True}, "
                f"Tunnel exists: {self._tunnel is not None}, "
                f"Tunnel closing: {self._tunnel.is_closing() if self._tunnel else True}"
            )
            
            # Clean up state
            self._state_manager.clear_state()
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error during connection cleanup: {e}")
        finally:
            super().connection_lost(exc)

    async def _handle_connect(self, host: str, port: int) -> None:
        """Handle CONNECT request."""
        if not self.transport or self.transport.is_closing():
            raise RuntimeError("Transport not available")

        # Send 200 Connection Established immediately
        response = (
            b"HTTP/1.1 200 Connection Established\r\n"
            b"Connection: keep-alive\r\n"
            b"Proxy-Agent: AnarchyProxy\r\n\r\n"
        )
        self.transport.write(response)
        logger.debug(f"[{self._connection_id}] Sent Connection Established response")
        
        # Check if ConnectHandler is initialized
        if not self._connect_handler:
            raise RuntimeError("Connection handler not initialized")
        
        # Now handle the connection with TLS interception
        try:
            if not self.transport or self.transport.is_closing():
                raise RuntimeError("Transport lost before handling CONNECT")

            transport_type = type(self.transport)
            socket_info = self.transport.get_extra_info('socket')
            logger.debug(f"[{self._connection_id}] Starting CONNECT handling with transport type: {transport_type}")
            logger.debug(f"[{self._connection_id}] Transport socket available: {socket_info is not None}")
            
            await self._connect_handler.handle_connect(
                self,
                host=host,
                port=port,
                intercept_tls=self._state_manager.is_intercept_enabled()
            )
            
            # Verify server transport is available
            if not self._connect_handler.server_transport:
                raise RuntimeError("Server transport not available after connection")
            
            # Log success and transport states
            logger.debug(f"[{self._connection_id}] Connection handler completed. Transport states:")
            logger.debug(f"[{self._connection_id}] - Client transport: {self.transport is not None and not self.transport.is_closing()}")
            logger.debug(f"[{self._connection_id}] - Server transport: {self._connect_handler.server_transport is not None and not self._connect_handler.server_transport.is_closing()}")
            logger.debug(f"[{self._connection_id}] - Tunnel transport: {self._tunnel is not None and not self._tunnel.is_closing() if self._tunnel else False}")
            
            logger.info(f"[{self._connection_id}] Successfully established tunnel to {host}:{port}")
            await self._state_manager.update_status("established")
            
            # Wait for the connection to be closed
            logger.debug(f"[{self._connection_id}] Waiting for connection to complete")
            while not self.transport.is_closing():
                await asyncio.sleep(0.1)
                # Log periodic state checks
                if not hasattr(self, '_last_state_check') or time.time() - self._last_state_check > 5:
                    logger.debug(
                        f"[{self._connection_id}] Connection state check - "
                        f"Client: {not self.transport.is_closing()}, "
                        f"Server: {not self._connect_handler.server_transport.is_closing() if self._connect_handler.server_transport else False}, "
                        f"Tunnel: {not self._tunnel.is_closing() if self._tunnel else False}"
                    )
                    self._last_state_check = time.time()
            logger.debug(f"[{self._connection_id}] Connection closed")
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to establish tunnel: {e}", exc_info=True)
            # Don't send error response here since we already sent 200
            return
