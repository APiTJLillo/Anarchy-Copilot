"""Custom protocol handler for HTTPS tunneling."""
import asyncio
import errno
import logging
import socket
import time
from datetime import datetime, timezone
from typing import Optional, Tuple, Any, Set, Dict, List
from urllib.parse import unquote
from uuid import uuid4
from async_timeout import timeout
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from database import engine

from uvicorn.protocols.http.h11_impl import H11Protocol as BaseH11Protocol
from sqlalchemy import text, select
import h11
from typing import Dict, List, Optional, Any, Tuple, Callable, TYPE_CHECKING
from h11 import Request, Response

from database import AsyncSessionLocal
from api.proxy.database_models import ProxyHistoryEntry
from .flow_control import FlowControl
from .state import proxy_state

if TYPE_CHECKING:
    from .protocol import HttpsInterceptProtocol

logger = logging.getLogger("proxy.core")

def log_connection(conn_id: str, message: str) -> None:
    """Helper for consistent connection logging."""
    logger.debug(f"[Connection {conn_id}] {message}")

class H11Protocol:
    """Custom H11Protocol that supports async data_received."""
    
    def __init__(self, connection_id: Optional[str] = None, config=None, server=None):
        """Initialize H11Protocol with required parameters."""
        logger.debug("Initializing H11Protocol")
        logger.debug(f"H11Protocol params - connection_id: {connection_id}, has_config: {config is not None}, has_server: {server is not None}")
        
        self.connection_id = connection_id or str(uuid4())
        self.conn = h11.Connection(h11.SERVER)
        self.transport = None
        self.flow = None
        self.server = server
        self.client = None
        self.scheme = None
        self.scope = None
        self.headers = None
        self.config = config
        logger.debug("H11Protocol initialization completed")
    
    def _send_error_response(self, status_code: int, message: str) -> None:
        """Send an error response."""
        try:
            if not self.transport or self.transport.is_closing():
                return
                
            # Reset connection state if needed
            if self.conn.our_state in {h11.ERROR}:
                self.conn = h11.Connection(h11.SERVER)
                
            response = h11.Response(
                status_code=status_code,
                headers=[
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"connection", b"close"),
                ],
                reason=message.encode()
            )
            
            try:
                self.transport.write(self.conn.send(response))
                self.transport.write(self.conn.send(h11.EndOfMessage()))
            except h11._util.LocalProtocolError:
                # If we can't send the error response, just close the connection
                pass
                
            self.transport.close()
            
        except Exception as e:
            logger.error(f"[{self.connection_id}] Error sending error response: {e}")
            if self.transport and not self.transport.is_closing():
                self.transport.close()

    async def data_received(self, data: bytes) -> None:
        """Handle incoming data."""
        try:
            # If in tunnel mode, forward data directly
            if self._in_tunnel_mode:
                if self._remote_transport and not self._remote_transport.is_closing():
                    self._remote_transport.write(data)
                    await self._record_event("data", "browser-proxy", "success", len(data))
                return

            # Otherwise handle as HTTP
            try:
                if self.conn.their_state in {h11.DONE, h11.MUST_CLOSE, h11.CLOSED}:
                    self.conn = h11.Connection(h11.SERVER)
                    
                self.conn.receive_data(data)
                
                while True:
                    event = self.conn.next_event()
                    if event is h11.NEED_DATA:
                        break
                    elif isinstance(event, h11.Request):
                        await self.handle_request(event)
                    elif isinstance(event, h11.Data):
                        await self.handle_body(event.data)
                    elif isinstance(event, h11.EndOfMessage):
                        await self.handle_endofmessage(event)
                        
            except h11._util.RemoteProtocolError as e:
                msg = "Invalid HTTP request received"
                self._send_error_response(400, msg)
                logger.warning(f"[{self.connection_id}] {msg}: {e}")
            except Exception as e:
                msg = "Error processing request"
                self._send_error_response(500, str(e))
                logger.error(f"[{self.connection_id}] {msg}: {e}", exc_info=True)
            
        except Exception as e:
            logger.error(f"[{self.connection_id}] Unhandled error: {e}", exc_info=True)
            if self.transport and not self.transport.is_closing():
                self.transport.close()

    async def handle_request(self, event: h11.Request) -> None:
        """Handle an HTTP request event."""
        self.scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.1"},
            "http_version": "1.1",
            "server": None,
            "client": self.transport.get_extra_info("peername"),
            "scheme": "http",
            "method": event.method.decode("ascii"),
            "root_path": self.config.root_path,
            "path": unquote(event.target.decode("ascii")),
            "raw_path": event.target,
            "query_string": b"",
            "headers": event.headers,
            "extensions": {"http.response.push": {}},
        }

        self.headers = [(name.lower(), value) for name, value in event.headers]
        self.flow = self.config.flow_factory(self.scope)
        
        try:
            await self.flow.handle_request()
        except Exception as exc:
            msg = "Error handling request"
            self._send_error_response(500, msg)
            logger.error(msg, exc_info=True)

    async def handle_body(self, body: bytes) -> None:
        """Handle request body data."""
        if self.flow is None:
            return
        
        try:
            await self.flow.handle_body(body)
        except Exception as exc:
            msg = "Error handling request body"
            self._send_error_response(500, msg)
            logger.error(msg, exc_info=True)

    async def handle_endofmessage(self, event: h11.EndOfMessage) -> None:
        """Handle end of the HTTP message."""
        if self.flow is None:
            return
            
        try:
            # Signal end of request to flow control
            if hasattr(self.flow, 'handle_endofmessage'):
                await self.flow.handle_endofmessage()
            elif hasattr(self.flow, 'handle_request_end'):
                await self.flow.handle_request_end()
            else:
                # Default handling - just log completion
                logger.debug("Request processing completed")
                
            # Reset connection state for next request
            self.flow = None
            
        except Exception as exc:
            msg = "Error handling end of message"
            self._send_error_response(500, msg)
            logger.error(msg, exc_info=True)

class TunnelProtocol(H11Protocol):
    """Custom protocol handler for HTTPS tunneling."""
    
    _active_connections: Dict[str, Dict[str, Any]] = {}  # Class variable to track active connections
    
    def __init__(self, connection_id: Optional[str] = None, client_transport: Any = None, flow_control: Any = None,
                 interceptor_class: Optional[Any] = None, buffer_size: int = 262144, 
                 metrics_interval: float = 0.1, write_limit: int = 1048576, 
                 write_interval: float = 0.0001):
        """Initialize tunnel protocol."""
        logger.debug("Initializing TunnelProtocol")
        logger.debug(f"TunnelProtocol params - connection_id: {connection_id}, has_client_transport: {client_transport is not None}")
        
        try:
            self._connection_id = connection_id or str(uuid4())
            logger.debug(f"Generated connection_id: {self._connection_id}")
            
            # Initialize base class without uvicorn dependencies
            super().__init__(connection_id=self._connection_id)
            logger.debug("H11Protocol.__init__ completed")
            
        except Exception as e:
            logger.error(f"Error in TunnelProtocol initialization: {e}", exc_info=True)
            raise
            
        self._transport = None
        self._client_transport = client_transport
        self._flow_control = flow_control
        self._interceptor_class = interceptor_class
        self._interceptor = None
        self._buffer_size = buffer_size
        self._metrics_interval = metrics_interval
        self._write_limit = write_limit
        self._write_interval = write_interval
        self._write_queue = asyncio.Queue()
        self._write_task = None
        self._monitor_task = None
        self._bytes_sent = 0
        self._bytes_received = 0
        self._requests_processed = 0
        self._last_activity = time.time()
        self._tunnel_start_time = None
        self._tunnel_end_time = None
        self._in_tunnel_mode = False
        self._last_request = None  # Store the last request for matching with response
        
        # Transfer settings
        self._write_limit = write_limit
        self._write_interval = write_interval
        self._last_write = 0.0
        
        # Statistics and state tracking
        self._write_queue = asyncio.Queue(maxsize=100)  # Increased queue size
        self._write_task = None
        self._write_lock = asyncio.Lock()
        self._buffer_stats = {
            "current_size": 0,
            "peak_size": 0,
            "total_processed": 0,
            "chunks_processed": 0,
            "write_count": 0,
            "write_rate": 0,
            "avg_chunk_size": 0
        }
        self._last_metrics_update = 0
        self._pending_updates = {}
        
        # Rate limiting state
        self._write_permits = asyncio.Semaphore(10)  # Increased concurrent writes
        
        # Database integration
        self.session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        self._db: Optional[AsyncSession] = None
        
        # Register state
        asyncio.create_task(self._update_metrics("initialized"))
        
        # Add timeout counter
        self._write_timeouts = 0
        self._max_queue_size = 2000  # Increased queue size limit
        
        self._tunnel_tasks: Set[asyncio.Task] = set()
        self._remote_transport = None
        self._remote_protocol = None
        self._tunnel_start_time: Optional[datetime] = None
        self._history_entry_id: Optional[int] = None
        self._connect_headers: Dict[str, str] = {}
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._tunnel_buffer = bytearray()
        self._buffer = bytearray()  # Buffer for TLS handshake data
        self._connect_response_sent = False
        
        # Connection tracking initialization
        self._events: List[Dict[str, Any]] = []
        self._event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._event_processor = None
        self._stall_timeout = 10.0  # 10 seconds stall detection
        self._monitor_task = None

        # Register connection in active connections
        self._active_connections[self._connection_id] = {
            "id": self._connection_id,
            "start_time": datetime.now(timezone.utc).timestamp(),
            "bytes_sent": 0,
            "bytes_received": 0,
            "events": [],
            "status": "initialized"
        }

        log_connection(self._connection_id, "Protocol initialized")

    async def process_client_data(self, data: bytes) -> bytes:
        """Process data from client to target."""
        if not self._in_tunnel_mode:
            self._in_tunnel_mode = True
            self._tunnel_start_time = datetime.now(timezone.utc)
            await self._record_event("tunnel_start", "client->target", "success")
        
        self._bytes_sent += len(data)
        await self._record_event("data", "client->target", "success", len(data))
        return data

    async def process_target_data(self, data: bytes) -> bytes:
        """Process data from target to client."""
        if not self._in_tunnel_mode:
            self._in_tunnel_mode = True
            self._tunnel_start_time = datetime.now(timezone.utc)
            await self._record_event("tunnel_start", "target->client", "success")
        
        self._bytes_received += len(data)
        await self._record_event("data", "target->client", "success", len(data))
        return data

    async def _record_event(self, event_type: str, direction: str, status: str, bytes_transferred: Optional[int] = None) -> None:
        """Record an event to both memory and WebSocket broadcast."""
        event = {
            "type": event_type,
            "direction": direction,
            "timestamp": datetime.now(timezone.utc).timestamp(),
            "connection_id": self._connection_id,
            "status": status
        }
        
        if bytes_transferred is not None:
            event["bytes_transferred"] = bytes_transferred
            
        if self._host:
            event["host"] = self._host
        if self._port:
            event["port"] = self._port

        # Store in memory
        self._events.append(event)
        
        # Update connection tracking
        if self._connection_id in self._active_connections:
            connection_info = self._active_connections[self._connection_id]
            connection_info["events"] = self._events
            
            # Update transfer stats
            if direction in ["browser-proxy", "web-proxy"] and bytes_transferred:
                connection_info["bytes_received"] = connection_info.get("bytes_received", 0) + bytes_transferred
            elif direction in ["proxy-browser", "proxy-web"] and bytes_transferred:
                connection_info["bytes_sent"] = connection_info.get("bytes_sent", 0) + bytes_transferred
        
        # Queue for WebSocket broadcast
        try:
            from api.proxy.websocket import connection_manager
            if connection_manager:
                await connection_manager.broadcast_connection_update(self._active_connections[self._connection_id])
        except Exception as e:
            logger.error(f"Failed to broadcast event: {e}")

        log_connection(self._connection_id, f"Event recorded: {event_type} {direction} {status}")

    def connection_made(self, transport: Any) -> None:
        """Handle new connection."""
        super().connection_made(transport)
        self._transport = transport
        self.flow_control = FlowControl(transport)
        
        # Configure socket for better performance
        if hasattr(transport, 'get_extra_info'):
            sock = transport.get_extra_info('socket')
            if sock is not None:
                try:
                    # Set TCP_NODELAY to disable Nagle's algorithm
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    
                    # Enable TCP keepalive
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    
                    # Set TCP keepalive parameters if supported
                    if hasattr(socket, 'TCP_KEEPIDLE'):  # Linux
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
                    if hasattr(socket, 'TCP_KEEPINTVL'):
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                    if hasattr(socket, 'TCP_KEEPCNT'):
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
                        
                    # Set receive buffer size
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)  # 256KB
                    
                except (AttributeError, socket.error) as e:
                    logger.warning(f"Could not set socket options: {e}")
        
        log_connection(self._connection_id, "Connection made to proxy")
        
        # Initialize connection tracking
        conn_info = {
            "created_at": datetime.now(timezone.utc),
            "host": None,
            "port": None,
            "events": self._events,
            "bytes_received": 0,
            "bytes_sent": 0,
            "requests_processed": 0,
            "error": None,
            "status": "initialized"
        }
        self._active_connections[self._connection_id] = conn_info

        # Get active session and associate with connection
        async def init_connection():
            try:
                async with AsyncSessionLocal() as db:
                    result = await db.execute(
                        text("SELECT * FROM proxy_sessions WHERE is_active = true ORDER BY start_time DESC LIMIT 1")
                    )
                    active_session = result.first()
                    if active_session:
                        conn_info["session_id"] = active_session.id
                        logger.debug(f"[{self._connection_id}] Associated with session {active_session.id}")
            except Exception as e:
                logger.error(f"[{self._connection_id}] Failed to get active session: {e}")
            
            # Add to state tracking
            await proxy_state.add_connection(self._connection_id, conn_info)

        # Schedule connection initialization
        asyncio.create_task(init_connection())

    async def _handle_large_write(self) -> None:
        """Handle flow control for large writes."""
        try:
            if self._remote_transport and self.flow_control:
                self.flow_control.pause_reading()
                try:
                    async with timeout(1.0) as cm:
                        await asyncio.sleep(0.1)  # Brief pause to let buffers drain
                finally:
                    if not self._remote_transport.is_closing():
                        self.flow_control.resume_reading()
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error handling large write: {e}")
            # Update state on error
            await proxy_state.update_connection(self._connection_id, "error", str(e))

    def set_tunnel_mode(self, enabled: bool = True) -> None:
        """Set protocol to tunnel mode after CONNECT is established."""
        self._in_tunnel_mode = enabled
        if enabled:
            # Reset HTTP state
            self.conn = h11.Connection(h11.SERVER)
            logger.debug(f"[{self._connection_id}] Switched to tunnel mode")

    async def handle_request(self, request: h11.Request) -> None:
        """Handle an HTTP request event."""
        try:
            if request.method == b"CONNECT":
                target = request.target.decode()
                logger.debug(f"[{self._connection_id}] Handling CONNECT request for {target}")
                
                try:
                    host, port = self._parse_authority(target)
                    self._host = host
                    self._port = port
                    
                    # Update connection info
                    if self._connection_id in self._active_connections:
                        self._active_connections[self._connection_id].update({
                            "host": host,
                            "port": port,
                            "status": "connecting"
                        })
                    
                    await self._handle_connect(host, port)
                    
                except Exception as e:
                    logger.error(f"[{self._connection_id}] CONNECT error: {e}", exc_info=True)
                    self._send_error_response(502, str(e))
            else:
                await super().handle_request(request)
                
        except Exception as e:
            logger.error(f"[{self._connection_id}] Request error: {e}", exc_info=True)
            self._send_error_response(500, str(e))

    async def _handle_connect(self, host: str, port: int) -> None:
        """Handle CONNECT tunnel requests."""
        try:
            # Record connection attempt
            await self._update_metrics("connecting")
            
            logger.debug(f"[{self._connection_id}] Attempting connection to {host}:{port}")
            loop = asyncio.get_event_loop()

            # Create remote connection with retries
            for attempt in range(3):
                try:
                    async with timeout(10) as cm:
                        transport, protocol = await loop.create_connection(
                            lambda: TunnelTransport(
                                client_transport=self.transport,
                                connection_id=self._connection_id
                            ),
                            host=host,
                            port=port
                        )
                        break
                except (ConnectionRefusedError, asyncio.TimeoutError) as e:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                    
            logger.debug(f"[{self._connection_id}] Remote connection established")
            
            # Send 200 Connection Established
            if self.transport and not self.transport.is_closing():
                self.transport.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                
            # Update state
            await self._update_metrics("connected")
            
            # Switch to tunnel mode
            self.set_tunnel_mode(True)
            
            # Store remote transport
            self._remote_transport = transport
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Tunnel error: {e}", exc_info=True)
            self._send_error_response(502, str(e))
            if self._remote_transport and not self._remote_transport.is_closing():
                self._remote_transport.close()

    def set_tunnel_mode(self, enabled: bool = True) -> None:
        """Set protocol to tunnel mode after CONNECT is established."""
        self._in_tunnel_mode = enabled
        if enabled:
            # Reset HTTP state
            self.conn = h11.Connection(h11.SERVER)
            logger.debug(f"[{self._connection_id}] Switched to tunnel mode")

    async def _create_history_entry(self, request: Request, response: Response, duration: float) -> None:
        """Create a history entry for this request/response pair."""
        try:
            # Extract request data
            method = request.method.decode()
            url = request.url.decode()
            host = request.headers.get(b"host", b"").decode()
            path = request.path.decode()
            
            # Convert headers to dict
            request_headers = {}
            for name, value in request.headers.items():
                request_headers[name.decode()] = value.decode()
                
            response_headers = {}
            for name, value in response.headers.items():
                response_headers[name.decode()] = value.decode()
            
            # Create history entry
            from api.proxy.history import create_history_entry
            from api.proxy.database import get_session
            
            async with get_session() as db:
                await create_history_entry(
                    db=db,
                    session_id=self.session_id,
                    method=method,
                    url=url,
                    host=host,
                    path=path,
                    request_headers=request_headers,
                    request_body=request.content.decode() if request.content else None,
                    response_headers=response_headers,
                    response_body=response.content.decode() if response.content else None,
                    status_code=response.status_code,
                    duration=duration,
                    is_intercepted=self.intercept_requests or self.intercept_responses,
                    is_encrypted=False,  # TODO: Implement TLS detection
                    tags=[],  # TODO: Implement tagging
                    notes=None
                )
                
        except Exception as e:
            logger.error(f"Failed to create history entry: {e}")

    async def _update_history_duration(self) -> None:
        """Update the duration of the history entry."""
        if self._history_entry_id:
            try:
                async with AsyncSessionLocal() as db:
                    # Get the history entry
                    result = await db.execute(
                        text("SELECT * FROM proxy_history_entries WHERE id = :id"),
                        {"id": self._history_entry_id}
                    )
                    history_entry = result.first()
                    
                    if history_entry:
                        # Calculate duration
                        duration = (datetime.utcnow() - history_entry.timestamp).total_seconds()
                        
                        # Update the entry
                        await db.execute(
                            text("""
                                UPDATE proxy_history_entries 
                                SET duration = :duration,
                                    response_status = :status,
                                    bytes_received = :bytes_received,
                                    bytes_sent = :bytes_sent
                                WHERE id = :id
                            """),
                            {
                                "id": self._history_entry_id,
                                "duration": duration,
                                "status": 200,
                                "bytes_received": self._bytes_received,
                                "bytes_sent": self._bytes_sent
                            }
                        )
                        await db.commit()
                        logger.debug(f"[{self._connection_id}] Updated history entry {self._history_entry_id} duration to {duration:.2f}s")
            except Exception as e:
                logger.error(f"[{self._connection_id}] Failed to update history duration: {e}")

    def _parse_authority(self, authority: str) -> Tuple[str, int]:
        """Parse host and port from authority string."""
        try:
            if ':' in authority:
                host, port = authority.rsplit(':', 1)
                return host, int(port)
            else:
                return authority, 443  # Default HTTPS port
        except (ValueError, TypeError) as e:
            log_connection(self._connection_id, f"Failed to parse authority: {e}")
            raise ValueError(f"Invalid authority format: {authority}")

    async def _graceful_shutdown(self) -> None:
        """Perform graceful shutdown of the tunnel."""
        try:
            # Set status to shutting down
            await proxy_state.update_connection(self._connection_id, "status", "shutting_down")
            
            # Pause reading on both sides
            if self.flow_control:
                self.flow_control.pause_reading()
            if hasattr(self._remote_transport, 'pause_reading'):
                self._remote_transport.pause_reading()
            
            # Brief pause to allow buffers to drain
            await asyncio.sleep(0.1)
            
            # Close remote first
            if self._remote_transport and not self._remote_transport.is_closing():
                self._remote_transport.close()
                await asyncio.sleep(0.1)
            
            # Close local transport
            if self.transport and not self.transport.is_closing():
                self.transport.close()
                
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error during graceful shutdown: {e}")
        finally:
            # Update state
            await proxy_state.update_connection(self._connection_id, "status", "closed")

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection loss."""
        try:
            # Update connection tracking
            end_time = datetime.now(timezone.utc)
            if self._connection_id in self._active_connections:
                self._active_connections[self._connection_id].update({
                    "end_time": end_time,
                    "error": str(exc) if exc else None,
                    "status": "error" if exc else "closed"
                })

                # Log final stats
                stats = self._active_connections[self._connection_id]
                log_connection(self._connection_id, 
                    f"Connection closed. Processed {stats['requests_processed']} requests, "
                    f"received {stats['bytes_received']} bytes")

                # Update state tracking
                asyncio.create_task(proxy_state.update_connection(
                    self._connection_id,
                    "final_stats",
                    {
                        "end_time": end_time,
                        "duration": (end_time - stats["created_at"]).total_seconds(),
                        "bytes_received": stats["bytes_received"],
                        "bytes_sent": stats["bytes_sent"],
                        "requests_processed": stats["requests_processed"]
                    }
                ))
                
            # Initiate graceful shutdown
            if asyncio.get_event_loop().is_running():
                asyncio.create_task(self._graceful_shutdown())
                asyncio.create_task(self._cleanup_connection())

        except Exception as e:
            logger.error(f"[{self._connection_id}] Error during connection cleanup: {e}")
        finally:
            super().connection_lost(exc)

    async def _cleanup_connection(self, delay: int = 30) -> None:
        """Clean up connection tracking with delay."""
        if self._connection_id in self._active_connections:
            log_connection(self._connection_id, f"Connection {self._connection_id} closed")
            
            # Send final events
            await self._record_event("connection", "proxy-browser", "closed")
            
            # Update history entry
            await self._update_history_duration()
            
            # Brief delay before cleanup
            await asyncio.sleep(delay)
            
            # Final cleanup
            self._active_connections.pop(self._connection_id, None)

    async def _update_metrics(self, status: str) -> None:
        """Update connection metrics."""
        try:
            if self._connection_id in self._active_connections:
                self._active_connections[self._connection_id].update({
                    "status": status,
                    "last_update": datetime.now(timezone.utc).timestamp()
                })
                
                # Broadcast update if WebSocket manager is available
                try:
                    from api.proxy.websocket import connection_manager
                    if connection_manager:
                        await connection_manager.broadcast_connection_update(
                            self._active_connections[self._connection_id]
                        )
                except Exception as e:
                    logger.error(f"Failed to broadcast metrics update: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

class TunnelTransport(asyncio.Protocol):
    """Protocol for tunnel connection."""
    
    def __init__(self, client_transport, connection_id: Optional[str] = None, write_buffer_size: int = 256 * 1024):
        self.client_transport = client_transport
        self.connection_id = connection_id or str(uuid4())  # Generate a new ID if not provided
        self.transport = None
        self._write_buffer_size = write_buffer_size
        self._current_buffer_size = 0
        self._high_water = int(write_buffer_size * 0.9)
        self._low_water = int(write_buffer_size * 0.3)
        self._paused = False
        self._drain_task = None
        self._write_tasks = set()
        self._consecutive_timeouts = 0
        self._max_timeouts = 3
        self._write_timeout = 1.0
        self._drain_timeout = 5.0
        self._closing = False
        self._write_monitor_task = None
        self.flow_control = None
        self._last_write_time = 0
        self._write_stall_timeout = 10.0
        self._handshake_complete = False
        self._handshake_buffer = bytearray()
        self._handshake_timeout = 30.0  # 30 seconds for handshake
        self._handshake_start_time = None
        self._bytes_forwarded = 0

        log_connection(self.connection_id, f"TunnelTransport initialized with buffer size {write_buffer_size}")

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle new connection."""
        self.transport = transport
        self._handshake_start_time = time.time()
        
        # Configure socket for better performance
        if hasattr(transport, 'get_extra_info'):
            sock = transport.get_extra_info('socket')
            if sock is not None:
                try:
                    # Set TCP_NODELAY to disable Nagle's algorithm
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    # Set receive buffer size
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)  # 256KB
                    log_connection(self.connection_id, "Socket options configured for performance")
                except socket.error as e:
                    log_connection(self.connection_id, f"Failed to set socket options: {e}")
        
        log_connection(self.connection_id, "TunnelTransport connection established")

    def data_received(self, data: bytes) -> None:
        """Handle received data with special handling for TLS handshake."""
        size = len(data)
        self._bytes_forwarded += size
        log_connection(self.connection_id, f"Received {size} bytes (total: {self._bytes_forwarded})")
        asyncio.create_task(self._handle_data(data))

    async def _handle_data(self, data: bytes) -> None:
        """Async handler for received data."""
        try:
            size = len(data)

            # Special handling for initial TLS handshake
            if not self._handshake_complete:
                # Check for handshake timeout
                if time.time() - self._handshake_start_time > self._handshake_timeout:
                    log_connection(self.connection_id, "TLS handshake timeout")
                    if self.transport:
                        self.transport.close()
                    return

                # Check if this looks like a TLS record
                first_byte = data[0] if data else 0
                log_connection(self.connection_id, f"First byte: {first_byte} (0x{first_byte:02x})")
                
                if first_byte in (20, 21, 22, 23):  # TLS record types
                    record_type = {
                        20: "ChangeCipherSpec",
                        21: "Alert",
                        22: "Handshake",
                        23: "Application Data"
                    }.get(first_byte, "Unknown")
                    
                    log_connection(self.connection_id, f"TLS record type: {record_type} ({first_byte})")
                    # Forward TLS records immediately without buffering
                    if self.client_transport and not self.client_transport.is_closing():
                        self.client_transport.write(data)
                        log_connection(self.connection_id, f"Forwarded TLS record: {size} bytes ({record_type})")
                        
                        # Mark handshake as complete after Application Data or ChangeCipherSpec
                        if first_byte in (20, 23):  # ChangeCipherSpec or Application Data
                            self._handshake_complete = True
                            log_connection(self.connection_id, f"TLS handshake completed ({record_type} received)")
                            # Reset socket options for normal operation
                            if self.transport:
                                sock = self.transport.get_extra_info('socket')
                                if sock:
                                    try:
                                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)  # 128KB
                                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)  # 128KB
                                        log_connection(self.connection_id, "Socket buffers adjusted for normal operation")
                                    except socket.error as e:
                                        log_connection(self.connection_id, f"Failed to adjust socket buffers: {e}")
                        return
                else:
                    log_connection(self.connection_id, f"Non-TLS record received during handshake: {first_byte}")

            # Normal data handling after handshake
            if self._current_buffer_size + size > self._write_buffer_size:
                log_connection(self.connection_id, f"Buffer overflow - current: {self._current_buffer_size}, new: {size}, max: {self._write_buffer_size}")
                if self.transport:
                    self.transport.close()
                return

            self._current_buffer_size += size
            log_connection(self.connection_id, f"Buffering {size} bytes for writing (total: {self._current_buffer_size})")

            if self.client_transport and not self.client_transport.is_closing():
                write_task = asyncio.create_task(self._handle_write(data))
                self._write_tasks.add(write_task)
                write_task.add_done_callback(self._write_tasks.discard)
                log_connection(self.connection_id, f"Created write task for {size} bytes")

                if not self._write_monitor_task or self._write_monitor_task.done():
                    self._write_monitor_task = asyncio.create_task(self._monitor_writes())
                    log_connection(self.connection_id, "Started write monitor task")
            else:
                log_connection(self.connection_id, "Client transport closed")
                if self.transport:
                    self.transport.close()

        except Exception as e:
            log_connection(self.connection_id, f"Data handling error: {e}")
            if self.transport:
                self.transport.close()

    async def _handle_write(self, data: bytes) -> None:
        """Handle writing data with flow control."""
        try:
            async with timeout(self._write_timeout) as cm:
                # Write data to client transport
                if self.client_transport and not self.client_transport.is_closing():
                    self.client_transport.write(data)
                    self._current_buffer_size -= len(data)
                    self._consecutive_timeouts = 0
                else:
                    log_connection(self.connection_id, "Client transport closed during write")
                    if self.transport:
                        self.transport.close()
                    return

                # Check if we need to pause reading
                if self._current_buffer_size >= self._high_water:
                    if hasattr(self.transport, "pause_reading"):
                        self.transport.pause_reading()
                        self._paused = True
                        log_connection(self.connection_id, "Paused reading due to high buffer")

        except asyncio.TimeoutError:
            self._consecutive_timeouts += 1
            log_connection(self.connection_id, f"Write timeout ({self._consecutive_timeouts}/{self._max_timeouts})")
            if self._consecutive_timeouts >= self._max_timeouts:
                log_connection(self.connection_id, "Max write timeouts reached - closing connection")
                if self.transport:
                    self.transport.close()
        except Exception as e:
            log_connection(self.connection_id, f"Write error: {e}")
            if self.transport:
                self.transport.close()

    async def _monitor_writes(self) -> None:
        """Monitor write tasks and handle cleanup."""
        try:
            while self._write_tasks and not self._closing:
                # Wait for all current write tasks
                if self._write_tasks:
                    done, pending = await asyncio.wait(
                        self._write_tasks,
                        timeout=self._write_stall_timeout,
                        return_when=asyncio.ALL_COMPLETED
                    )
                    
                    # Check for timeouts
                    if pending:
                        log_connection(self.connection_id, f"Write stall detected - {len(pending)} pending tasks")
                        for task in pending:
                            task.cancel()
                        if self.transport:
                            self.transport.close()
                        return

                # Resume reading if buffer is low enough
                if self._paused and self._current_buffer_size <= self._low_water:
                    if hasattr(self.transport, "resume_reading"):
                        self.transport.resume_reading()
                        self._paused = False
                        log_connection(self.connection_id, "Resumed reading")

                # Brief pause before next check
                await asyncio.sleep(0.1)

        except Exception as e:
            log_connection(self.connection_id, f"Write monitor error: {e}")
            if self.transport:
                self.transport.close()

    def eof_received(self) -> bool:
        """Handle EOF from the remote end."""
        log_connection(self.connection_id, "EOF received")
        if self.client_transport and not self.client_transport.is_closing():
            self.client_transport.write_eof()
            return True
        return False

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection loss."""
        self._closing = True
        if exc:
            log_connection(self.connection_id, f"Connection lost with error: {exc}")
        else:
            log_connection(self.connection_id, "Connection closed normally")

        # Cancel any pending write tasks
        for task in self._write_tasks:
            task.cancel()

        # Close client transport if still open
        if self.client_transport and not self.client_transport.is_closing():
            self.client_transport.close()
