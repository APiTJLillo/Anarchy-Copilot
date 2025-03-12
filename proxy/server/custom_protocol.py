"""Custom protocol handler for HTTPS tunneling."""
import asyncio
import errno
import logging
import socket
import time
import weakref
from datetime import datetime, timezone
from typing import Optional, Tuple, Any, Set, Dict, List
from urllib.parse import unquote
from uuid import uuid4
from async_timeout import timeout

from uvicorn.protocols.http.h11_impl import H11Protocol as BaseH11Protocol
from sqlalchemy import text, select
import h11
from typing import Dict, List, Optional, Any, Tuple, Callable, TYPE_CHECKING

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

class H11Protocol(BaseH11Protocol):
    """Custom H11Protocol that supports async data_received."""
    
    async def data_received(self, data: bytes) -> None:
        """Handle incoming data asynchronously."""
        try:
            if self.conn.their_state in {h11.DONE, h11.MUST_CLOSE, h11.CLOSED}:
                return
            
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
        except Exception as exc:
            msg = "Invalid HTTP request received."
            self._send_error_response(400, msg)
            logger.warning(msg, exc_info=True)

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
    
    def _send_error(self, status_code: int, reason: str) -> None:
        """Send an HTTP error response to the client."""
        try:
            if self.transport and not self.transport.is_closing():
                error_response = (
                    f"HTTP/1.1 {status_code} {reason}\r\n"
                    f"Connection: close\r\n"
                    f"Content-Type: text/plain\r\n"
                    f"Content-Length: {len(reason)}\r\n"
                    f"\r\n"
                    f"{reason}"
                ).encode()
                self.transport.write(error_response)
                log_connection(self._connection_id, f"Sent error response: {status_code} {reason}")
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to send error response: {e}")

    # Class-level connection tracking
    _active_connections: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tunnel_tasks: Set[asyncio.Task] = set()
        self._remote_transport = None
        self._remote_protocol = None
        self._tunnel_start_time: Optional[datetime] = None
        self._history_entry_id: Optional[int] = None
        self._connect_headers: Dict[str, str] = {}
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._in_tunnel_mode = False
        self._tunnel_buffer = bytearray()
        self._buffer = bytearray()  # Buffer for TLS handshake data
        self._connect_response_sent = False
        self.flow_control = None
        
        # Connection tracking initialization
        self._connection_id = str(uuid4())
        self._bytes_received = 0
        self._bytes_sent = 0
        self._requests_processed = 0
        self._events: List[Dict[str, Any]] = []
        self._event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._event_processor = None

        log_connection(self._connection_id, "Protocol initialized")

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
            await super().data_received(data)
            
        except Exception as exc:
            if not self._in_tunnel_mode:
                msg = "Invalid HTTP request received."
                self._send_error_response(400, msg)
                logger.warning(msg, exc_info=True)

    def set_tunnel_mode(self, enabled: bool = True) -> None:
        """Set protocol to tunnel mode after CONNECT is established."""
        self._in_tunnel_mode = enabled
        if enabled:
            # Reset HTTP state
            self.conn = h11.Connection(h11.SERVER)
            logger.debug(f"[{self._connection_id}] Switched to tunnel mode")

    async def handle_request(self, request: h11.Request) -> None:
        """Override request handling to intercept and modify requests."""
        # Handle CONNECT method
        if request.method == b"CONNECT":
            try:
                target = request.target.decode()
                log_connection(self._connection_id, f"Handling CONNECT request: {target}")
                host, port = self._parse_authority(target)
                self._host = host
                self._port = port
                
                # Update connection info
                if self._connection_id in self._active_connections:
                    self._active_connections[self._connection_id].update({
                        "host": host,
                        "port": port
                    })
                
                # Only handle CONNECT if we're not in HTTPS intercept mode
                if not hasattr(self, '_intercept_enabled'):
                    log_connection(self._connection_id, f"Establishing tunnel to {host}:{port}")
                    await self._handle_connect(host, port)
                else:
                    # Let the parent class handle HTTPS interception
                    await super().handle_request(request)
                
            except asyncio.CancelledError:
                log_connection(self._connection_id, "CONNECT tunnel cancelled")
                asyncio.create_task(self._record_event("connection", "browser-proxy", "cancelled"))
                self.transport.close()
            except Exception as e:
                log_connection(self._connection_id, f"Failed to handle CONNECT: {e}")
                asyncio.create_task(self._record_event("connection", "browser-proxy", "error"))
                self._send_error(502, str(e))
        else:
            # Handle non-CONNECT requests
            await super().handle_request(request)

    async def _handle_connect(self, host: str, port: int) -> None:
        """Handle CONNECT request by establishing a tunnel."""
        try:
            # Record connection attempt
            asyncio.create_task(self._record_event("request", "browser-proxy", "pending"))
            
            log_connection(self._connection_id, f"Attempting connection to {host}:{port}")
            loop = asyncio.get_event_loop()

            # Send 200 Connection Established
            log_connection(self._connection_id, "Sending 200 Connection Established")
            response = (
                b"HTTP/1.1 200 Connection Established\r\n"
                b"Connection: keep-alive\r\n"
                b"\r\n"
            )
            
            try:
                # Send response with timeout protection
                async with timeout(5) as cm:
                    if self.transport and not self.transport.is_closing():
                        self.transport.write(response)
                        self._connect_response_sent = True  # Mark response as sent
                        await asyncio.sleep(0.1)  # Brief pause to ensure response is sent
            except asyncio.TimeoutError:
                logger.error(f"[{self._connection_id}] Timeout sending 200 response")
                raise
                
            # Record response sent
            asyncio.create_task(self._record_event("response", "proxy-browser", "success"))
            
            try:
                # Create the remote connection with retries
                for attempt in range(3):
                    try:
                        async with timeout(10) as cm:
                            self._remote_transport, self._remote_protocol = await loop.create_connection(
                                lambda: TunnelTransport(
                                    client_transport=self.transport,
                                    connection_id=self._connection_id,
                                    write_buffer_size=256 * 1024  # 256KB buffer
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
                        
                log_connection(self._connection_id, f"Remote transport created with ID {id(self._remote_transport)}")
                
                # Record successful connection
                asyncio.create_task(self._record_event("request", "proxy-web", "success"))

                # Record HTTPS connection in proxy history
                self._tunnel_start_time = datetime.utcnow()
                await self._create_history_entry(host, port)

                # Switch to tunnel mode
                log_connection(self._connection_id, "Switching to tunnel mode")
                self._in_tunnel_mode = True
                self.set_tunnel_mode(True)  # Important: This resets the HTTP state
                
                # Process any buffered data
                if self._tunnel_buffer:
                    log_connection(self._connection_id, f"Processing {len(self._tunnel_buffer)} bytes of buffered data")
                    if self._remote_transport and not self._remote_transport.is_closing():
                        self._remote_transport.write(bytes(self._tunnel_buffer))
                    self._tunnel_buffer.clear()

            except Exception as e:
                log_connection(self._connection_id, f"Remote connection failed: {str(e)}")
                raise

        except Exception as e:
            log_connection(self._connection_id, f"Tunnel error: {e}")
            asyncio.create_task(self._record_event("connection", "proxy-web", "error"))
            self._send_error(502, str(e))
            if self._remote_transport and not self._remote_transport.is_closing():
                self._remote_transport.close()

    async def _create_history_entry(self, host: str, port: int) -> None:
        """Create a history entry for this connection."""
        try:
            if self._connection_id in self._active_connections:
                conn_info = self._active_connections[self._connection_id]
                session_id = conn_info.get("session_id")
                
                if session_id:
                    async with AsyncSessionLocal() as db:
                        history_entry = ProxyHistoryEntry(
                            session_id=session_id,
                            timestamp=datetime.utcnow(),
                            method="CONNECT",
                            url=f"{host}:{port}",
                            request_headers=self._connect_headers,
                            request_body=None,
                            response_status=200,
                            response_headers={"Connection": "Upgrade"},
                            response_body=None,
                            duration=0,  # Will be updated when connection closes
                            is_intercepted=False,
                            tags=[],
                            notes=None
                        )
                        db.add(history_entry)
                        await db.commit()
                        await db.refresh(history_entry)
                        self._history_entry_id = history_entry.id
                        logger.debug(f"[{self._connection_id}] Created history entry {history_entry.id}")
                else:
                    logger.warning(f"[{self._connection_id}] No active session found for connection")
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to create history entry: {e}")

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

class TunnelTransport(asyncio.Protocol):
    """Protocol for tunnel connection."""
    
    def __init__(self, client_transport, connection_id: str, write_buffer_size: int = 256 * 1024):
        self.client_transport = client_transport
        self.connection_id = connection_id
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

        log_connection(connection_id, f"TunnelTransport initialized with buffer size {write_buffer_size}")

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
