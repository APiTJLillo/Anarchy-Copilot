"""Custom protocol handler for HTTPS tunneling."""
import asyncio
import errno
import logging
import socket
import weakref
from datetime import datetime, timezone
from typing import Optional, Tuple, Any, Set, Dict, List
from urllib.parse import unquote
from uuid import uuid4
from async_timeout import timeout as async_timeout

from uvicorn.protocols.http.h11_impl import H11Protocol
from sqlalchemy import text, select
import h11
from typing import Dict, List, Optional, Any, Tuple, Callable, TYPE_CHECKING

from database import AsyncSessionLocal
from .flow_control import FlowControl
from .state import proxy_state

if TYPE_CHECKING:
    from .https_intercept_protocol import HttpsInterceptProtocol

logger = logging.getLogger("proxy.core")

def log_connection(conn_id: str, message: str) -> None:
    """Helper for consistent connection logging."""
    logger.debug(f"[Connection {conn_id}] {message}")

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
        
        # Add to state tracking
        asyncio.create_task(proxy_state.add_connection(self._connection_id, conn_info))

    async def _handle_large_write(self) -> None:
        """Handle flow control for large writes."""
        try:
            if self._remote_transport and self.flow_control:
                self.flow_control.pause_reading()
                try:
                    async with async_timeout(1.0) as cm:
                        await asyncio.sleep(0.1)  # Brief pause to let buffers drain
                finally:
                    if not self._remote_transport.is_closing():
                        self.flow_control.resume_reading()
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error handling large write: {e}")
            # Update state on error
            await proxy_state.update_connection(self._connection_id, "error", str(e))

    async def _handle_connect(self, host: str, port: int) -> None:
        """Handle CONNECT request by establishing a tunnel."""
        try:
            # Record connection attempt
            asyncio.create_task(self._record_event("request", "browser-proxy", "pending"))
            
            log_connection(self._connection_id, f"Attempting connection to {host}:{port}")
            loop = asyncio.get_event_loop()

            # Send 200 Connection Established first
            log_connection(self._connection_id, "Sending 200 Connection Established")
            response = (
                b"HTTP/1.1 200 Connection Established\r\n"
                b"Connection: keep-alive\r\n"
                b"\r\n"
            )
            
            try:
                # Send response with timeout protection
                async with async_timeout(5) as cm:
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
                        async with async_timeout(10) as cm:
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

                log_connection(self._connection_id, "Connection established")

                # Switch to tunnel mode
                log_connection(self._connection_id, "Switching to tunnel mode")
                self._in_tunnel_mode = True
                
                # Process any buffered data with timeout protection
                if self._tunnel_buffer:
                    log_connection(self._connection_id, f"Processing {len(self._tunnel_buffer)} bytes of buffered data")
                    try:
                        async with async_timeout(5) as cm:
                            if self._remote_transport and not self._remote_transport.is_closing():
                                self._remote_transport.write(bytes(self._tunnel_buffer))
                    except asyncio.TimeoutError:
                        logger.error(f"[{self._connection_id}] Timeout processing buffered data")
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

    def data_received(self, data: bytes) -> None:
        """Handle incoming data."""
        # Update connection metrics
        self._bytes_received += len(data)
        if self._connection_id in self._active_connections:
            self._active_connections[self._connection_id]["bytes_received"] = self._bytes_received

        # First check if this looks like a CONNECT request
        if data.startswith(b'CONNECT '):
            log_connection(self._connection_id, "Processing CONNECT request")
            try:
                # Store the initial CONNECT data for potential replay
                self._tunnel_buffer.extend(data)
                
                # Parse full HTTP request
                lines = data.split(b'\r\n')
                first_line = lines[0].decode()
                _, target, _ = first_line.split(' ')
                
                # Extract headers
                headers = []
                for line in lines[1:]:
                    if not line:  # Empty line marks end of headers
                        break
                    try:
                        name, value = line.decode().split(':', 1)
                        headers.append((
                            name.strip().lower().encode(),
                            value.strip().encode()
                        ))
                    except ValueError:
                        continue

                # Add required Host header if missing
                if not any(k == b"host" for k, _ in headers):
                    host = target.split(":")[0]  # Extract host from target
                    headers.append((b"host", host.encode()))
                
                # Store headers for history recording
                self._connect_headers = {
                    name.decode(): value.decode()
                    for name, value in headers
                }
                
                # Create h11 request
                log_connection(self._connection_id, f"Creating CONNECT request for {target}")
                request = h11.Request(
                    method=b"CONNECT",
                    target=target.encode(),
                    headers=headers
                )
                asyncio.create_task(self.handle_request(request))
                return
            except Exception as e:
                log_connection(self._connection_id, f"Failed to parse CONNECT request: {e}")
                self._send_error(502, str(e))
                return
        
        # Handle tunnel mode or regular HTTP
        if self._in_tunnel_mode:
            if self._remote_transport and not self._remote_transport.is_closing():
                try:
                    # If we're an HttpsInterceptProtocol instance and handshake isn't complete,
                    # only buffer the data
                    if hasattr(self, '_tls_state') and not self._tls_state.get('handshake_complete', False):
                        log_connection(self._connection_id, f"Buffering {len(data)} bytes during TLS handshake")
                        if len(self._buffer) + len(data) <= 16640:  # Max TLS record size
                            self._buffer.extend(data)
                        else:
                            log_connection(self._connection_id, "Buffer would exceed TLS record size")
                            self._buffer.clear()
                    else:
                        log_connection(self._connection_id, f"Tunneling {len(data)} bytes to remote")
                        self._remote_transport.write(data)
                        # Create task for event recording
                        asyncio.create_task(self._record_event("data", "tunnel", "forwarded", len(data)))
                        # Handle flow control for large writes
                        if len(data) > 65536:
                            asyncio.create_task(self._handle_large_write())
                except Exception as e:
                    log_connection(self._connection_id, f"Tunnel write error: {e}")
                    self.transport.close()
            return
        else:
            # Process as HTTP or buffer based on response sent state
            if not getattr(self, '_connect_response_sent', False):
                # Process using super() to handle HTTP
                log_connection(self._connection_id, "Processing direct HTTP request")
                super().data_received(data)
            else:
                # Already sent 200 response (for CONNECT)
                log_connection(self._connection_id, f"Buffering {len(data)} bytes for tunnel")
                self._tunnel_buffer.extend(data)

    async def handle_request(self, request: h11.Request) -> None:
        """Override request handling to intercept and modify requests."""
        # First update host/port from headers or URL
        if request.target.startswith(b'http://'):
            # Extract host and port from absolute URL
            url = request.target.decode()
            from urllib.parse import urlparse
            parsed = urlparse(url)
            self._host = parsed.hostname
            self._port = parsed.port or 80
            # Keep the path, query, and fragment components
            path = parsed.path or b'/'
            if parsed.query:
                path += f"?{parsed.query}"
            if parsed.fragment:
                path += f"#{parsed.fragment}"
            request.target = path.encode() if isinstance(path, str) else path
        else:
            # Extract from Host header
            host_header = None
            for name, value in request.headers:
                if name.lower() == b'host':
                    host_header = value.decode()
                    break

        if host_header:
            if ':' in host_header:
                self._host, port = host_header.rsplit(':', 1)
                self._port = int(port)
            else:
                self._host = host_header
                self._port = 80 if request.method != b"CONNECT" else 443

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
                
                log_connection(self._connection_id, f"Establishing tunnel to {host}:{port}")
                await self._handle_connect(host, port)
                
            except asyncio.CancelledError:
                log_connection(self._connection_id, "CONNECT tunnel cancelled")
                asyncio.create_task(self._record_event("connection", "browser-proxy", "cancelled"))
                self.transport.close()
            except Exception as e:
                log_connection(self._connection_id, f"Failed to handle CONNECT: {e}")
                asyncio.create_task(self._record_event("connection", "browser-proxy", "error"))
                self._send_error(502, str(e))
        else:
            log_connection(self._connection_id, "Processing non-CONNECT request")
            self._requests_processed += 1
            if self._connection_id in self._active_connections:
                self._active_connections[self._connection_id]["requests_processed"] = self._requests_processed

            # Reconstruct URL if needed
            if not request.target.startswith(b'http://') and not request.target.startswith(b'https://'):
                scheme = b'http'
                if not self._host:
                    # Extract host from headers if not already set
                    for name, value in request.headers:
                        if name.lower() == b'host':
                            self._host = value.decode().split(':')[0]
                            break
                
                if self._host:
                    path = request.target
                    if not path.startswith(b'/'):
                        path = b'/' + path
                    request.target = f"{scheme}://{self._host}:{self._port}{path.decode()}".encode()
                    log_connection(self._connection_id, f"Reconstructed URL: {request.target.decode()}")

            # Update headers for proxy forwarding
            new_headers = []
            has_connection = False
            has_host = False
            
            for name, value in request.headers:
                name_lower = name.lower()
                if name_lower == b'connection':
                    has_connection = True
                    if value.lower() != b'close':
                        value = b'close'  # Force close to prevent keep-alive issues
                elif name_lower == b'host':
                    has_host = True
                new_headers.append((name, value))

            if not has_connection:
                new_headers.append((b'connection', b'close'))
            if not has_host and self._host:
                host_value = f"{self._host}:{self._port}".encode() if self._port != 80 else self._host.encode()
                new_headers.append((b'host', host_value))

            # Add proxy-related headers
            new_headers.append((b'x-forwarded-for', self.transport.get_extra_info('peername')[0].encode()))
            new_headers.append((b'x-forwarded-proto', b'http'))
            new_headers.append((b'x-forwarded-port', str(self._port).encode()))

            request.headers = new_headers
            await super().handle_request(request)

    async def _handle_connect(self, host: str, port: int) -> None:
        """Handle CONNECT request by establishing a tunnel."""
        try:
            # Record connection attempt
            asyncio.create_task(self._record_event("request", "browser-proxy", "pending"))
            
            log_connection(self._connection_id, f"Attempting connection to {host}:{port}")
            loop = asyncio.get_event_loop()

            # Send 200 Connection Established only if not in HTTPS intercept mode
            # Check if we're in the HttpsInterceptProtocol class
            if not hasattr(self, '_intercept_enabled'):
                log_connection(self._connection_id, "Sending 200 Connection Established")
                response = (
                    b"HTTP/1.1 200 Connection Established\r\n"
                    b"Connection: keep-alive\r\n"
                    b"\r\n"
                )
                if self.transport and not self.transport.is_closing():
                    self.transport.write(response)
                    self._connect_response_sent = True  # Mark response as sent
                    await asyncio.sleep(0.1)  # Brief pause to ensure response is sent
                    
                # Record response sent
                asyncio.create_task(self._record_event("response", "proxy-browser", "success"))
            
            try:
                # Create the remote connection
                self._remote_transport, self._remote_protocol = await loop.create_connection(
                    lambda: TunnelTransport(
                        client_transport=self.transport,
                        connection_id=self._connection_id,
                        write_buffer_size=256 * 1024  # 256KB buffer
                    ),
                    host=host,
                    port=port
                )
                log_connection(self._connection_id, f"Remote transport created with ID {id(self._remote_transport)}")
                
                # Record successful connection
                asyncio.create_task(self._record_event("request", "proxy-web", "success"))

                # Record HTTPS connection in proxy history
                self._tunnel_start_time = datetime.utcnow()
                await self._create_history_entry(host, port)

                log_connection(self._connection_id, "Connection established")

                # Switch to tunnel mode
                log_connection(self._connection_id, "Switching to tunnel mode")
                self._in_tunnel_mode = True
                
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
            if self._remote_transport:
                self._remote_transport.close()

    async def _create_history_entry(self, host: str, port: int) -> None:
        """Create history entry for the connection."""
        try:
            from api.proxy.database_models import ProxySession, ProxyHistoryEntry
            
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(ProxySession)
                    .where(ProxySession.is_active == True)
                    .order_by(ProxySession.start_time.desc())
                    .limit(1)
                )
                active_session = result.scalar_one_or_none()
                
                if active_session:
                    history_entry = ProxyHistoryEntry(
                        session_id=active_session.id,
                        timestamp=self._tunnel_start_time,
                        method="CONNECT",
                        url=f"https://{host}:{port}",
                        request_headers=self._connect_headers,
                        request_body=None,
                        response_status=200,
                        response_headers={"Connection": "keep-alive"},
                        response_body=None,
                        duration=None,
                        is_intercepted=False,
                        tags=["HTTPS"],
                        notes=f"HTTPS tunnel established to {host}:{port}"
                    )
                    
                    db.add(history_entry)
                    await db.commit()
                    await db.refresh(history_entry)
                    self._history_entry_id = history_entry.id
                    log_connection(self._connection_id, f"History entry created with ID: {self._history_entry_id}")
        except Exception as e:
            log_connection(self._connection_id, f"Failed to create history entry: {e}")

    async def _update_history_duration(self) -> None:
        """Update history entry with connection duration."""
        try:
            if self._tunnel_start_time and self._history_entry_id:
                end_time = datetime.utcnow()
                duration = (end_time - self._tunnel_start_time).total_seconds()

                async with AsyncSessionLocal() as db:
                    await db.execute(
                        text("""
                        UPDATE proxy_history 
                        SET duration = :duration,
                            notes = notes || ' (Connection closed)'
                        WHERE id = :id
                        """),
                        {"duration": duration, "id": self._history_entry_id}
                    )
                    await db.commit()
                    log_connection(self._connection_id, f"Updated history duration: {duration:.2f}s")
        except Exception as e:
            log_connection(self._connection_id, f"Failed to update history duration: {e}")

    def _parse_authority(self, authority: str) -> Tuple[str, int]:
        """Parse host and port from CONNECT target."""
        if not authority:
            raise ValueError("Empty authority")

        # Handle potential URL encoding
        authority = unquote(authority)
        
        # Split host:port
        if ":" in authority:
            host, port_str = authority.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                raise ValueError(f"Invalid port: {port_str}")
        else:
            host = authority
            port = 443  # Default HTTPS port

        return host, port

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
        self.transport = None
        self.client_transport = client_transport
        self.connection_id = connection_id
        
        # Buffer configuration
        self._write_buffer_size = write_buffer_size
        self._high_water = int(write_buffer_size * 0.8)
        self._low_water = int(write_buffer_size * 0.2)
        self._current_buffer_size = 0
        self._paused = False

        log_connection(connection_id, "TunnelTransport initialized")

    def connection_made(self, transport: asyncio.Transport):
        """Store the transport for use."""
        self.transport = transport
        self.flow_control = FlowControl(transport)
        self._drain_waiter = None
        self._handshake_complete = False
        self._handshake_buffer_size = 0  # Track handshake data size
        
        # Configure socket for optimal performance
        sock = transport.get_extra_info('socket')
        if sock is not None:
            try:
                # Set TCP_NODELAY to disable Nagle's algorithm
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                # Enable TCP keepalive with shorter timeouts during handshake
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                if hasattr(socket, 'TCP_KEEPIDLE'):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)  # Shorter initial delay
                if hasattr(socket, 'TCP_KEEPINTVL'):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
                if hasattr(socket, 'TCP_KEEPCNT'):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
                    
                # Larger buffers for handshake
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)  # 128KB
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)  # 128KB
                
            except (AttributeError, socket.error) as e:
                log_connection(self.connection_id, f"Socket option error: {e}")

            # Set write buffer limits - higher during handshake
            if hasattr(transport, 'set_write_buffer_limits'):
                transport.set_write_buffer_limits(high=524288)  # 512KB for handshake
        
        log_connection(self.connection_id, f"TunnelTransport connection established")

    def _may_be_tls_shutdown(self, data: bytes) -> bool:
        """Check if data looks like a TLS shutdown record."""
        # TLS shutdown record is typically 21 (alert) followed by 3,3 (TLS 1.2) or 3,4 (TLS 1.3)
        if len(data) == 7 and data[0] == 21 and data[1] == 3:
            return True
        return False

    def data_received(self, data):
        """Forward data to client with handshake-aware flow control."""
        size = len(data)
        log_connection(self.connection_id, f"Remote received {size} bytes")

        try:
            if not self.client_transport or self.client_transport.is_closing():
                return

            # Detect and handle initial connection messages
            first_byte = data[0] if data else 0
            is_initial_handshake = (
                not self._handshake_complete and 
                self._handshake_buffer_size == 0 and
                (first_byte in (20, 22))  # ChangeCipherSpec or Handshake
            )

            if is_initial_handshake:
                log_connection(self.connection_id, "Initial TLS handshake message detected")
                # Allow more data during handshake
                if hasattr(self.transport, 'set_write_buffer_limits'):
                    self.transport.set_write_buffer_limits(high=1048576)  # 1MB during handshake
            
            try:
                # Track handshake progress
                if not self._handshake_complete:
                    self._handshake_buffer_size += len(data)
                    is_still_handshake = (
                        len(data) <= 16384 and  # Max TLS record size
                        self._handshake_buffer_size < 32768 and  # Total handshake size
                        self._current_buffer_size < 65536  # Buffer limit during handshake
                    )
                    
                    # Detect handshake completion
                    if not is_still_handshake or self._handshake_buffer_size >= 32768:
                        self._handshake_complete = True
                        log_connection(self.connection_id, "TLS handshake phase complete")
                        # Adjust buffer limits after handshake
                        if hasattr(self.transport, 'set_write_buffer_limits'):
                            self.transport.set_write_buffer_limits(high=262144)  # 256KB
                        # Ensure write buffer is clear
                        if self._drain_waiter is not None:
                            self._drain_waiter.cancel()
                            self._drain_waiter = None
                        # Reset buffer tracking after handshake
                        self._current_buffer_size = 0
                        if self._paused:
                            self._paused = False
                            self.flow_control.resume_reading()
                        log_connection(self.connection_id, "Reset buffer state after handshake")
            except Exception as e:
                log_connection(self.connection_id, f"Error during handshake phase: {e}")
                if self.transport:
                    self.transport.close()
                return
            
            # Update buffer size
            self._current_buffer_size += size
            
            # Write data
            if self.client_transport and not self.client_transport.is_closing():
                self.client_transport.write(data)
                
                if not self._handshake_complete:
                    # During handshake, be more lenient with buffer limits
                    if self._current_buffer_size > self._high_water * 2:
                        log_connection(self.connection_id, "Buffer full during handshake")
                        self._paused = True
                        self.flow_control.pause_reading()
                    else:
                        log_connection(self.connection_id, f"Forwarding handshake data ({len(data)} bytes)")
                else:
                    # Normal operation flow control
                    if self._current_buffer_size > self._high_water:
                        if self._drain_waiter is None:
                            self._drain_waiter = asyncio.create_task(self._wait_for_drain())
                            log_connection(self.connection_id, "Creating drain waiter for flow control")
                        if not self._paused:
                            self.flow_control.pause_reading()
                            self._paused = True
                            log_connection(self.connection_id, "Pausing reads for flow control")

            else:
                log_connection(self.connection_id, "Client transport closed")
                if self.transport:
                    self.transport.close()
        except Exception as e:
            log_connection(self.connection_id, f"Forward error: {e}")
            if self.transport:
                self.transport.close()

    async def _wait_for_drain(self) -> None:
        """Wait for write buffer to drain."""
        try:
            if hasattr(self.client_transport, 'get_write_buffer_size'):
                # Wait for buffer to drain below low water mark
                while self.client_transport.get_write_buffer_size() > self._low_water:
                    await asyncio.sleep(0.01)
            else:
                # Fallback: just wait a bit
                await asyncio.sleep(0.1)
            
            # Reset buffer state
            self._current_buffer_size = 0
            if self._paused:
                self._paused = False
                self.flow_control.resume_reading()
            
            # Clear drain waiter
            self._drain_waiter = None
            
        except Exception as e:
            log_connection(self.connection_id, f"Drain wait error: {e}")
            self._drain_waiter = None

    async def _check_buffer_size(self):
        """Monitor buffer size and manage flow control."""
        try:
            # Only check if not already draining
            if self._drain_waiter is None:
                # Create new drain waiter if needed
                if self._current_buffer_size > self._high_water:
                    self._drain_waiter = asyncio.create_task(self._wait_for_drain())
                    
                # Check if we should resume
                elif self._current_buffer_size <= self._low_water and self._paused:
                    self._paused = False
                    self.flow_control.resume_reading()
                    log_connection(self.connection_id, "Resuming reading - buffer drained")
            
        except Exception as e:
            log_connection(self.connection_id, f"Buffer check error: {e}")

    def resume_writing(self) -> None:
        """Resume writing when buffer is drained."""
        if self._paused:
            self._paused = False
            self._current_buffer_size = 0  # Reset buffer size on resume
            if self.flow_control:
                self.flow_control.resume_reading()
                # Create drain monitor task
                loop = asyncio.get_event_loop()
                if not self._drain_waiter:
                    self._drain_waiter = loop.create_task(self._wait_for_drain())
            log_connection(self.connection_id, "Resumed writing")

    def pause_writing(self) -> None:
        """Pause writing when buffers are full."""
        if not self._paused:
            self._paused = True
            if self.flow_control:
                self.flow_control.pause_reading()
                # Schedule buffer check
                loop = asyncio.get_event_loop()
                loop.create_task(self._check_buffer_size())
            log_connection(self.connection_id, "Paused writing - buffer full")

    def eof_received(self) -> bool:
        """Handle EOF from remote peer."""
        log_connection(self.connection_id, "Remote EOF received")
        
        # During handshake, delay EOF to allow buffered data to be processed
        if not self._handshake_complete:
            log_connection(self.connection_id, "Delaying EOF during handshake")
            if self.client_transport and not self.client_transport.is_closing():
                # Schedule deferred EOF
                loop = asyncio.get_event_loop()
                loop.call_later(0.5, self._delayed_eof)
                return True  # Keep connection open temporarily
            
        # Normal EOF handling for established connections
        if self.client_transport and not self.client_transport.is_closing():
            try:
                if hasattr(self.client_transport, 'write_eof'):
                    self.client_transport.write_eof()
            except Exception as e:
                log_connection(self.connection_id, f"Error sending EOF: {e}")
        return False  # Don't keep connection open

    def _delayed_eof(self) -> None:
        """Handle delayed EOF after handshake attempt."""
        try:
            if self.client_transport and not self.client_transport.is_closing():
                # Drain any remaining buffered data
                self._current_buffer_size = 0
                if self._paused:
                    self._paused = False
                    self.flow_control.resume_reading()
                
                # Signal EOF
                if hasattr(self.client_transport, 'write_eof'):
                    self.client_transport.write_eof()
        except Exception as e:
            log_connection(self.connection_id, f"Error in delayed EOF: {e}")
        finally:
            # Ensure transport is closed
            if self.transport and not self.transport.is_closing():
                self.transport.close()

    def connection_lost(self, exc):
        """Handle connection loss."""
        try:
            # Log error if present
            if exc:
                log_connection(self.connection_id, f"Remote connection lost with error: {exc}")
            else:
                log_connection(self.connection_id, "Remote connection closed normally")
            
            # Cancel any pending drain waiter
            if self._drain_waiter is not None:
                self._drain_waiter.cancel()
                self._drain_waiter = None

            # Ensure client transport is closed gracefully
            if self.client_transport and not self.client_transport.is_closing():
                try:
                    # Allow time for any pending writes to complete
                    if not self._handshake_complete:
                        # Quick close during handshake
                        self.client_transport.close()
                    else:
                        # During normal operation, schedule graceful close
                        loop = asyncio.get_event_loop()
                        loop.call_later(0.1, self.client_transport.close)
                except Exception as e:
                    log_connection(self.connection_id, f"Error during client transport close: {e}")
                    # Force close on error
                    self.client_transport.abort()
        except Exception as e:
            log_connection(self.connection_id, f"Error during connection cleanup: {e}")
        finally:
            # Clear references
            self.transport = None
            self.client_transport = None
