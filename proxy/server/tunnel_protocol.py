"""Protocol implementation for tunneling connections."""
import asyncio
import logging
import socket
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List, Protocol, runtime_checkable, Set
from async_timeout import timeout as async_timeout
import time
import uuid

from .flow_control import FlowControl
from .state import proxy_state

logger = logging.getLogger("proxy.core")

@runtime_checkable
class TransferCallback(Protocol):
    """Callback interface for transfer tracking."""
    def __call__(self, nbytes: int) -> None: ...

class TunnelProtocol(asyncio.Protocol):
    """Protocol for handling tunneled connections with flow control and metrics."""

    # Transfer callback types
    on_data_sent: Optional[TransferCallback] = None
    on_data_received: Optional[TransferCallback] = None
    
    def __init__(self, client_transport: asyncio.Transport, flow_control: FlowControl, 
                 connection_id: str, buffer_size: int = 262144,  # Increased buffer size
                 metrics_interval: float = 0.1,
                 write_limit: int = 1048576,  # Increased write limit
                 write_interval: float = 0.0001):  # Decreased write interval
        """Initialize tunnel protocol."""
        super().__init__()
        self.client_transport = client_transport
        self._transport = None  # Server transport
        self.flow_control = flow_control
        self._connection_id = connection_id
        self._buffer_size = buffer_size
        self._metrics_interval = metrics_interval
        self._write_limit = write_limit
        self._write_interval = write_interval
        self._write_queue = asyncio.Queue()
        self._write_task = None
        self._closing = False
        self._bytes_sent = 0
        self._bytes_received = 0
        self._last_activity = time.time()
        
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
        self._in_tunnel_mode = False
        self._tunnel_buffer = bytearray()
        self._buffer = bytearray()  # Buffer for TLS handshake data
        self._connect_response_sent = False
        
        # Connection tracking initialization
        self._connection_id = str(uuid.uuid4())
        self._requests_processed = 0
        self._events: List[Dict[str, Any]] = []
        self._event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._event_processor = None
        self._stall_timeout = 10.0  # 10 seconds stall detection
        self._monitor_task = None

        logger.debug(f"[{self._connection_id}] TunnelProtocol initialized")
    
    @property
    def transport(self) -> Optional[asyncio.Transport]:
        """Get the current transport."""
        return self._transport
    
    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle new connection."""
        super().connection_made(transport)
        self.transport = transport
        self._tunnel_start_time = datetime.now()
        
        # Configure socket
        sock = transport.get_extra_info('socket')
        if sock:
            try:
                # Set TCP_NODELAY to disable Nagle's algorithm
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # Set receive buffer size
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
                logger.debug(f"[{self._connection_id}] Socket configured for performance")
            except socket.error as e:
                logger.warning(f"[{self._connection_id}] Failed to configure socket: {e}")
        
        # Start activity monitor
        self._monitor_task = asyncio.create_task(self._monitor_activity())
        logger.info(f"[{self._connection_id}] New connection established")
        
        self._write_task = asyncio.create_task(self._process_write_queue())
        logger.debug(f"[{self._connection_id}] Tunnel transport established")
        
        if hasattr(transport, 'get_write_buffer_size'):
            transport.set_write_buffer_limits(high=self._buffer_size)
        
        # Log connection
        logger.debug(f"[{self._connection_id}] Connection made to {transport.get_extra_info('peername')}")
        
        asyncio.create_task(self._update_metrics("connected"))
    
    async def _monitor_activity(self):
        """Monitor connection for stalls."""
        while not self.transport.is_closing():
            await asyncio.sleep(1.0)
            now = time.time()
            if now - self._last_activity > self._stall_timeout:
                logger.warning(
                    f"[{self._connection_id}] Connection stall detected - "
                    f"No activity for {now - self._last_activity:.1f}s"
                )
            if self._in_tunnel_mode:
                logger.debug(
                    f"[{self._connection_id}] Tunnel stats: "
                    f"Sent={self._bytes_sent}, "
                    f"Received={self._bytes_received}, "
                    f"Buffer={len(self._tunnel_buffer)}"
                )

    def data_received(self, data: bytes) -> None:
        """Handle received data with detailed logging."""
        size = len(data)
        self._bytes_received += size
        self._last_activity = time.time()
        
        logger.debug(
            f"[{self._connection_id}] Received {size} bytes "
            f"(total: {self._bytes_received}, tunnel_mode: {self._in_tunnel_mode})"
        )

        if self._in_tunnel_mode:
            # Check if this looks like a TLS record
            if data and len(data) >= 5:  # Minimum TLS record size
                record_type = data[0]
                version = (data[1] << 8) | data[2]
                length = (data[3] << 8) | data[4]
                
                record_types = {
                    20: "ChangeCipherSpec",
                    21: "Alert",
                    22: "Handshake",
                    23: "Application"
                }
                
                record_name = record_types.get(record_type, "Unknown")
                logger.debug(
                    f"[{self._connection_id}] TLS Record: "
                    f"Type={record_name}({record_type}), "
                    f"Version=0x{version:04x}, "
                    f"Length={length}"
                )

            # Forward data to tunnel
            if self._remote_transport and not self._remote_transport.is_closing():
                try:
                    self._remote_transport.write(data)
                    logger.debug(f"[{self._connection_id}] Forwarded {size} bytes to tunnel")
                except Exception as e:
                    logger.error(f"[{self._connection_id}] Failed to forward data: {e}")
            else:
                logger.warning(f"[{self._connection_id}] No remote transport available")
        else:
            logger.debug(f"[{self._connection_id}] Buffering {size} bytes (not in tunnel mode)")
            self._buffer.extend(data)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection loss with cleanup."""
        duration = None
        if self._tunnel_start_time:
            duration = (datetime.now() - self._tunnel_start_time).total_seconds()
        
        if exc:
            logger.error(f"[{self._connection_id}] Connection lost with error: {exc}")
        else:
            logger.info(f"[{self._connection_id}] Connection closed normally")
            
        logger.info(
            f"[{self._connection_id}] Connection statistics:\n"
            f"  Duration: {duration:.1f}s\n"
            f"  Bytes sent: {self._bytes_sent}\n"
            f"  Bytes received: {self._bytes_received}\n"
            f"  Requests processed: {self._requests_processed}"
        )
        
        # Clean up
        if self._monitor_task:
            self._monitor_task.cancel()
        
        for task in self._tunnel_tasks:
            task.cancel()
        
        if self._remote_transport:
            self._remote_transport.close()
            
        super().connection_lost(exc)
    
    async def _process_write_queue(self) -> None:
        """Process queued writes with flow control."""
        try:
            consecutive_errors = 0
            while not self._closing:
                try:
                    # Process chunks in batches with timeout
                    chunks: List[bytes] = []
                    total_size = 0
                    
                    # Try to get multiple chunks if available
                    try:
                        async with async_timeout(0.1):  # Reduced timeout for responsiveness
                            while not self._closing and total_size < self._write_limit:
                                try:
                                    data = self._write_queue.get_nowait()
                                    chunks.append(data)
                                    total_size += len(data)
                                    if total_size >= self._write_limit:
                                        break
                                except asyncio.QueueEmpty:
                                    break
                    except asyncio.TimeoutError:
                        # Timeout is expected when queue is empty
                        pass
                    
                    if chunks:
                        # Write chunks with timeout protection
                        try:
                            async with async_timeout(5.0):  # Increased timeout for writes
                                await self._write_chunks(chunks, total_size)
                                
                                # Reset error counter on successful write
                                if consecutive_errors > 0:
                                    consecutive_errors = 0
                                    logger.debug(f"[{self._connection_id}] Recovered from write errors")
                                
                                # Resume reading if queue has space
                                if (self._write_queue.qsize() < self._write_queue.maxsize / 2 and 
                                    hasattr(self.transport, "resume_reading")):
                                    self.transport.resume_reading()
                                    logger.debug(f"[{self._connection_id}] Resumed reading")
                                    
                        except asyncio.TimeoutError:
                            logger.error(f"[{self._connection_id}] Write operation timed out")
                            consecutive_errors += 1
                            
                            # Put chunks back in queue if possible
                            for chunk in reversed(chunks):
                                try:
                                    self._write_queue.put_nowait(chunk)
                                except asyncio.QueueFull:
                                    logger.error(f"[{self._connection_id}] Write queue full, dropping chunk")
                            
                            if consecutive_errors > 5:
                                raise RuntimeError("Too many consecutive write timeouts")
                            
                            # Add backoff delay
                            await asyncio.sleep(min(0.1 * consecutive_errors, 1.0))
                            continue
                    else:
                        # No data to write, brief sleep to prevent busy loop
                        await asyncio.sleep(0.001)
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"[{self._connection_id}] Write queue error: {e}")
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        break
                    await asyncio.sleep(0.1)
                    continue
                    
        except Exception as e:
            logger.error(f"[{self._connection_id}] Write queue processor failed: {e}")
        finally:
            # Ensure reading is resumed
            if hasattr(self.transport, "resume_reading"):
                self.transport.resume_reading()

    async def _write_chunks(self, chunks: List[bytes], total_size: int) -> None:
        """Write chunks to target transport with proper flow control."""
        try:
            target = self._get_target_transport()
            if not target or target.is_closing():
                raise RuntimeError("Target transport not available")
            
            # Write each chunk
            for chunk in chunks:
                # Check write buffer size
                if hasattr(target, 'get_write_buffer_size'):
                    while target.get_write_buffer_size() > self._buffer_size:
                        await asyncio.sleep(0.001)
                
                # Write chunk
                target.write(chunk)
                
                # Update metrics
                if self.on_data_sent:
                    self.on_data_sent(len(chunk))
                
                # Small yield to prevent blocking
                await asyncio.sleep(0)
            
            logger.debug(f"[{self._connection_id}] Wrote {total_size} bytes in {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error writing chunks: {e}")
            raise

    def _get_target_transport(self) -> Optional[asyncio.Transport]:
        """Get the appropriate target transport for forwarding."""
        if hasattr(self, '_server_transport') and self.transport == self._server_transport:
            return self.client_transport
        return self._server_transport if hasattr(self, '_server_transport') else self.transport

    def eof_received(self) -> Optional[bool]:
        """Handle EOF from the remote end."""
        try:
            if self.client_transport and not self.client_transport.is_closing():
                # Try to properly close the write side
                if hasattr(self.client_transport, 'write_eof'):
                    self.client_transport.write_eof()
                
            asyncio.create_task(self._update_metrics("eof_received"))
            return True
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error handling EOF: {e}")
            asyncio.create_task(self._cleanup(error=str(e)))
            return False

    async def _update_metrics(self, status: str, **kwargs) -> None:
        """Update connection metrics with rate limiting."""
        try:
            now = datetime.now().timestamp()
            
            # Check if we should rate limit updates
            if status not in ["error", "closed", "eof_received"]:  # Always send important states
                if now - self._last_metrics_update < self._metrics_interval:
                    # Queue update for later
                    self._pending_updates.update(kwargs)
                    self._pending_updates["status"] = status
                    return
            
            # Include any pending updates
            if self._pending_updates:
                kwargs.update(self._pending_updates)
                self._pending_updates.clear()
            
            metrics = {
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "buffer_stats": {
                    "current_size": self._buffer_stats["current_size"],
                    "peak_size": self._buffer_stats["peak_size"],
                    "total_processed": self._buffer_stats["total_processed"],
                    "chunks_processed": self._buffer_stats["chunks_processed"],
                    "write_count": self._buffer_stats["write_count"],
                    "write_rate": self._buffer_stats["write_rate"],
                    "avg_chunk_size": self._buffer_stats["avg_chunk_size"],
                    "queue_size": self._write_queue.qsize() if not self._closing else 0,
                    "write_buffer_size": (
                        self.client_transport.get_write_buffer_size()
                        if hasattr(self.client_transport, 'get_write_buffer_size')
                        else None
                    )
                },
                "paused": self._closing,
                **kwargs
            }
            
            await proxy_state.update_connection(
                self._connection_id,
                "tunnel_metrics",
                metrics
            )
            
            self._last_metrics_update = now
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to update metrics: {e}")
    
    async def _handle_connect(self, host: str, port: int) -> None:
        """Handle CONNECT by establishing direct tunnel."""
        try:
            logger.debug(f"[{self._connection_id}] Establishing direct tunnel to {host}:{port}")
            
            # Send 200 Connection Established
            response = b"HTTP/1.1 200 Connection Established\r\nConnection: keep-alive\r\n\r\n"
            if self.transport and not self.transport.is_closing():
                self.transport.write(response)
            
            # Store original client transport
            original_transport = self.transport
            
            # Create server-side protocol instance
            server_protocol = TunnelProtocol(
                client_transport=original_transport,
                flow_control=self.flow_control,
                connection_id=f"{self._connection_id}-server",
                buffer_size=self._buffer_size,
                metrics_interval=self._metrics_interval,
                write_limit=self._write_limit,
                write_interval=self._write_interval
            )
            
            # Create direct connection to target with timeout
            loop = asyncio.get_event_loop()
            try:
                async with async_timeout(10) as cm:  # 10 second timeout for connection
                    logger.debug(f"[{self._connection_id}] Connecting to {host}:{port}")
                    server_transport, _ = await loop.create_connection(
                        lambda: server_protocol,
                        host=host,
                        port=port
                    )
                    logger.debug(f"[{self._connection_id}] Connected to {host}:{port}")
            except asyncio.TimeoutError:
                raise RuntimeError(f"Connection to {host}:{port} timed out")
            
            # Update transport references and verify
            self._client_protocol = self  # Client side
            self._server_protocol = server_protocol  # Server side
            self.client_transport = original_transport
            self._server_transport = server_transport
            
            # Verify transports are valid
            if not self.client_transport or self.client_transport.is_closing():
                raise RuntimeError("Client transport is invalid or closed")
            if not self._server_transport or self._server_transport.is_closing():
                raise RuntimeError("Server transport is invalid or closed")
            
            # Link the protocols for bidirectional forwarding
            server_protocol._client_protocol = self
            server_protocol.client_transport = original_transport
            server_protocol._server_transport = server_transport  # Set server transport for proper forwarding
            server_protocol.transport = server_transport  # Set main transport
            
            # Wait briefly to ensure connection is stable
            await asyncio.sleep(0.05)  # Reduced from 0.1
            
            # Verify connection is still alive
            if self._closing or server_protocol._closing:
                raise RuntimeError("Connection closed during setup")
            
            # Configure socket with error checking
            sock = server_transport.get_extra_info('socket')
            if sock:
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 524288)  # Increased from 262144
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 524288)  # Increased from 262144
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    
                    # Verify socket settings
                    actual_rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                    actual_sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
                    logger.debug(f"[{self._connection_id}] Socket buffers: rcv={actual_rcvbuf}, snd={actual_sndbuf}")
                    
                except socket.error as e:
                    logger.warning(f"[{self._connection_id}] Socket configuration error: {e}")
            
            # Update state and log
            await self._update_metrics("tunnel_established", 
                                     remote_host=host,
                                     remote_port=port,
                                     client_id=id(self.client_transport),
                                     server_id=id(self._server_transport))
            logger.debug(f"[{self._connection_id}] Tunnel established to {host}:{port}")
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to establish tunnel: {e}")
            await self._update_metrics("error", error=str(e))
            asyncio.create_task(self._cleanup(error=str(e)))
            raise
            
    def _pause_reading(self) -> None:
        """Pause reading when buffer is full."""
        if not self._closing and self.transport:
            self._closing = True
            self.transport.pause_reading()
            self.flow_control.pause_reading()
            logger.debug(f"[{self._connection_id}] Paused reading from tunnel")
            asyncio.create_task(self._update_metrics("paused"))
    
    def _resume_reading(self) -> None:
        """Resume reading when buffer drains."""
        if self._closing and self.transport:
            self._closing = False
            self.transport.resume_reading()
            self.flow_control.resume_reading()
            logger.debug(f"[{self._connection_id}] Resumed reading from tunnel")
            asyncio.create_task(self._update_metrics("resumed"))
    
    async def _cleanup(self, error: Optional[str] = None) -> None:
        """Clean up the connection."""
        if not self._closing:
            self._closing = True
            
            # Cancel write processor
            if self._write_task and not self._write_task.done():
                self._write_task.cancel()
            
            # Close transports
            transports = [
                self.transport,
                self.client_transport,
                getattr(self, '_server_transport', None)
            ]
            
            for transport in transports:
                try:
                    if transport and not transport.is_closing():
                        if hasattr(transport, 'write_eof'):
                            try:
                                transport.write_eof()
                                await asyncio.sleep(0.05)  # Reduced from 0.1
                            except Exception:
                                pass
                        transport.close()
                except Exception as e:
                    logger.warning(f"[{self._connection_id}] Error closing transport: {e}")
            
            # Clear transport references
            self.transport = None
            self.client_transport = None
            if hasattr(self, '_server_transport'):
                self._server_transport = None
            
            # Update metrics
            if error:
                asyncio.create_task(self._update_metrics(
                    "error", 
                    error=error,
                    cleanup_time=datetime.now().isoformat()
                ))
            else:
                asyncio.create_task(self._update_metrics(
                    "closed",
                    cleanup_time=datetime.now().isoformat()
                ))
    
    async def _update_metrics(self, status: str, **kwargs) -> None:
        """Update connection metrics with rate limiting."""
        try:
            now = datetime.now().timestamp()
            
            # Check if we should rate limit updates
            if status not in ["error", "closed", "eof_received"]:  # Always send important states
                if now - self._last_metrics_update < self._metrics_interval:
                    # Queue update for later
                    self._pending_updates.update(kwargs)
                    self._pending_updates["status"] = status
                    return
            
            # Include any pending updates
            if self._pending_updates:
                kwargs.update(self._pending_updates)
                self._pending_updates.clear()
            
            metrics = {
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "buffer_stats": {
                    "current_size": self._buffer_stats["current_size"],
                    "peak_size": self._buffer_stats["peak_size"],
                    "total_processed": self._buffer_stats["total_processed"],
                    "chunks_processed": self._buffer_stats["chunks_processed"],
                    "write_count": self._buffer_stats["write_count"],
                    "write_rate": self._buffer_stats["write_rate"],
                    "avg_chunk_size": self._buffer_stats["avg_chunk_size"],
                    "queue_size": self._write_queue.qsize() if not self._closing else 0,
                    "write_buffer_size": (
                        self.client_transport.get_write_buffer_size()
                        if hasattr(self.client_transport, 'get_write_buffer_size')
                        else None
                    )
                },
                "paused": self._closing,
                **kwargs
            }
            
            await proxy_state.update_connection(
                self._connection_id,
                "tunnel_metrics",
                metrics
            )
            
            self._last_metrics_update = now
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to update metrics: {e}")
