"""Protocol implementation for tunneling connections."""
import asyncio
import logging
import socket
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List, Protocol, runtime_checkable
from async_timeout import timeout as async_timeout

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
                 connection_id: str, buffer_size: int = 65536, 
                 metrics_interval: float = 0.1,
                 write_limit: int = 262144,  # 256KB write limit
                 write_interval: float = 0.001):  # 1ms minimum between writes
        self.client_transport = client_transport
        self.transport = None
        self.flow_control = flow_control
        self.connection_id = connection_id
        self._buffer_size = buffer_size
        self._paused = False
        self._closed = False
        
        # Transfer settings
        self._write_limit = write_limit
        self._write_interval = write_interval
        self._last_write = 0.0
        
        # Statistics and state tracking
        self._write_queue = asyncio.Queue()
        self._write_task: Optional[asyncio.Task] = None
        self._buffer_stats = {
            "current_size": 0,
            "peak_size": 0,
            "total_processed": 0,
            "chunks_processed": 0,
            "last_metric_update": 0.0,
            "write_rate": 0.0,
            "avg_chunk_size": 0.0,
            "write_count": 0
        }
        self._metrics_interval = metrics_interval
        self._last_metrics_update = 0.0
        self._pending_updates: Dict[str, Any] = {}
        
        # Rate limiting state
        self._write_permits = asyncio.Semaphore(1)
        
        # Register state
        asyncio.create_task(self._update_metrics("initialized"))
    
    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle new connection."""
        self.transport = transport
        if hasattr(transport, 'get_write_buffer_size'):
            transport.set_write_buffer_limits(high=self._buffer_size)
        
        # Start write queue processor
        self._write_task = asyncio.create_task(self._process_write_queue())
        asyncio.create_task(self._update_metrics("connected"))
    
    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection loss."""
        self._closed = True
        if exc:
            logger.error(f"[{self.connection_id}] Tunnel connection lost: {exc}")
            asyncio.create_task(self._update_metrics("error", error=str(exc)))
        else:
            asyncio.create_task(self._update_metrics("closed"))
        
        # Cancel write processor
        if self._write_task and not self._write_task.done():
            self._write_task.cancel()
        
        # Close client transport if still open
        if self.client_transport and not self.client_transport.is_closing():
            self.client_transport.close()
    
    def data_received(self, data: bytes) -> None:
        """Handle received data with flow control and metrics."""
        if self._closed:
            return
        
        try:
            data_len = len(data)
            asyncio.create_task(self._update_metrics("data_received", bytes_count=data_len))
            
            if self.on_data_received:
                self.on_data_received(data_len)
            
            # Get target transport and check if we need TLS record size limits
            target = self._get_target_transport(self.transport)
            if target and not target.is_closing():
                # Check if this is a TLS connection
                ssl_object = target.get_extra_info('ssl_object')
                is_tls = bool(ssl_object)
                
                if is_tls:
                    # Split data into TLS record-sized chunks (max 16KB)
                    MAX_CHUNK = 16384  # 16KB TLS record limit
                    logger.debug(f"[{self.connection_id}] Using TLS chunking for {data_len} bytes")
                    
                    for i in range(0, data_len, MAX_CHUNK):
                        chunk = data[i:i + MAX_CHUNK]
                        self._write_queue.put_nowait(chunk)
                        self._buffer_stats["current_size"] += len(chunk)
                        self._buffer_stats["peak_size"] = max(
                            self._buffer_stats["peak_size"], 
                            self._buffer_stats["current_size"]
                        )
                else:
                    # For non-TLS connections, use larger chunks
                    self._write_queue.put_nowait(data)
                    self._buffer_stats["current_size"] += data_len
                    self._buffer_stats["peak_size"] = max(
                        self._buffer_stats["peak_size"], 
                        self._buffer_stats["current_size"]
                    )
                
                # Check if we need to pause reading
                if self._buffer_stats["current_size"] > self._buffer_size * 2:
                    self._pause_reading()
                
                # Log transfer details for debugging
                logger.debug(
                    f"[{self.connection_id}] Queued {data_len} bytes "
                    f"(current buffer: {self._buffer_stats['current_size']})"
                )
            else:
                logger.warning(f"[{self.connection_id}] No target transport for forwarding")
                asyncio.create_task(self._cleanup())
        
        except Exception as e:
            logger.error(f"[{self.connection_id}] Error handling tunnel data: {e}")
            asyncio.create_task(self._cleanup(error=str(e)))
    
    async def _process_write_queue(self) -> None:
        """Process queued writes with flow control."""
        try:
            while not self._closed:
                try:
                    # Process chunks in batches for efficiency
                    chunks: List[bytes] = []
                    total_size = 0
                    
                    # Try to get multiple chunks if available
                    while not self._closed and total_size < self._write_limit:
                        try:
                            async with async_timeout(0.1) as cm:  # Short timeout for batching
                                data = await self._write_queue.get()
                                chunks.append(data)
                                total_size += len(data)
                                if total_size >= self._write_limit:
                                    break
                        except asyncio.TimeoutError:
                            break
                    
                    if chunks:
                        await self._write_chunks(chunks, total_size)
                
                except asyncio.TimeoutError:
                    continue
                
        except asyncio.CancelledError:
            logger.debug(f"[{self.connection_id}] Write queue processor cancelled")
        except Exception as e:
            logger.error(f"[{self.connection_id}] Write queue processor error: {e}")
            self._cleanup(error=str(e))

    async def _write_chunks(self, chunks: List[bytes], total_size: int) -> None:
        """Write chunks to transport with rate limiting."""
        target = self._get_target_transport(self.transport)
        if not target or target.is_closing():
            return

        # Apply rate limiting
        now = datetime.now().timestamp()
        time_since_last_write = now - self._last_write
        if time_since_last_write < self._write_interval:
            await asyncio.sleep(self._write_interval - time_since_last_write)

        async with self._write_permits:
            try:
                # Check if target needs TLS record size limits
                ssl_object = target.get_extra_info('ssl_object')
                is_tls = bool(ssl_object)
                chunks_written = 0

                if is_tls:
                    # Write chunks with TLS record size limits
                    MAX_CHUNK = 16384  # 16KB TLS record limit
                    logger.debug(f"[{self.connection_id}] Writing {len(chunks)} chunks with TLS chunking")
                    
                    for chunk in chunks:
                        # Split chunk if needed
                        for i in range(0, len(chunk), MAX_CHUNK):
                            sub_chunk = chunk[i:i + MAX_CHUNK]
                            target.write(sub_chunk)
                            chunks_written += 1
                            await asyncio.sleep(0)  # Allow other tasks to run
                else:
                    # For non-TLS connections, write chunks directly
                    logger.debug(f"[{self.connection_id}] Writing {len(chunks)} chunks directly")
                    for chunk in chunks:
                        target.write(chunk)
                        chunks_written += 1
                        await asyncio.sleep(0)  # Allow other tasks to run
                
                # Mark chunks as processed
                for _ in chunks:
                    self._write_queue.task_done()
                
                # Update write statistics
                self._buffer_stats["write_count"] += len(chunks)
                total_chunks = self._buffer_stats["chunks_processed"] + chunks_written
                self._buffer_stats["avg_chunk_size"] = (
                    (self._buffer_stats["avg_chunk_size"] * self._buffer_stats["chunks_processed"] + total_size) 
                    / total_chunks
                )
                self._buffer_stats["chunks_processed"] = total_chunks
                
                # Update buffer stats
                self._buffer_stats["current_size"] -= total_size
                self._buffer_stats["total_processed"] += total_size
                
                # Update write rate
                self._last_write = datetime.now().timestamp()
                elapsed = self._last_write - self._last_metrics_update if self._last_metrics_update else 0.001
                self._buffer_stats["write_rate"] = total_size / elapsed if elapsed > 0 else 0
                
                # Notify of data sent
                if self.on_data_sent is not None:
                    self.on_data_sent(total_size)
                
                # Check if we can resume reading
                if self._paused and self._buffer_stats["current_size"] <= self._buffer_size:
                    self._resume_reading()
                
            except Exception as e:
                logger.error(f"[{self.connection_id}] Error writing chunks: {e}")
                raise
    
    def _pause_reading(self) -> None:
        """Pause reading when buffer is full."""
        if not self._paused and self.transport:
            self._paused = True
            self.transport.pause_reading()
            self.flow_control.pause_reading()
            logger.debug(f"[{self.connection_id}] Paused reading from tunnel")
            asyncio.create_task(self._update_metrics("paused"))
    
    def _resume_reading(self) -> None:
        """Resume reading when buffer drains."""
        if self._paused and self.transport:
            self._paused = False
            self.transport.resume_reading()
            self.flow_control.resume_reading()
            logger.debug(f"[{self.connection_id}] Resumed reading from tunnel")
            asyncio.create_task(self._update_metrics("resumed"))
    
    async def _cleanup(self, error: Optional[str] = None) -> None:
        """Clean up the connection."""
        if not self._closed:
            self._closed = True
            
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
                                await asyncio.sleep(0.1)  # Brief pause for EOF to be sent
                            except Exception:
                                pass
                        transport.close()
                except Exception as e:
                    logger.warning(f"[{self.connection_id}] Error closing transport: {e}")
            
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
                    "queue_size": self._write_queue.qsize() if not self._closed else 0,
                    "write_buffer_size": (
                        self.client_transport.get_write_buffer_size()
                        if hasattr(self.client_transport, 'get_write_buffer_size')
                        else None
                    )
                },
                "paused": self._paused,
                **kwargs
            }
            
            await proxy_state.update_connection(
                self.connection_id,
                "tunnel_metrics",
                metrics
            )
            
            self._last_metrics_update = now
            
        except Exception as e:
            logger.error(f"[{self.connection_id}] Failed to update metrics: {e}")
    
    async def _handle_connect(self, host: str, port: int) -> None:
        """Handle CONNECT by establishing direct tunnel."""
        try:
            logger.debug(f"[{self.connection_id}] Establishing direct tunnel to {host}:{port}")
            
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
                connection_id=f"{self.connection_id}-server",
                buffer_size=self._buffer_size,
                metrics_interval=self._metrics_interval,
                write_limit=self._write_limit,
                write_interval=self._write_interval
            )
            
            # Create direct connection to target with timeout
            loop = asyncio.get_event_loop()
            try:
                async with async_timeout(10) as cm:  # 10 second timeout for connection
                    logger.debug(f"[{self.connection_id}] Connecting to {host}:{port}")
                    server_transport, _ = await loop.create_connection(
                        lambda: server_protocol,
                        host=host,
                        port=port
                    )
                    logger.debug(f"[{self.connection_id}] Connected to {host}:{port}")
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
            server_protocol.client_transport = self.transport
            server_protocol._server_transport = None  # Server protocol doesn't need server transport
            
            # Wait briefly to ensure connection is stable
            await asyncio.sleep(0.1)
            
            # Verify connection is still alive
            if self._closed or server_protocol._closed:
                raise RuntimeError("Connection closed during setup")
            
            # Configure socket with error checking
            sock = server_transport.get_extra_info('socket')
            if sock:
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)  # 256KB
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)  # 256KB
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    
                    # Verify socket settings
                    actual_rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                    actual_sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
                    logger.debug(f"[{self.connection_id}] Socket buffers: rcv={actual_rcvbuf}, snd={actual_sndbuf}")
                    
                except socket.error as e:
                    logger.warning(f"[{self.connection_id}] Socket configuration error: {e}")
            
            # Update state and log
            await self._update_metrics("tunnel_established", 
                                     remote_host=host,
                                     remote_port=port,
                                     client_id=id(self.client_transport),
                                     server_id=id(self._server_transport))
            logger.debug(f"[{self.connection_id}] Tunnel established to {host}:{port}")
            
        except Exception as e:
            logger.error(f"[{self.connection_id}] Failed to establish tunnel: {e}")
            await self._update_metrics("error", error=str(e))
            asyncio.create_task(self._cleanup(error=str(e)))
            raise
            
    def _get_target_transport(self, source_transport: asyncio.Transport) -> Optional[asyncio.Transport]:
        """Get the target transport for forwarding data."""
        if self._closed:
            return None
            
        # Check if we're the client or server protocol
        if hasattr(self, '_server_protocol'):
            # We're the client protocol
            if source_transport == self.transport:
                return self._server_transport
            return self.transport
        else:
            # We're the server protocol
            return self.client_transport

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
            logger.error(f"[{self.connection_id}] Error handling EOF: {e}")
            asyncio.create_task(self._cleanup(error=str(e)))
            return False
