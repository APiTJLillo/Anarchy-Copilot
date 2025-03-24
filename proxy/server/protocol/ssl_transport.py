"""SSL transport implementation."""
import asyncio
import socket
import ssl
import errno
import logging
from typing import Optional, Dict, Any, cast

from .socket_helpers import get_raw_socket
from .tls_handler_protocol import TlsHandlerProtocol

logger = logging.getLogger("proxy.core")

class SslTransport(asyncio.Transport):
    """Enhanced SSL transport with robust socket handling."""

    @classmethod
    async def create(cls,
                    handler: TlsHandlerProtocol,
                    transport: asyncio.Transport,
                    context: ssl.SSLContext,
                    server_side: bool = False,
                    server_hostname: Optional[str] = None) -> "SslTransport":
        """Create SSL transport."""
        loop = asyncio.get_event_loop()
        protocol = cast(asyncio.Protocol, transport.get_protocol())
        
        if server_side and not server_hostname:
            raise ValueError("server_hostname required for server side")
            
        sock = transport.get_extra_info('socket')
        if not sock:
            raise RuntimeError("No socket available from transport")
            
        # Create SSL socket
        ssl_sock = context.wrap_socket(
            sock,
            server_side=server_side,
            server_hostname=server_hostname,
            do_handshake_on_connect=False
        )
        
        # Return initialized transport
        return cls(handler, loop, ssl_sock, protocol)
        
    def __init__(self, handler: TlsHandlerProtocol, loop: asyncio.AbstractEventLoop,
                 ssl_sock: ssl.SSLSocket, protocol: asyncio.Protocol):
        """Initialize SSL transport with safe socket access."""
        super().__init__()
        
        self._loop = loop
        self._ssl_sock = ssl_sock
        self._protocol = protocol
        self._handler = handler
        
        # Get raw socket safely
        raw_sock = get_raw_socket(ssl_sock)
        if not raw_sock:
            raise RuntimeError("Could not get raw socket")
        self._sock = raw_sock
            
        # Initialize state
        self._closing = False
        self._write_buffer = bytearray()
        self._read_buffer = bytearray()
        self._read_paused = False
        self._write_paused = False
        self._read_task: Optional[asyncio.Task] = None
        self._write_task: Optional[asyncio.Task] = None 
        self._processing_task: Optional[asyncio.Task] = None
        self._extra: Dict[str, Any] = {}
        
        # Set socket non-blocking
        self._sock.setblocking(False)
        
        # Configure socket
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except socket.error as e:
            logger.warning(f"Could not set socket options: {e}")
            
        # Initialize transport info
        self._init_transport_info()
        
        # Start reading
        self._start_reading()
        
        logger.debug(f"SSL Transport initialized: {self._extra}")
        
    def _init_transport_info(self) -> None:
        """Initialize transport extra info."""
        try:
            socket_info: Dict[str, Any] = {
                'socket': self._sock,
                'ssl_object': self._ssl_sock,
                'bytes_sent': 0,
                'bytes_received': 0
            }
            
            # Get peername safely
            try:
                if self._sock is not None:
                    socket_info['peername'] = self._sock.getpeername()
            except (socket.error, AttributeError):
                socket_info['peername'] = None
                
            # Get socket family/protocol safely
            for attr in ('family', 'proto'):
                try:
                    if self._sock is not None:
                        socket_info[f'socket_{attr}'] = getattr(self._sock, attr)
                except (socket.error, AttributeError):
                    socket_info[f'socket_{attr}'] = None
                    
            # Get SSL info safely
            for attr in ('cipher', 'version'):
                try:
                    if self._ssl_sock is not None:
                        socket_info[attr] = getattr(self._ssl_sock, attr)()
                except (AttributeError, ssl.SSLError):
                    socket_info[attr] = None
                    
            # Get socket state safely
            try:
                if self._sock is not None:
                    fileno = self._sock.fileno()
                    socket_info['socket_state'] = 'connected' if fileno >= 0 else 'closed'
            except (socket.error, AttributeError):
                socket_info['socket_state'] = 'unknown'
                
            self._extra = socket_info
                
        except Exception as e:
            logger.error(f"Error initializing transport info: {e}")
            # Initialize with minimal info
            self._extra = {
                'socket': self._sock,
                'ssl_object': self._ssl_sock,
                'bytes_sent': 0,
                'bytes_received': 0
            }
            
    def _start_reading(self) -> None:
        """Start reading data."""
        if not self._read_task:
            self._read_task = self._loop.create_task(self._read_loop())
            
    async def _read_loop(self) -> None:
        """Read data from socket."""
        BUFFER_SIZE = 32768  # 32KB buffer
        
        while not self._closing:
            try:
                # Check if socket is still valid
                if not self._ssl_sock or getattr(self._ssl_sock, '_closed', True):
                    logger.debug("SSL socket closed, stopping read loop")
                    break
                    
                # Wait for data
                try:
                    await self._loop.sock_recv(self._ssl_sock, 0)
                except (ConnectionError, socket.error) as e:
                    if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                        logger.error(f"Socket error while waiting: {e}")
                        break
                    continue
                    
                # Read data
                try:
                    data = await self._loop.sock_recv(self._ssl_sock, BUFFER_SIZE)
                    if not data:
                        logger.debug("Connection closed by peer")
                        break
                        
                    # Process data
                    if self._processing_task and not self._processing_task.done():
                        await self._processing_task
                    self._processing_task = self._loop.create_task(
                        self._handler.process_decrypted_data(bytes(data), self._protocol)
                    )
                    
                    # Update stats    
                    self._extra['bytes_received'] += len(data)
                        
                except (ConnectionError, socket.error) as e:
                    if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                        logger.error(f"Socket read error: {e}")
                        break
                    continue
                    
            except Exception as e:
                logger.error(f"Error in read loop: {e}")
                break
                
        # Clean up
        self._read_task = None
        if not self._closing:
            self.close()
            
    def write(self, data: bytes) -> None:
        """Write data to transport."""
        if self._closing or not data:
            return
            
        if not self._write_task:
            self._write_task = self._loop.create_task(self._write_loop())
            
        self._write_buffer.extend(data)
        
    async def _write_loop(self) -> None:
        """Write buffered data."""
        try:
            while self._write_buffer and not self._closing:
                # Write chunk
                chunk = bytes(self._write_buffer[:32768])  # 32KB chunks
                self._write_buffer = self._write_buffer[len(chunk):]
                
                try:
                    await self._loop.sock_sendall(self._ssl_sock, chunk)
                    self._extra['bytes_sent'] += len(chunk)
                        
                except (ConnectionError, socket.error) as e:
                    if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                        logger.error(f"Socket write error: {e}")
                        break
                    await asyncio.sleep(0.001)  # Small delay before retry
                    continue
                    
        except Exception as e:
            logger.error(f"Error in write loop: {e}")
            
        finally:
            self._write_task = None
            if self._write_buffer and not self._closing:
                self._write_task = self._loop.create_task(self._write_loop())
                
    def close(self) -> None:
        """Close transport."""
        if self._closing:
            return
            
        self._closing = True
        
        # Close SSL socket
        try:
            if self._ssl_sock:
                try:
                    self._ssl_sock.unwrap()
                except ssl.SSLError:
                    pass
                self._ssl_sock.close()
        except Exception as e:
            logger.error(f"Error closing SSL socket: {e}")
            
        # Close raw socket    
        try:
            if self._sock:
                self._sock.close()
        except Exception as e:
            logger.error(f"Error closing raw socket: {e}")
            
        self._protocol.connection_lost(None)
        
    def abort(self) -> None:
        """Abort transport."""
        self.close()
        
    def get_write_buffer_size(self) -> int:
        """Get current write buffer size."""
        return len(self._write_buffer)
        
    def set_write_buffer_limits(self, high: Optional[int] = None) -> None:
        """Set write buffer limits."""
        if high is None:
            high = 64 * 1024
        # Not implementing for now
        pass
        
    def get_extra_info(self, name: str, default: Any = None) -> Any:
        """Get transport extra info."""
        return self._extra.get(name, default)
        
    def is_closing(self) -> bool:
        """Check if transport is closing."""
        return self._closing
        
    def pause_reading(self) -> None:
        """Pause reading."""
        self._read_paused = True
        
    def resume_reading(self) -> None:
        """Resume reading."""
        self._read_paused = False
        
    def get_protocol(self) -> asyncio.Protocol:
        """Get protocol."""
        return self._protocol
        
    def set_protocol(self, protocol: asyncio.Protocol) -> None:
        """Set protocol."""
        self._protocol = protocol
