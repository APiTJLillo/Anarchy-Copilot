"""SSL transport implementation with enhanced TLS handling."""
import asyncio
import ssl
import logging
from typing import Optional, Union, Any, Dict
from async_timeout import timeout
import collections
import time

logger = logging.getLogger("proxy.core")

class SslTransport(asyncio.Transport):
    """SSL transport implementation using memory BIOs."""
    
    def __init__(self, loop: asyncio.AbstractEventLoop, ssl_object: ssl.SSLObject, protocol: asyncio.Protocol):
        super().__init__()
        self._loop = loop
        self._ssl = ssl_object
        self._protocol = protocol
        self._closing = False
        self._handshake_complete = False
        self._write_backlog = collections.deque()
        self._write_buffer_size = 0
        self._read_buffer = bytearray()
        self._write_wants_read = False
        self._read_wants_write = False
        self._extra = {
            'ssl_object': ssl_object,
            'bytes_sent': 0,
            'bytes_received': 0,
            'last_activity': time.time()
        }
        
        # Start handshake
        self._protocol.connection_made(self)
        asyncio.create_task(self._do_handshake())
        
    async def _do_handshake(self) -> None:
        """Perform SSL handshake."""
        start_time = time.time()
        try:
            while True:
                try:
                    self._ssl.do_handshake()
                    break
                except ssl.SSLWantReadError:
                    if self._read_buffer:
                        continue
                    self._read_wants_write = True
                    await asyncio.sleep(0.001)
                except ssl.SSLWantWriteError:
                    self._write_wants_read = True
                    await asyncio.sleep(0.001)
                    
            # Handshake complete
            self._handshake_complete = True
            duration = time.time() - start_time
            logger.debug(f"SSL handshake completed in {duration:.2f}s")
            
            # Log cipher information
            version = self._ssl.version()
            cipher = self._ssl.cipher()
            if cipher:
                logger.debug(f"Using {version}, cipher: {cipher[0]}")
                
        except Exception as e:
            logger.error(f"SSL handshake failed: {e}")
            self.close()
            return
            
        # Process any pending writes
        if self._write_backlog:
            asyncio.create_task(self._process_write_backlog())
            
    def data_received(self, data: bytes) -> None:
        """Process received encrypted data."""
        if self._closing:
            return
            
        self._read_buffer.extend(data)
        self._extra['last_activity'] = time.time()
        
        if not self._handshake_complete:
            # Data received during handshake
            return
            
        asyncio.create_task(self._process_read_buffer())
        
    async def _process_read_buffer(self) -> None:
        """Process and decrypt received data."""
        while self._read_buffer and not self._closing:
            try:
                # Try to decrypt data
                data = self._ssl.recv(len(self._read_buffer))
                if data:
                    self._extra['bytes_received'] += len(data)
                    self._protocol.data_received(data)
                    
                # Clear processed data
                self._read_buffer.clear()
                
            except ssl.SSLWantReadError:
                # Need more data
                break
            except ssl.SSLWantWriteError:
                # Need to write before can read more
                self._read_wants_write = True
                await asyncio.sleep(0.001)
            except Exception as e:
                logger.error(f"Error processing read buffer: {e}")
                self.close()
                break
                
    def write(self, data: Union[bytes, bytearray, memoryview]) -> None:
        """Queue data for writing."""
        if self._closing:
            return
            
        # Add data to write backlog
        self._write_backlog.append(bytes(data))
        self._write_buffer_size += len(data)
        
        # Schedule processing of write backlog
        if self._handshake_complete:
            asyncio.create_task(self._process_write_backlog())
            
    async def _process_write_backlog(self) -> None:
        """Process queued write data."""
        while self._write_backlog and not self._closing:
            data = self._write_backlog[0]
            try:
                # Try to encrypt and write data
                sent = await self._write_encrypted_data(data)
                if sent:
                    self._write_backlog.popleft()
                    self._write_buffer_size -= len(data)
                    self._extra['bytes_sent'] += sent
                else:
                    # No progress made, small delay
                    await asyncio.sleep(0.001)
                    
            except Exception as e:
                logger.error(f"Error processing write backlog: {e}")
                self.close()
                break
                
    async def _write_encrypted_data(self, data: bytes) -> int:
        """Encrypt and write data using SSL object."""
        try:
            # Encrypt data
            return self._ssl.send(data)
        except ssl.SSLWantReadError:
            self._write_wants_read = True
            return 0
        except ssl.SSLWantWriteError:
            return 0
            
    def close(self) -> None:
        """Close the transport."""
        if not self._closing:
            self._closing = True
            try:
                if self._handshake_complete:
                    # Attempt graceful shutdown
                    while True:
                        try:
                            self._ssl.unwrap()
                            break
                        except ssl.SSLWantReadError:
                            if not self._read_buffer:
                                break
                        except ssl.SSLWantWriteError:
                            break
                        except Exception:
                            break
            except Exception:
                pass
            self._protocol.connection_lost(None)
            
    def abort(self) -> None:
        """Abort the transport."""
        self.close()
        
    def get_extra_info(self, name: str, default: Any = None) -> Any:
        """Get transport extra information."""
        return self._extra.get(name, default)
        
    def pause_reading(self) -> None:
        """Pause reading from transport."""
        pass  # Not implemented
        
    def resume_reading(self) -> None:
        """Resume reading from transport."""
        pass  # Not implemented
        
    def set_write_buffer_limits(self, high: Optional[int] = None, low: Optional[int] = None) -> None:
        """Set write buffer limits."""
        pass  # Not implemented
        
    def get_write_buffer_size(self) -> int:
        """Get current write buffer size."""
        return self._write_buffer_size
        
    def is_closing(self) -> bool:
        """Check if transport is closing."""
        return self._closing 