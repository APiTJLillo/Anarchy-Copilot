"""Buffer management for connection data."""
import asyncio
import logging
from typing import Optional, Any
from async_timeout import timeout

logger = logging.getLogger("proxy.core")

class BufferManager:
    """Manages data buffering and flow control."""
    
    def __init__(self, connection_id: str, transport: Any):
        self._connection_id = connection_id
        self._transport = transport
        self._buffer = bytearray()
        self._max_buffer_size = 4 * 1024 * 1024  # Increased to 4MB max buffer
        self._handshake_buffer_size = 64 * 1024  # Increased to 64KB for handshake
        self._write_lock = asyncio.Lock()
        self._last_write_time = 0
        self._write_stats = {
            'total_bytes': 0,
            'chunks_written': 0,
            'write_errors': 0,
            'last_error': None
        }

    def clear_buffers(self) -> None:
        """Clear all buffers."""
        self._buffer.clear()

    def get_buffer(self) -> bytes:
        """Get current buffer contents.
        
        Returns:
            Bytes object containing buffered data
        """
        return bytes(self._buffer)

    def buffer_handshake_data(self, data: bytes) -> bool:
        """Buffer handshake data with size limits.
        
        Args:
            data: Data to buffer
            
        Returns:
            bool: True if buffering succeeded, False if buffer would overflow
        """
        try:
            if len(self._buffer) + len(data) > self._handshake_buffer_size:
                logger.error(f"[{self._connection_id}] Handshake buffer overflow - current: {len(self._buffer)}, new: {len(data)}")
                return False
                
            self._buffer.extend(data)
            logger.debug(f"[{self._connection_id}] Buffered {len(data)} bytes for handshake (total: {len(self._buffer)})")
            return True
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error buffering handshake data: {e}")
            return False

    async def write_chunked(self, data: bytes, target: Any, chunk_size: int = 32768,
                          tls_version: Optional[str] = None) -> None:
        """Write data in chunks with optimized flow control.
        
        Args:
            data: Data to write
            target: Transport to write to
            chunk_size: Size of chunks to write (increased default)
            tls_version: TLS version being used (affects timing)
        """
        try:
            if not target or target.is_closing():
                logger.error(f"[{self._connection_id}] Cannot write - target transport closed/invalid")
                return

            logger.debug(f"[{self._connection_id}] Starting chunked write of {len(data)} bytes")
            
            # Set timeout based on data size with more reasonable minimum
            timeout_seconds = max(60, len(data) / 16384)  # Increased base timeout
            
            async with timeout(timeout_seconds):
                async with self._write_lock:
                    offset = 0
                    total = len(data)
                    start_time = asyncio.get_event_loop().time()

                    # Adjust chunk size based on TLS version
                    if tls_version:
                        if "1.0" in tls_version:
                            chunk_size = min(chunk_size, 16384)  # Max 16KB for TLS 1.0
                        elif "1.1" in tls_version:
                            chunk_size = min(chunk_size, 24576)  # Max 24KB for TLS 1.1
                        else:
                            chunk_size = min(chunk_size, 32768)  # Max 32KB for modern TLS

                    # Dynamic chunk delay based on TLS version and transfer speed
                    base_delay = 0.0001  # Reduced base delay to 0.1ms
                    if tls_version and "1.0" in tls_version:
                        base_delay = 0.001  # 1ms for TLS 1.0

                    logger.debug(f"[{self._connection_id}] Using chunk_size={chunk_size}, base_delay={base_delay}s for {tls_version or 'unknown TLS'}")

                    chunk_delay = base_delay
                    consecutive_errors = 0
                    backoff_delay = 0.001  # Start with 1ms backoff

                    while offset < total:
                        end = min(offset + chunk_size, total)
                        chunk = data[offset:end]
                        
                        try:
                            if target.is_closing():
                                raise RuntimeError("Transport closed during write")
                                
                            # Add write buffer size check
                            if hasattr(target, 'get_write_buffer_size'):
                                while target.get_write_buffer_size() > chunk_size * 2:
                                    await asyncio.sleep(chunk_delay)
                            
                            target.write(chunk)
                            self._write_stats['total_bytes'] += len(chunk)
                            self._write_stats['chunks_written'] += 1
                            offset = end
                            
                            # Reset error counter on successful write
                            if consecutive_errors > 0:
                                consecutive_errors = 0
                                backoff_delay = 0.001  # Reset backoff
                            
                            # Progress logging for large transfers
                            if total > 131072 and offset % 131072 == 0:  # Increased threshold
                                elapsed = asyncio.get_event_loop().time() - start_time
                                progress = (offset / total) * 100
                                speed = offset / elapsed if elapsed > 0 else 0
                                logger.debug(
                                    f"[{self._connection_id}] Write progress: {progress:.1f}% "
                                    f"({offset}/{total} bytes), {speed:.1f} bytes/sec"
                                )
                            
                            # Adaptive flow control
                            if offset < total:
                                elapsed = asyncio.get_event_loop().time() - start_time
                                if elapsed > 0:
                                    current_speed = offset / elapsed
                                    if current_speed > chunk_size:
                                        # Reduce delay if we're making good progress
                                        chunk_delay = max(0.00001, chunk_delay * 0.75)  # More aggressive reduction
                                    elif current_speed < chunk_size / 2:
                                        # Increase delay if we're going too slow
                                        chunk_delay = min(0.005, chunk_delay * 1.25)
                                await asyncio.sleep(chunk_delay)
                            
                        except Exception as e:
                            logger.error(f"[{self._connection_id}] Write error at offset {offset}/{total}: {e}")
                            self._write_stats['write_errors'] += 1
                            self._write_stats['last_error'] = str(e)
                            
                            consecutive_errors += 1
                            if consecutive_errors > 5:
                                raise RuntimeError(f"Too many consecutive write errors: {consecutive_errors}")
                            
                            # Exponential backoff
                            await asyncio.sleep(backoff_delay)
                            backoff_delay = min(0.1, backoff_delay * 2)  # Max 100ms backoff
                            continue

                    # Log completion with stats
                    elapsed = asyncio.get_event_loop().time() - start_time
                    speed = total / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"[{self._connection_id}] Completed write of {total} bytes "
                        f"in {elapsed:.2f}s ({speed:.1f} bytes/sec), "
                        f"chunks: {self._write_stats['chunks_written']}, "
                        f"errors: {self._write_stats['write_errors']}"
                    )

        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to write data: {e}")
            raise
