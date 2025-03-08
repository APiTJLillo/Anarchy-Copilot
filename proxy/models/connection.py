import asyncio
import socket
import ssl
import struct
import errno
import enum
import os
import psutil
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import logger
from ..utils.constants import NetworkConfig, SSLConfig
from .server_state import ServerState
from .ssl_context import SSLContextManager

class ConnectionState(Enum):
    """Connection lifecycle states."""
    INITIAL = "initial"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    HANDSHAKING = "handshaking"
    ESTABLISHED = "established"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"

@dataclass
class MemoryStats:
    """Memory usage statistics."""
    rss: int = 0  # Resident Set Size
    vms: int = 0  # Virtual Memory Size 
    shared: int = 0  # Shared Memory
    timestamp: float = 0.0

@dataclass
class ConnectionStats:
    """Statistics for a connection."""
    bytes_sent: int = 0
    bytes_received: int = 0
    start_time: float = 0.0
    last_activity: float = 0.0
    errors: int = 0
    handshake_completed: bool = False
    memory_samples: List[MemoryStats] = None
    
    def __post_init__(self):
        self.memory_samples = []
        
    def add_memory_sample(self, timestamp: float):
        """Add a memory usage sample."""
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        self.memory_samples.append(MemoryStats(
            rss=mem.rss,
            vms=mem.vms,
            shared=getattr(mem, 'shared', 0),
            timestamp=timestamp
        ))
        
        # Keep only last hour of samples
        cutoff = timestamp - 3600
        self.memory_samples = [s for s in self.memory_samples if s.timestamp > cutoff]

    def get_memory_deltas(self) -> Dict[str, int]:
        """Calculate memory usage deltas."""
        if len(self.memory_samples) < 2:
            return {"rss": 0, "vms": 0, "shared": 0}
            
        first = self.memory_samples[0]
        last = self.memory_samples[-1]
        return {
            "rss": last.rss - first.rss,
            "vms": last.vms - first.vms,
            "shared": last.shared - first.shared
        }


class ProxyConnection:
    """Handles a single proxy connection between client and target server."""
    
    def __init__(self, 
                 reader: asyncio.StreamReader,
                 writer: asyncio.StreamWriter,
                 ssl_manager: SSLContextManager,
                 server_state: ServerState):
        self.reader = reader
        self.writer = writer
        self.ssl_manager = ssl_manager
        self.state = server_state
        self.client_addr = writer.get_extra_info('peername')
        self.hostname: Optional[str] = None
        self._state = ConnectionState.INITIAL
        self._closing = False
        self._cleanup_lock = asyncio.Lock()
        self._server_transport = None
        self._client_transport = None
        self._client_reader: Optional[asyncio.StreamReader] = None
        self._client_writer: Optional[asyncio.StreamWriter] = None
        self._server_reader: Optional[asyncio.StreamReader] = None
        self._server_writer: Optional[asyncio.StreamWriter] = None
        self._stats = ConnectionStats()
        self._last_error: Optional[Exception] = None
        self._stats.start_time = asyncio.get_event_loop().time()
        self._stats.last_activity = self._stats.start_time
        self._tls_handshake_done = asyncio.Event()
        self._connect_response_sent = asyncio.Event()
        self._client_hello_received = asyncio.Event()

    def _set_state(self, new_state: ConnectionState) -> None:
        """Update connection state with logging."""
        old_state = self._state
        self._state = new_state
        logger.debug(f"Connection {self.client_addr} state change: {old_state.value} -> {new_state.value}")

    async def cleanup(self) -> None:
        """Clean up connection resources."""
        if self._closing:
            return

        async with self._cleanup_lock:
            try:
                cleanup_start = asyncio.get_event_loop().time()
                # Take pre-cleanup memory snapshot
                pre_cleanup_time = cleanup_start
                self._stats.add_memory_sample(pre_cleanup_time)
                pre_cleanup_stats = self._stats.memory_samples[-1]

                self._closing = True
                self._set_state(ConnectionState.CLOSING)
                
                if self.hostname:
                    try:
                        self.ssl_manager.remove_context(self.hostname)
                    except Exception as e:
                        logger.debug(f"Error removing SSL context: {e}")

                # First handle TLS shutdown
                ssl_shutdown_errors = []
                for transport in [self._server_transport, self._client_transport]:
                    if transport:
                        try:
                            ssl_obj = transport.get_extra_info('ssl_object')
                            if ssl_obj:
                                try:
                                    ssl_obj.unwrap()
                                except ssl.SSLError as e:
                                    if "PROTOCOL_IS_SHUTDOWN" not in str(e):
                                        ssl_shutdown_errors.append(str(e))
                                        logger.debug(f"SSL shutdown error: {e}")

                            sock = transport.get_extra_info('socket')
                            if sock:
                                try:
                                    sock.shutdown(socket.SHUT_RDWR)
                                except (OSError, socket.error) as e:
                                    if e.errno != errno.ENOTCONN:  # Ignore "not connected"
                                        logger.debug(f"Socket shutdown error: {e}")
                                try:
                                    sock.close()
                                except Exception as e:
                                    logger.debug(f"Socket close error: {e}")

                        except Exception as e:
                            logger.debug(f"Error cleaning up transport: {e}")

                # Take post-TLS shutdown memory snapshot
                post_ssl_time = asyncio.get_event_loop().time()
                self._stats.add_memory_sample(post_ssl_time)
                post_ssl_stats = self._stats.memory_samples[-1]
                
                # Log SSL cleanup memory impact
                ssl_cleanup_duration = post_ssl_time - cleanup_start
                ssl_memory_delta = {
                    "rss": post_ssl_stats.rss - pre_cleanup_stats.rss,
                    "vms": post_ssl_stats.vms - pre_cleanup_stats.vms,
                    "shared": post_ssl_stats.shared - pre_cleanup_stats.shared
                }
                
                logger.debug(
                    f"SSL cleanup completed in {ssl_cleanup_duration:.2f}s:\n"
                    f"  Memory impact:\n"
                    f"    RSS delta: {ssl_memory_delta['rss']/1024/1024:.1f}MB\n"
                    f"    VMS delta: {ssl_memory_delta['vms']/1024/1024:.1f}MB\n"
                    f"    Errors: {len(ssl_shutdown_errors)}"
                )

                # Then close writers in reverse order
                writer_errors = []
                for writer in [self._server_writer, self._client_writer, self.writer]:
                    if writer:
                        try:
                            if not writer.is_closing():
                                writer.close()
                                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
                        except Exception as e:
                            writer_errors.append(str(e))
                            logger.debug(f"Error closing writer: {e}")
                
                # Take final cleanup memory snapshot
                cleanup_end = asyncio.get_event_loop().time()
                self._stats.add_memory_sample(cleanup_end)
                final_stats = self._stats.memory_samples[-1]
                
                # Calculate total cleanup impact
                total_cleanup_duration = cleanup_end - cleanup_start
                total_memory_delta = {
                    "rss": final_stats.rss - pre_cleanup_stats.rss,
                    "vms": final_stats.vms - pre_cleanup_stats.vms,
                    "shared": final_stats.shared - pre_cleanup_stats.shared
                }
                
                # Get connection lifetime memory profile
                lifetime_memory_samples = len(self._stats.memory_samples)
                lifetime_deltas = self._stats.get_memory_deltas()
                peak_rss = max(s.rss for s in self._stats.memory_samples)
                peak_vms = max(s.vms for s in self._stats.memory_samples)
                
                self._set_state(ConnectionState.CLOSED)
                
                # Log comprehensive cleanup and lifetime stats
                logger.info(
                    f"Connection {self.client_addr} cleaned up:\n"
                    f"Cleanup Statistics:\n"
                    f"  Duration: {total_cleanup_duration:.2f}s\n"
                    f"  Memory reclaimed:\n"
                    f"    RSS: {abs(total_memory_delta['rss'])/1024/1024:.1f}MB\n"
                    f"    VMS: {abs(total_memory_delta['vms'])/1024/1024:.1f}MB\n"
                    f"  Errors:\n"
                    f"    SSL shutdown: {len(ssl_shutdown_errors)}\n"
                    f"    Writer cleanup: {len(writer_errors)}\n"
                    f"\nLifetime Statistics:\n"
                    f"  Memory samples: {lifetime_memory_samples}\n"
                    f"  Peak RSS: {peak_rss/1024/1024:.1f}MB\n"
                    f"  Peak VMS: {peak_vms/1024/1024:.1f}MB\n"
                    f"  Net memory change:\n"
                    f"    RSS: {lifetime_deltas['rss']/1024/1024:.1f}MB\n"
                    f"    VMS: {lifetime_deltas['vms']/1024/1024:.1f}MB"
                )
                
                # Alert on potential memory leaks
                if total_memory_delta['rss'] > 0 or total_memory_delta['vms'] > 0:
                    logger.warning(
                        f"Possible memory leak detected for {self.client_addr}:\n"
                        f"  Memory not fully reclaimed after cleanup:\n"
                        f"    RSS retained: {total_memory_delta['rss']/1024/1024:.1f}MB\n"
                        f"    VMS retained: {total_memory_delta['vms']/1024/1024:.1f}MB"
                    )

            except Exception as e:
                logger.error(f"Error during connection cleanup: {str(e)}")
                self._set_state(ConnectionState.ERROR)
                raise

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_val:
            logger.error(f"Connection error: {exc_type.__name__}: {str(exc_val)}")
            self._set_state(ConnectionState.ERROR)
            self._last_error = exc_val
        await self.cleanup()

    async def _wait_for_client_hello(self, reader: asyncio.StreamReader, timeout: float = None) -> bool:
        """Wait for and validate TLS Client Hello message."""
        try:
            if timeout is None:
                timeout = NetworkConfig.CLIENT_HELLO_TIMEOUT

            start_time = asyncio.get_event_loop().time()
            logger.debug(f"Waiting {NetworkConfig.RESPONSE_WAIT}s for client readiness")

            # Wait after sending 200 to ensure client is ready
            await asyncio.sleep(NetworkConfig.RESPONSE_WAIT)
            
            attempts = 0
            max_attempts = SSLConfig.MAX_HANDSHAKE_ATTEMPTS
            
            while attempts < max_attempts:
                try:
                    if attempts > 0:
                        delay = min(
                            SSLConfig.HANDSHAKE_RETRY_DELAY * (2 ** attempts),
                            SSLConfig.MAX_HANDSHAKE_DELAY
                        )
                        logger.debug(f"Retrying ClientHello detection (attempt {attempts + 1}/{max_attempts})")
                        await asyncio.sleep(delay)

                    # Read TLS record header (5 bytes)
                    header = await asyncio.wait_for(reader.readexactly(5), timeout=timeout)
                    if not header or len(header) != 5:
                        attempts += 1
                        continue

                    # Parse TLS record header
                    content_type, version_major, version_minor, length = struct.unpack('!BBBH', header)
                    
                    # Verify it's a handshake record (content_type = 22)
                    if content_type != 22:
                        attempts += 1
                        continue
                        
                    # Read the handshake message
                    handshake = await asyncio.wait_for(reader.readexactly(length), timeout=timeout)
                    if not handshake or handshake[0] != 1:  # Type 1 is ClientHello
                        attempts += 1
                        continue
                        
                    duration = asyncio.get_event_loop().time() - start_time
                    logger.info(f"Received valid TLS ClientHello after {duration:.2f}s and {attempts + 1} attempts")
                    self._client_hello_received.set()
                    return True

                except (asyncio.TimeoutError, asyncio.IncompleteReadError) as e:
                    logger.debug(f"ClientHello detection attempt {attempts + 1} failed: {e}")
                    attempts += 1

            logger.error(f"Failed to detect ClientHello after {max_attempts} attempts")
            return False

        except (struct.error, Exception) as e:
            logger.error(f"Error during ClientHello detection: {type(e).__name__}: {str(e)}")
            return False

    async def handle_connect(self, hostname: str, port: int) -> None:
        """Handle CONNECT request to target server."""
        self.hostname = hostname
        self._set_state(ConnectionState.CONNECTING)

        try:
            # First establish connection to target server
            try:
                self._server_reader, self._server_writer = await asyncio.wait_for(
                    asyncio.open_connection(
                        hostname, 
                        port,
                        ssl=self.ssl_manager.create_server_context(hostname)
                    ),
                    timeout=NetworkConfig.CONNECT_TIMEOUT
                )
                self._server_transport = self._server_writer.transport
                logger.debug(f"Established secure connection to target server {hostname}:{port}")
                self._set_state(ConnectionState.CONNECTED)
            except Exception as e:
                logger.error(f"Failed to connect to target server {hostname}:{port}: {e}")
                self._set_state(ConnectionState.ERROR)
                raise

            # Send 200 Connection Established to client
            try:
                self.writer.write(b'HTTP/1.1 200 Connection Established\r\n\r\n')
                await self.writer.drain()
                self._connect_response_sent.set()
                logger.debug("Sent 200 Connection Established to client")
            except ConnectionError as e:
                logger.error(f"Failed to send 200 response: {e}")
                raise
            
            # Wait for client's TLS ClientHello
            self._set_state(ConnectionState.HANDSHAKING)
            if not await self._wait_for_client_hello(self.reader):
                raise ssl.SSLError("No valid ClientHello received")

            # Now upgrade client connection to TLS
            try:
                await self._setup_client_tls(hostname)
                logger.debug("Successfully upgraded client connection to TLS")
                self._set_state(ConnectionState.ESTABLISHED)
                self._stats.handshake_completed = True
                self._tls_handshake_done.set()
            except Exception as e:
                logger.error(f"Failed to setup client TLS: {e}")
                self._set_state(ConnectionState.ERROR)
                raise

            try:
                # Begin proxying data
                await self._proxy_data(
                    self._client_reader, self._client_writer,
                    self._server_reader, self._server_writer
                )
            except Exception as e:
                logger.error(f"Error during data transfer: {e}")
                self._set_state(ConnectionState.ERROR)
                raise
            finally:
                await self.cleanup()

        except Exception as e:
            logger.error(f"Error handling CONNECT: {str(e)}")
            self._last_error = e
            if not self._closing:
                await self.cleanup()
            raise

    async def _setup_client_tls(self, hostname: str) -> None:
        """Set up TLS for the client connection."""
        loop = asyncio.get_event_loop()
        start_time = loop.time()
        client_reader = asyncio.StreamReader(limit=NetworkConfig.BUFFER_SIZE)
        protocol = asyncio.StreamReaderProtocol(client_reader)

        # Create SSL context
        client_context = self.ssl_manager.create_client_context(hostname)
        
        # Perform TLS handshake with retries
        client_transport = None
        last_error = None

        for attempt in range(SSLConfig.MAX_HANDSHAKE_ATTEMPTS):
            try:
                if attempt > 0:
                    delay = min(
                        SSLConfig.HANDSHAKE_RETRY_DELAY * (2 ** attempt),
                        SSLConfig.MAX_HANDSHAKE_DELAY
                    )
                    logger.debug(f"Retrying client TLS handshake (attempt {attempt + 1}/{SSLConfig.MAX_HANDSHAKE_ATTEMPTS}, delay: {delay:.2f}s)")
                    await asyncio.sleep(delay)

                handshake_start = loop.time()
                try:
                    client_transport = await asyncio.wait_for(
                        loop.start_tls(
                            transport=self.writer.transport,
                            protocol=protocol,
                            sslcontext=client_context,
                            server_side=True,
                            ssl_handshake_timeout=NetworkConfig.SSL_HANDSHAKE_TIMEOUT
                        ),
                        timeout=NetworkConfig.SSL_HANDSHAKE_TIMEOUT + 5.0
                    )
                except ssl.SSLError as e:
                    handshake_duration = loop.time() - handshake_start
                    logger.error(f"SSL handshake failed after {handshake_duration:.2f}s: {str(e)}")
                    if self._is_fatal_ssl_error(str(e)):
                        logger.error("Fatal SSL error detected, aborting handshake attempts")
                        raise
                    raise  # Re-raise for retry

                if not client_transport:
                    raise RuntimeError("TLS upgrade returned None transport")

                handshake_duration = loop.time() - handshake_start
                total_duration = loop.time() - start_time
                logger.info(
                    f"TLS handshake successful after {attempt + 1} attempts:\n"
                    f"  Handshake duration: {handshake_duration:.2f}s\n"
                    f"  Total setup time: {total_duration:.2f}s"
                )
                break

            except ConnectionError as e:
                if "Connection reset by peer" in str(e):
                    logger.warning("Client disconnected during handshake")
                    last_error = e
                    break
                last_error = e
                logger.error(f"Connection error during handshake (attempt {attempt + 1}): {e}")

            except asyncio.TimeoutError as e:
                last_error = e
                duration = loop.time() - start_time
                logger.error(f"Handshake timeout (attempt {attempt + 1}) after {duration:.2f}s")

            except Exception as e:
                last_error = e
                logger.error(f"TLS handshake error (attempt {attempt + 1}): {type(e).__name__}: {str(e)}")
                if isinstance(e, ssl.SSLError) and self._is_fatal_ssl_error(str(e)):
                    logger.error("Fatal SSL error detected, aborting handshake attempts")
                    break

        if not client_transport:
            total_duration = loop.time() - start_time
            error_msg = f"Failed to establish TLS connection after {total_duration:.2f}s"
            if last_error:
                error_msg += f": {type(last_error).__name__}: {str(last_error)}"
            raise RuntimeError(error_msg)

        # Store the upgraded connection
        self._client_transport = client_transport
        self._client_reader = client_reader
        self._client_writer = asyncio.StreamWriter(
            client_transport, protocol, client_reader, loop
        )

    @staticmethod
    def _is_fatal_ssl_error(error_msg: str) -> bool:
        """Check if an SSL error should be considered fatal."""
        return any(err in error_msg for err in SSLConfig.FATAL_SSL_ERRORS)

    async def _proxy_data(self,
                       client_reader: asyncio.StreamReader,
                       client_writer: asyncio.StreamWriter,
                       server_reader: asyncio.StreamReader,
                       server_writer: asyncio.StreamWriter) -> None:
        """Handle bidirectional proxy data transfer."""
        client_done = asyncio.Event()
        server_done = asyncio.Event()
        start_time = asyncio.get_event_loop().time()
        logger.info(f"Starting bidirectional proxy for {self.hostname}")

        # Take initial memory snapshot
        self._stats.add_memory_sample(start_time)

        async def pipe(reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter,
                      name: str,
                      done_event: asyncio.Event,
                      other_done: asyncio.Event) -> Tuple[int, int]:
            """Pipe data between reader and writer."""
            bytes_transferred = 0
            transfer_start = asyncio.get_event_loop().time()
            last_activity = transfer_start
            errors = 0
            retries = 0
            buffer = bytearray(NetworkConfig.BUFFER_SIZE)
            transfer_intervals: List[float] = []
            last_throughput_log = transfer_start
            last_memory_sample = transfer_start

            try:
                while not done_event.is_set() and not other_done.is_set():
                    try:
                        current_time = asyncio.get_event_loop().time()

                        # Take periodic memory samples
                        if current_time - last_memory_sample >= NetworkConfig.MEMORY_SAMPLE_INTERVAL:
                            self._stats.add_memory_sample(current_time)
                            last_memory_sample = current_time
                            
                            # Check for memory growth
                            deltas = self._stats.get_memory_deltas()
                            if deltas["rss"] > NetworkConfig.MEMORY_GROWTH_THRESHOLD:
                                logger.warning(
                                    f"{name}: Significant memory growth detected:\n"
                                    f"  RSS delta: {deltas['rss']/1024/1024:.1f}MB\n"
                                    f"  VMS delta: {deltas['vms']/1024/1024:.1f}MB"
                                )

                        # Check connection state periodically
                        if writer.is_closing():
                            logger.warning(f"{name}: Writer closed")
                            break

                        # Read with timeout and retry
                        read_success = False
                        read_retries = 0
                        while not read_success and read_retries < NetworkConfig.MAX_RETRIES:
                            try:
                                n = await asyncio.wait_for(
                                    reader.readinto(buffer),
                                    timeout=NetworkConfig.KEEPALIVE_TIME
                                )
                                read_success = True
                            except ConnectionError as e:
                                read_retries += 1
                                if read_retries >= NetworkConfig.MAX_RETRIES:
                                    logger.error(f"{name}: Max read retries ({read_retries}) reached: {e}")
                                    raise
                                logger.warning(f"{name}: Connection error during read (attempt {read_retries}): {e}")
                                await asyncio.sleep(NetworkConfig.RETRY_DELAY * (2 ** read_retries))
                                continue
                            except asyncio.TimeoutError:
                                if current_time - last_activity > NetworkConfig.INACTIVITY_TIMEOUT:
                                    logger.warning(f"{name}: Connection inactive for {NetworkConfig.INACTIVITY_TIMEOUT}s")
                                    break
                                continue

                        if not read_success:
                            break

                        if n == 0:  # EOF
                            logger.debug(f"{name}: Received EOF")
                            break

                        data = memoryview(buffer)[:n]
                        
                        # Write with timeout and retry
                        write_success = False
                        write_retries = 0
                        while not write_success and write_retries < NetworkConfig.MAX_RETRIES:
                            try:
                                writer.write(data)
                                await asyncio.wait_for(
                                    writer.drain(),
                                    timeout=NetworkConfig.KEEPALIVE_TIME
                                )
                                write_success = True
                            except ConnectionError as e:
                                write_retries += 1
                                if write_retries >= NetworkConfig.MAX_RETRIES:
                                    logger.error(f"{name}: Max write retries ({write_retries}) reached: {e}")
                                    raise
                                logger.warning(f"{name}: Connection error during write (attempt {write_retries}): {e}")
                                await asyncio.sleep(NetworkConfig.RETRY_DELAY * (2 ** write_retries))
                                continue
                            except asyncio.TimeoutError:
                                write_retries += 1
                                if write_retries >= NetworkConfig.MAX_RETRIES:
                                    logger.error(f"{name}: Write operation timed out after {write_retries} retries")
                                    raise
                                continue

                        if not write_success:
                            break

                        bytes_transferred += n
                        self.state.stats['bytes_transferred'] += n
                        current_time = asyncio.get_event_loop().time()
                        transfer_intervals.append(current_time - last_activity)
                        last_activity = current_time
                        self._stats.last_activity = last_activity

                        # Log throughput and memory metrics periodically
                        if current_time - last_throughput_log >= NetworkConfig.THROUGHPUT_LOG_INTERVAL:
                            interval_duration = current_time - last_throughput_log
                            recent_throughput = bytes_transferred / interval_duration
                            avg_interval = sum(transfer_intervals) / len(transfer_intervals) if transfer_intervals else 0
                            
                            # Get memory deltas
                            mem_deltas = self._stats.get_memory_deltas()
                            
                            logger.info(
                                f"{name}: Transfer metrics:\n"
                                f"  Current throughput: {recent_throughput/1024/1024:.2f} MB/s\n"
                                f"  Average interval: {avg_interval*1000:.2f}ms\n"
                                f"  Memory RSS delta: {mem_deltas['rss']/1024/1024:.1f}MB\n"
                                f"  Memory VMS delta: {mem_deltas['vms']/1024/1024:.1f}MB\n"
                                f"  Retries: {retries}\n"
                                f"  Errors: {errors}"
                            )
                            last_throughput_log = current_time
                            transfer_intervals = []

                    except Exception as e:
                        errors += 1
                        retries += 1
                        self._stats.errors += 1
                        logger.error(f"{name}: Transfer error: {type(e).__name__}: {str(e)}")
                        
                        if isinstance(e, (ssl.SSLError, ConnectionError)):
                            logger.error(f"{name}: Fatal connection error: {e}")
                            break
                            
                        if errors >= NetworkConfig.MAX_ERRORS or retries >= NetworkConfig.MAX_RETRIES:
                            logger.error(f"{name}: Too many errors ({errors}) or retries ({retries}), closing connection")
                            break
                            
                        await asyncio.sleep(NetworkConfig.RETRY_DELAY * (2 ** min(retries, 5)))
                        continue

            except Exception as e:
                logger.error(f"{name}: Pipe error: {type(e).__name__}: {str(e)}")
                self._stats.errors += 1

            finally:
                done_event.set()
                if not writer.is_closing():
                    writer.close()
                    
                duration = asyncio.get_event_loop().time() - transfer_start
                throughput = bytes_transferred / duration if duration > 0 else 0
                avg_interval = sum(transfer_intervals) / len(transfer_intervals) if transfer_intervals else 0
                
                # Final memory snapshot
                final_time = asyncio.get_event_loop().time()
                self._stats.add_memory_sample(final_time)
                mem_deltas = self._stats.get_memory_deltas()
                
                logger.info(
                    f"{name}: Connection closed.\n"
                    f"  Bytes transferred: {bytes_transferred:,}\n"
                    f"  Duration: {duration:.1f}s\n"
                    f"  Average throughput: {throughput/1024/1024:.2f} MB/s\n"
                    f"  Average interval: {avg_interval*1000:.2f}ms\n"
                    f"  Memory RSS delta: {mem_deltas['rss']/1024/1024:.1f}MB\n"
                    f"  Memory VMS delta: {mem_deltas['vms']/1024/1024:.1f}MB\n"
                    f"  Total retries: {retries}\n"
                    f"  Total errors: {errors}"
                )
                return bytes_transferred, errors

        try:
            logger.debug(f"Creating proxy pipes for {self.hostname}")
            
            # Create pipe tasks
            client_pipe = asyncio.create_task(
                pipe(client_reader, server_writer, "client->server", client_done, server_done)
            )
            server_pipe = asyncio.create_task(
                pipe(server_reader, client_writer, "server->client", server_done, client_done)
            )

            # Wait for pipes with timeout
            done, pending = await asyncio.wait(
                [client_pipe, server_pipe],
                timeout=NetworkConfig.PIPE_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            logger.debug(f"Cancelling {len(pending)} pending pipe tasks")
            for task in pending:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=NetworkConfig.CLEANUP_TIMEOUT)
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error during task cancellation: {e}")

            # Process results
            results = []
            for task in done:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error getting task result: {e}")
                    results.append((0, 1))

            if self._stats.handshake_completed:
                # Update total statistics only after handshake is done
                total_bytes = sum(r[0] for r in results)
                total_errors = sum(r[1] for r in results)
                total_duration = asyncio.get_event_loop().time() - start_time
                total_throughput = total_bytes / total_duration if total_duration > 0 else 0
                
                # Final memory deltas for the connection
                mem_deltas = self._stats.get_memory_deltas()
                
                logger.info(
                    f"Connection {self.hostname} completed:\n"
                    f"  Total bytes: {total_bytes:,}\n"
                    f"  Total duration: {total_duration:.1f}s\n"
                    f"  Average throughput: {total_throughput/1024/1024:.2f} MB/s\n"
                    f"  Total errors: {total_errors}\n"
                    f"  Final memory RSS delta: {mem_deltas['rss']/1024/1024:.1f}MB\n"
                    f"  Final memory VMS delta: {mem_deltas['vms']/1024/1024:.1f}MB"
                )

        except asyncio.TimeoutError:
            logger.error(f"Proxy operation timed out after {NetworkConfig.PIPE_TIMEOUT}s")
        except Exception as e:
            logger.error(f"Error in proxy operation: {type(e).__name__}: {str(e)}")
        finally:
            if not self._closing:
                await self.cleanup()
