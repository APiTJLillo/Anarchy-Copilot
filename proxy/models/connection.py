"""Connection handling for proxy server."""
from datetime import datetime
import asyncio
import socket
import ssl
import struct
import errno
import enum
import os
import uuid
from typing import Optional, Tuple, Dict, Any, List, cast
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logging import logger
from ..utils.config_types import NetworkConfigDict, SSLConfigDict, MemoryConfigDict, EnvVarsDict, HTTPStatusDict
from ..utils.config_instance import (
    network_config, ssl_config, memory_config, env_vars, http_status
)
from .server_state import ServerState
from .ssl_context import SSLContextManager
from proxy.interceptors.database import DatabaseInterceptor

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
    memory_samples: List[MemoryStats] = field(default_factory=list)  # Fixed type annotation syntax

    def add_memory_sample(self, timestamp: float) -> None:
        """Add a memory usage sample."""
        process = os.getpid()
        try:
            with open(f"/proc/{process}/status") as f:
                lines = f.readlines()
                vm_rss = 0
                vm_size = 0
                vm_shared = 0
                
                for line in lines:
                    if line.startswith('VmRSS:'):
                        vm_rss = int(line.split()[1]) * 1024
                    elif line.startswith('VmSize:'):
                        vm_size = int(line.split()[1]) * 1024
                    elif line.startswith('RssFile:'):  # Shared memory estimate
                        vm_shared = int(line.split()[1]) * 1024
                
                self.memory_samples.append(MemoryStats(
                    rss=vm_rss,
                    vms=vm_size,
                    shared=vm_shared,
                    timestamp=timestamp
                ))
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to collect memory stats: {e}")
            return
        
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
    
    def __init__(self, reader: asyncio.StreamReader,
                writer: asyncio.StreamWriter,
                ssl_manager: SSLContextManager,
                server_state: ServerState):
        """Initialize connection."""
        self.reader = reader
        self.writer = writer 
        self.ssl_manager = ssl_manager
        self.state = server_state
        self.client_addr = writer.get_extra_info('peername')
        self.hostname = None
        self._state = ConnectionState.INITIAL
        self._closing = False
        self._cleanup_lock = asyncio.Lock()
        self._server_transport = None
        self._client_transport = None
        self._client_reader = None
        self._client_writer = None
        self._server_reader = None 
        self._server_writer = None
        self._stats = ConnectionStats()
        self._last_error = None
        self._stats.start_time = asyncio.get_event_loop().time()
        self._stats.last_activity = self._stats.start_time
        self._tls_handshake_done = asyncio.Event()
        self._connect_response_sent = asyncio.Event()
        self._client_hello_received = asyncio.Event()
        self._db_interceptor = None
        self.connection_id = str(uuid.uuid4())

    async def _setup_database_interceptor(self) -> None:
        """Initialize database interceptor."""
        try:
            if self._db_interceptor is None:
                logger.info(f"[{self.connection_id}] Setting up database interceptor")
                self._db_interceptor = DatabaseInterceptor(self.connection_id)
                # Do an initial active session check to validate database connectivity
                session_id = await self._db_interceptor._get_session_id()
                if session_id:
                    logger.info(f"[{self.connection_id}] Connected to active session {session_id}")
                else:
                    logger.warning(f"[{self.connection_id}] No active session found during interceptor setup")
        except Exception as e:
            logger.error(f"[{self.connection_id}] Failed to setup database interceptor: {e}", exc_info=True)
            # Don't re-raise - we want the proxy to work even if database logging fails
            self._db_interceptor = None

    def _set_state(self, new_state: ConnectionState) -> None:
        """Update connection state with logging."""
        old_state = self._state
        self._state = new_state
        logger.debug(f"[{self.connection_id}] Connection {self.client_addr} state change: {old_state.value} -> {new_state.value}")

    async def cleanup(self) -> None:
        """Clean up connection resources."""
        if self._closing:
            return

        async with self._cleanup_lock:
            try:
                self._closing = True
                self._set_state(ConnectionState.CLOSING)

                # Close database interceptor
                if self._db_interceptor:
                    await self._db_interceptor.close()
                    self._db_interceptor = None

                # Close all streams
                if self._client_writer and not self._client_writer.is_closing():
                    self._client_writer.close()
                    await self._client_writer.wait_closed()

                if self._server_writer and not self._server_writer.is_closing():
                    self._server_writer.close()
                    await self._server_writer.wait_closed()

                # Clear all references
                self._client_reader = None
                self._client_writer = None
                self._server_reader = None
                self._server_writer = None
                self._client_transport = None
                self._server_transport = None

                self._set_state(ConnectionState.CLOSED)
                logger.debug(f"[{self.connection_id}] Connection cleaned up")

            except Exception as e:
                logger.error(f"[{self.connection_id}] Error during cleanup: {e}")
                self._set_state(ConnectionState.ERROR)
                self._last_error = e

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_val:
            logger.error(f"[{self.connection_id}] Connection error: {exc_type.__name__}: {str(exc_val)}")
            self._set_state(ConnectionState.ERROR)
            self._last_error = exc_val
        await self.cleanup()

    async def _wait_for_client_hello(self, reader: asyncio.StreamReader, timeout: Optional[float] = None) -> bool:
        """Wait for and validate TLS Client Hello message."""
        try:
            if timeout is None:
                timeout = network_config.CLIENT_HELLO_TIMEOUT

            start_time = asyncio.get_event_loop().time()
            logger.debug(f"[{self.connection_id}] Waiting {network_config.RESPONSE_WAIT}s for client readiness")

            # Wait after sending 200 to ensure client is ready
            await asyncio.sleep(network_config.RESPONSE_WAIT)
            
            attempts = 0
            max_attempts = ssl_config.MAX_HANDSHAKE_ATTEMPTS
            
            while attempts < max_attempts:
                try:
                    if attempts > 0:
                        delay = min(
                            ssl_config.HANDSHAKE_RETRY_DELAY * (2 ** attempts),
                            ssl_config.MAX_HANDSHAKE_DELAY
                        )
                        logger.debug(f"[{self.connection_id}] Retrying ClientHello detection (attempt {attempts + 1}/{max_attempts})")
                        await asyncio.sleep(delay)

                    # Read TLS record header (5 bytes)
                    header = await asyncio.wait_for(reader.readexactly(ssl_config.TLS_RECORD_HEADER_SIZE), timeout=timeout)
                    if not header or len(header) != ssl_config.TLS_RECORD_HEADER_SIZE:
                        attempts += 1
                        continue

                    # Parse TLS record
                    content_type, version_major, version_minor, length = struct.unpack('!BBBH', header)
                    
                    # Validate it's a handshake record (type 22) and appears to be ClientHello
                    if content_type == 22:  # Handshake record
                        # Peek at handshake message type
                        handshake_header = await asyncio.wait_for(reader.readexactly(1), timeout=timeout)
                        if handshake_header[0] == 1:  # ClientHello
                            logger.debug(f"[{self.connection_id}] Detected ClientHello")
                            return True
                    
                    attempts += 1

                except (asyncio.TimeoutError, asyncio.IncompleteReadError) as e:
                    logger.warning(f"[{self.connection_id}] Error reading ClientHello (attempt {attempts + 1}): {e}")
                    attempts += 1
                    continue

            logger.error(f"[{self.connection_id}] Failed to detect ClientHello after {max_attempts} attempts")
            return False

        except (struct.error, Exception) as e:
            logger.error(f"[{self.connection_id}] Error during ClientHello detection: {type(e).__name__}: {str(e)}")
            return False

    async def handle_connect(self, hostname: str, port: int) -> None:
        """Handle CONNECT request to target server."""
        if not isinstance(hostname, str) or not isinstance(port, int):
            raise ValueError("Invalid hostname or port")

        self.hostname = hostname
        self._set_state(ConnectionState.CONNECTING)
        await self._setup_database_interceptor()
        logger.debug(f"[{self.connection_id}] Handling CONNECT for {hostname}:{port}")

        try:
            # First establish connection to target server
            try:
                reader, writer = await asyncio.open_connection(
                    hostname, 
                    port,
                    ssl=None,  # We'll handle SSL/TLS separately
                    local_addr=None,
                    server_hostname=None
                )
                self._server_reader = reader
                self._server_writer = writer
                self._set_state(ConnectionState.CONNECTED)
                logger.debug(f"[{self.connection_id}] Connected to {hostname}:{port}")

            except Exception as e:
                logger.error(f"[{self.connection_id}] Failed to connect to {hostname}:{port}: {e}")
                if self._db_interceptor:
                    await self._db_interceptor.close()
                raise

            # Set up TLS
            if self._client_reader and self._client_writer:
                await self._setup_client_tls(hostname)
                self._set_state(ConnectionState.ESTABLISHED)
                logger.debug(f"[{self.connection_id}] TLS established with {hostname}")

                # Start proxying data
                if self._client_reader and self._client_writer and self._server_reader and self._server_writer:
                    await self._proxy_data(
                        cast(asyncio.StreamReader, self._client_reader),
                        cast(asyncio.StreamWriter, self._client_writer),
                        cast(asyncio.StreamReader, self._server_reader),
                        cast(asyncio.StreamWriter, self._server_writer)
                    )
                else:
                    raise RuntimeError("Missing stream objects after TLS setup")
            else:
                raise RuntimeError("Client connection not initialized")

        except Exception as e:
            logger.error(f"[{self.connection_id}] Error handling CONNECT: {e}")
            self._set_state(ConnectionState.ERROR)
            self._last_error = e
            if not self._closing:
                await self.cleanup()
            raise

    async def _setup_client_tls(self, hostname: str) -> None:
        """Set up TLS for the client connection."""
        loop = asyncio.get_event_loop()
        start_time = loop.time()
        client_reader = asyncio.StreamReader(limit=network_config.BUFFER_SIZE)
        protocol = asyncio.StreamReaderProtocol(client_reader)

        # Create SSL context
        client_context = self.ssl_manager.create_client_context(hostname)
        
        # Perform TLS handshake with retries
        client_transport = None
        last_error = None

        for attempt in range(ssl_config.MAX_HANDSHAKE_ATTEMPTS):
            try:
                if attempt > 0:
                    delay = min(
                        ssl_config.HANDSHAKE_RETRY_DELAY * (2 ** attempt),
                        ssl_config.MAX_HANDSHAKE_DELAY
                    )
                    logger.debug(f"Retrying client TLS handshake (attempt {attempt + 1}/{ssl_config.MAX_HANDSHAKE_ATTEMPTS}, delay: {delay:.2f}s)")
                    await asyncio.sleep(delay)

                handshake_start = loop.time()
                try:
                    client_transport = await asyncio.wait_for(
                        loop.start_tls(
                            transport=self.writer.transport,
                            protocol=protocol,
                            sslcontext=client_context,
                            server_side=True,
                            ssl_handshake_timeout=network_config.SSL_HANDSHAKE_TIMEOUT
                        ),
                        timeout=network_config.SSL_HANDSHAKE_TIMEOUT + 5.0
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
        return any(err in error_msg for err in ssl_config.FATAL_SSL_ERRORS)

    async def _proxy_data(self,
                       client_reader: asyncio.StreamReader,
                       client_writer: asyncio.StreamWriter,
                       server_reader: asyncio.StreamReader,
                       server_writer: asyncio.StreamWriter) -> None:
        """Handle bidirectional proxy data transfer."""
        if not all([client_reader, client_writer, server_reader, server_writer]):
            raise ValueError("Missing required stream objects")

        try:
            client_done = asyncio.Event()
            server_done = asyncio.Event()
            start_time = asyncio.get_event_loop().time()
            logger.info(f"[{self.connection_id}] Starting bidirectional proxy for {self.hostname}")

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
                buffer = bytearray(network_config.BUFFER_SIZE)
                transfer_intervals: List[float] = []
                last_throughput_log = transfer_start
                last_memory_sample = transfer_start

                try:
                    while not done_event.is_set() and not other_done.is_set():
                        try:
                            current_time = asyncio.get_event_loop().time()

                            # Take periodic memory samples
                            if current_time - last_memory_sample >= network_config.MEMORY_SAMPLE_INTERVAL:
                                self._stats.add_memory_sample(current_time)
                                last_memory_sample = current_time
                                
                                # Check for memory growth
                                deltas = self._stats.get_memory_deltas()
                                if deltas["rss"] > network_config.MEMORY_GROWTH_THRESHOLD:
                                    logger.warning(
                                        f"{name}: Significant memory growth detected:\n"
                                        f"  RSS delta: {deltas['rss']/1024/1024:.1f}MB\n"
                                        f"  VMS delta: {deltas['vms']/1024/1024:.1f}MB"
                                    )

                            # Check connection state periodically
                            if writer.is_closing():
                                logger.warning(f"{name}: Writer closed")
                                break

                            # Read with timeout
                            data = await asyncio.wait_for(
                                reader.read(network_config.BUFFER_SIZE),
                                timeout=network_config.KEEPALIVE_TIME
                            )

                            if not data:  # EOF
                                logger.debug(f"{name}: Received EOF")
                                break

                            if len(data) == 0:
                                continue  # Skip empty data

                            # Store in database if interceptor available
                            if self._db_interceptor:
                                try:
                                    if name == "client->server":
                                        await self._db_interceptor.store_raw_request(data, host=self.hostname)
                                    else:
                                        await self._db_interceptor.store_raw_response(data, host=self.hostname)
                                except Exception as e:
                                    logger.error(f"[{self.connection_id}] Database intercept error: {e}")
                                    # Continue proxying even if database storage fails
                                    errors += 1

                            # Write data
                            writer.write(data)
                            await writer.drain()

                            bytes_transferred += len(data)
                            self.state.stats['bytes_transferred'] += len(data)
                            current_time = asyncio.get_event_loop().time()
                            transfer_intervals.append(current_time - last_activity)
                            last_activity = current_time
                            self._stats.last_activity = last_activity

                            # Log throughput periodically
                            if current_time - last_throughput_log >= network_config.THROUGHPUT_LOG_INTERVAL:
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

                        except asyncio.TimeoutError:
                            # Normal timeout, just continue
                            continue
                        except Exception as e:
                            errors += 1
                            retries += 1
                            self._stats.errors += 1
                            logger.error(f"{name}: Transfer error: {type(e).__name__}: {str(e)}")
                            
                            if isinstance(e, (ssl.SSLError, ConnectionError)):
                                logger.error(f"{name}: Fatal connection error: {e}")
                                break
                                
                            if errors >= network_config.MAX_ERRORS or retries >= network_config.MAX_RETRIES:
                                logger.error(f"{name}: Too many errors ({errors}) or retries ({retries}), closing connection")
                                break
                                
                            await asyncio.sleep(network_config.RETRY_DELAY * (2 ** min(retries, 5)))
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
                        f"[{self.connection_id}] {name}: Connection closed.\n"
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
                logger.debug(f"[{self.connection_id}] Creating proxy pipes for {self.hostname}")
                
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
                    timeout=network_config.PIPE_TIMEOUT,
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel any pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Get results
                results = []
                for task in done:
                    try:
                        result = await task
                        results.append(result)
                    except Exception as e:
                        logger.error(f"[{self.connection_id}] Error getting pipe results: {e}")

                if results:
                    total_bytes, total_errors = zip(*results)
                    logger.info(f"[{self.connection_id}] Proxy complete: {sum(total_bytes):,} bytes, {sum(total_errors)} errors")

            except asyncio.TimeoutError:
                logger.error(f"[{self.connection_id}] Proxy operation timed out after {network_config.PIPE_TIMEOUT}s")
            except Exception as e:
                logger.error(f"[{self.connection_id}] Error in proxy operation: {type(e).__name__}: {str(e)}")
            finally:
                if not self._closing:
                    await self.cleanup()

        except Exception as e:
            logger.error(f"[{self.connection_id}] Fatal error in proxy data handler: {type(e).__name__}: {str(e)}")
            if not self._closing:
                await self.cleanup()
            raise
