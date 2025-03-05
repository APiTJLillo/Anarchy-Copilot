import asyncio
import socket
import ssl
from typing import Optional, Tuple

from ..utils.logging import logger
from ..utils.constants import NetworkConfig, SSLConfig
from .server_state import ServerState
from .ssl_context import SSLContextManager

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
        self._closing = False
        self._cleanup_lock = asyncio.Lock()
        self._server_transport = None
        self._client_transport = None

    async def cleanup(self) -> None:
        """Clean up connection resources."""
        if self._closing:
            return

        async with self._cleanup_lock:
            self._closing = True
            
            if self.hostname:
                try:
                    self.ssl_manager.remove_context(self.hostname)
                except Exception as e:
                    logger.debug(f"Error removing SSL context: {e}")

            # Clean up transports
            for transport in [self._server_transport, self._client_transport]:
                if transport:
                    try:
                        ssl_obj = transport.get_extra_info('ssl_object')
                        sock = transport.get_extra_info('socket')

                        if ssl_obj:
                            try:
                                ssl_obj.unwrap()
                            except ssl.SSLError as e:
                                if "PROTOCOL_IS_SHUTDOWN" not in str(e):
                                    logger.debug(f"SSL shutdown error: {e}")

                        if sock:
                            try:
                                sock.close()
                            except Exception as e:
                                logger.debug(f"Error closing socket: {e}")

                    except Exception as e:
                        logger.debug(f"Error cleaning up transport: {e}")

            # Close writers
            for writer in [self.writer]:
                if writer:
                    try:
                        writer.close()
                        await asyncio.wait_for(
                            writer.wait_closed(), 
                            timeout=NetworkConfig.CLEANUP_TIMEOUT
                        )
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.debug(f"Error closing writer: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def handle_connect(self, hostname: str, port: int) -> None:
        """Handle CONNECT request to target server."""
        self.hostname = hostname
        try:
            # Connect to target server
            server_reader, server_writer = await asyncio.open_connection(
                hostname, port,
                ssl=self.ssl_manager.create_server_context(hostname)
            )
            self._server_transport = server_writer.transport
            self._client_transport = self.writer.transport

            try:
                # Setup client SSL context
                client_ssl_context = self.ssl_manager.create_client_context(hostname)

                # Get raw socket from transport
                sock = self._client_transport.get_extra_info('socket')
                if not sock:
                    raise RuntimeError("Could not get raw socket from transport")

                # Set up TLS tunnel
                await self._setup_tls_tunnel(
                    sock, client_ssl_context, server_reader, server_writer
                )
            except Exception as e:
                logger.error(f"Error during SSL setup: {str(e)}")
                raise
            finally:
                await self.cleanup()

        except Exception as e:
            logger.error(f"Error handling CONNECT: {str(e)}")
            raise

    async def _setup_tls_tunnel(self,
                              sock: socket.socket,
                              ssl_context: ssl.SSLContext,
                              server_reader: asyncio.StreamReader,
                              server_writer: asyncio.StreamWriter) -> None:
        """Set up TLS tunnel between client and server."""
        loop = asyncio.get_event_loop()
        client_reader = asyncio.StreamReader(limit=NetworkConfig.BUFFER_SIZE)
        protocol = asyncio.StreamReaderProtocol(client_reader)

        # Perform TLS handshake with retries
        client_transport = None
        last_error = None

        for attempt in range(SSLConfig.MAX_HANDSHAKE_ATTEMPTS):
            try:
                delay = SSLConfig.HANDSHAKE_RETRY_DELAY * (2 ** attempt)
                if attempt > 0:
                    await asyncio.sleep(delay)

                client_transport = await asyncio.wait_for(
                    loop.start_tls(
                        self._client_transport,
                        protocol,
                        ssl_context,
                        server_side=True,
                        ssl_handshake_timeout=NetworkConfig.SSL_HANDSHAKE_TIMEOUT
                    ),
                    timeout=NetworkConfig.SSL_HANDSHAKE_TIMEOUT + 5.0  # Add buffer
                )

                if not client_transport:
                    raise RuntimeError("TLS upgrade returned None transport")

                break
            except ssl.SSLError as e:
                last_error = e
                logger.error(f"SSL error during handshake (attempt {attempt + 1}): {e}")
                if self._is_fatal_ssl_error(str(e)):
                    break
            except Exception as e:
                last_error = e
                logger.error(f"Error during handshake (attempt {attempt + 1}): {e}")
                if not isinstance(e, (ConnectionError, TimeoutError)):
                    break

        if not client_transport:
            raise last_error or RuntimeError("Failed to establish TLS tunnel")

        # Create writer from upgraded transport
        client_writer = asyncio.StreamWriter(
            client_transport, protocol, client_reader, loop
        )

        try:
            # Set up bidirectional proxy
            await self._proxy_data(
                client_reader, client_writer,
                server_reader, server_writer
            )
        finally:
            await self.cleanup()

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

        async def pipe(reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter,
                      name: str,
                      done_event: asyncio.Event,
                      other_done: asyncio.Event) -> Tuple[int, int]:
            """Pipe data between reader and writer, return (bytes_transferred, errors)."""
            bytes_transferred = 0
            transfer_start = asyncio.get_event_loop().time()
            last_activity = transfer_start
            errors = 0

            try:
                while not done_event.is_set() and not other_done.is_set():
                    try:
                        data = await asyncio.wait_for(
                            reader.read(NetworkConfig.BUFFER_SIZE),
                            timeout=NetworkConfig.KEEPALIVE_TIME
                        )
                        if not data:
                            break

                        writer.write(data)
                        await writer.drain()
                        
                        # Update statistics
                        bytes_transferred += len(data)
                        self.state.stats['bytes_transferred'] += len(data)
                        last_activity = asyncio.get_event_loop().time()

                    except asyncio.TimeoutError:
                        # Check for long inactivity
                        if asyncio.get_event_loop().time() - last_activity > NetworkConfig.INACTIVITY_TIMEOUT:
                            logger.warning(f"{name}: Connection inactive for {NetworkConfig.INACTIVITY_TIMEOUT}s")
                            break

                        # Send keepalive
                        if hasattr(writer, '_transport'):
                            ssl_obj = writer._transport.get_extra_info('ssl_object')
                            if ssl_obj:
                                try:
                                    ssl_obj.write(b'')
                                except (ssl.SSLWantWriteError, ssl.SSLError):
                                    break
                        continue

                    except Exception as e:
                        errors += 1
                        logger.error(f"{name}: Error during transfer: {e}")
                        if errors >= 5:  # Too many errors
                            logger.error(f"{name}: Too many errors, closing connection")
                            break
                        continue

            except Exception as e:
                logger.error(f"{name}: Fatal error: {e}")
            finally:
                done_event.set()
                await self.cleanup()
                
                # Log connection statistics
                duration = asyncio.get_event_loop().time() - transfer_start
                throughput = bytes_transferred / duration if duration > 0 else 0
                logger.info(
                    f"{name}: Connection closed.\n"
                    f"  Bytes transferred: {bytes_transferred:,}\n"
                    f"  Duration: {duration:.1f}s\n"
                    f"  Throughput: {throughput/1024/1024:.1f} MB/s\n"
                    f"  Errors: {errors}"
                )
                return bytes_transferred, errors

        # Run both pipes concurrently
        results = await asyncio.gather(
            pipe(client_reader, server_writer, "client->server", client_done, server_done),
            pipe(server_reader, client_writer, "server->client", server_done, client_done)
        )

        # Update total statistics
        total_bytes = sum(r[0] for r in results)
        total_errors = sum(r[1] for r in results)
        logger.info(f"Connection {self.hostname} completed: {total_bytes:,} bytes, {total_errors} errors")
