"""Custom protocol handler for HTTPS tunneling."""
import asyncio
import errno
import logging
import socket
from typing import Optional, Tuple, Any, Set
from urllib.parse import unquote

from uvicorn.protocols.http.h11_impl import H11Protocol
import h11

logger = logging.getLogger("proxy.core")

class TunnelProtocol(H11Protocol):
    """Custom protocol handler for HTTPS tunneling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tunnel_tasks: Set[asyncio.Task] = set()
        self._remote_transport = None
        self._remote_protocol = None

    def connection_made(self, transport: Any) -> None:
        """Handle new connection."""
        super().connection_made(transport)
        self._transport = transport

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection lost."""
        if self._remote_transport:
            self._remote_transport.close()
        super().connection_lost(exc)

    def data_received(self, data: bytes) -> None:
        """Handle incoming data."""
        if not hasattr(self, 'conn'):
            # In tunnel mode, forward to remote
            if self._remote_transport:
                try:
                    logger.debug(f"Forwarding {len(data)} bytes to remote")
                    self._remote_transport.write(data)
                    # Schedule flow control handling without awaiting
                    if len(data) > 65536:
                        asyncio.create_task(self._handle_large_write())
                except Exception as e:
                    logger.error(f"Error forwarding to remote: {e}")
                    self.transport.close()
            else:
                logger.warning("Received data in tunnel mode but no remote transport")
            return

        # Check if this looks like a CONNECT request
        if data.startswith(b'CONNECT '):
            logger.debug("Detected raw CONNECT request")
            try:
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
                
                # Create h11 request
                logger.debug(f"Creating CONNECT request for {target} with headers: {headers}")
                request = h11.Request(
                    method=b"CONNECT",
                    target=target.encode(),
                    headers=headers
                )
                asyncio.create_task(self.handle_request(request))
                return
            except Exception as e:
                logger.error(f"Failed to parse CONNECT request: {e}")
                self._send_error(502, str(e))
        
        # Regular HTTP data
        logger.debug("HTTP data received, passing to parser")
        super().data_received(data)

    async def _handle_large_write(self):
        """Handle flow control for large writes without blocking."""
        try:
            await asyncio.sleep(0.01)  # Brief pause to let writes process
            if hasattr(self, '_remote_transport') and self._remote_transport:
                self._remote_transport.resume_reading()
        except Exception as e:
            logger.error(f"Flow control error: {e}")

    async def handle_request(self, request: h11.Request) -> None:
        """Override request handling to intercept CONNECT method."""
        if request.method == b"CONNECT":
            try:
                logger.debug(f"TunnelProtocol received CONNECT request: {request.target.decode()}")
                host, port = self._parse_authority(request.target.decode())
                logger.debug(f"TunnelProtocol establishing tunnel to {host}:{port}")
                await self._handle_connect(host, port)
            except asyncio.CancelledError:
                logger.debug(f"CONNECT tunnel cancelled")
                self.transport.close()
            except Exception as e:
                logger.error(f"Failed to handle CONNECT: {e}")
                self._send_error(502, str(e))
        else:
            # Pass non-CONNECT requests to parent handler
            await super().handle_request(request)

    def _send_error(self, status: int, message: str) -> None:
        """Send error response directly through transport."""
        response = (
            f"HTTP/1.1 {status} Error\r\n"
            "Content-Type: text/plain\r\n"
            f"Content-Length: {len(message)}\r\n"
            "\r\n"
            f"{message}"
        ).encode()
        self.transport.write(response)
        self.transport.close()

    async def _handle_connect(self, host: str, port: int) -> None:
        """Handle CONNECT request by establishing a tunnel."""
        try:
            # Create connection to remote host
            logger.debug(f"CONNECT attempt: {host}:{port}")
            loop = asyncio.get_event_loop()

            # Create the remote connection
            self._remote_transport, self._remote_protocol = await loop.create_connection(
                lambda: TunnelTransport(self.transport),
                host=host,
                port=port
            )

            logger.debug(f"Remote connect success for {host}:{port}")

            # Send 200 Connection Established
            logger.debug(f"Connected to {host}:{port}, sending 200 response")
            response = (
                b"HTTP/1.1 200 Connection Established\r\n"
                b"Connection: keep-alive\r\n"
                b"\r\n"
            )
            self.transport.write(response)
            logger.debug(f"[{host}:{port}] Sent 200 response")

            # Switch to tunnel mode
            logger.debug(f"[{host}:{port}] Switching to tunnel mode")
            if hasattr(self, 'conn'):
                logger.debug(f"[{host}:{port}] Clearing HTTP state")
                del self.conn

        except Exception as e:
            logger.error(f"Tunnel error for {host}:{port}: {e}")
            self._send_error(502, str(e))
            if self._remote_transport:
                self._remote_transport.close()

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

class TunnelTransport(asyncio.Protocol):
    """Protocol for tunnel connection."""
    
    def __init__(self, client_transport):
        self.transport = None
        self.client_transport = client_transport
        self._write_buffer_size = 0
        self._paused = False

    def connection_made(self, transport):
        """Store the transport for use."""
        self.transport = transport
        # Configure higher watermarks for better throughput
        transport.set_write_buffer_limits(high=256 * 1024)
        transport.get_extra_info('socket').setsockopt(
            socket.SOL_SOCKET,
            socket.SO_KEEPALIVE,
            1
        )

    def pause_writing(self):
        """Called when transport buffer is full."""
        logger.debug("Remote transport buffer full, pausing writes")
        self._paused = True
        if self.client_transport:
            self.client_transport.pause_reading()

    def resume_writing(self):
        """Called when transport buffer has drained."""
        logger.debug("Remote transport buffer drained, resuming writes")
        self._paused = False
        if self.client_transport:
            self.client_transport.resume_reading()

    def data_received(self, data):
        """Forward data to client."""
        try:
            size = len(data)
            logger.debug(f"Remote received {size} bytes")
            
            # Handle flow control
            if self._write_buffer_size > 256 * 1024 and not self._paused:
                self.pause_writing()
            else:
                self.client_transport.write(data)
                self._write_buffer_size += size
                if not self._paused:
                    asyncio.create_task(self._check_buffer())

        except Exception as e:
            logger.error(f"Error forwarding to client: {e}")
            if self.transport:
                self.transport.close()

    async def _check_buffer(self):
        """Monitor write buffer size."""
        try:
            if self._write_buffer_size > 0:
                await asyncio.sleep(0.01)
                self._write_buffer_size = 0
                if self._paused:
                    self.resume_writing()
        except Exception as e:
            logger.error(f"Buffer check error: {e}")

    def connection_lost(self, exc):
        """Close client connection when remote closes."""
        if exc:
            logger.debug(f"Remote connection lost with error: {exc}")
        else:
            logger.debug("Remote connection closed normally")
        if self.client_transport:
            self.client_transport.close()
