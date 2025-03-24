"""Mock HTTPS server for testing proxy behavior."""
import asyncio
import ssl
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import json
from dataclasses import dataclass
from datetime import datetime
import random
import time
from http import HTTPStatus
import re

logger = logging.getLogger(__name__)

@dataclass
class ServerConfig:
    """Configuration for mock server behavior."""
    port: int = 0  # 0 for automatic port selection
    latency_range: tuple[float, float] = (0.01, 0.05)  # Random latency range
    error_rate: float = 0.0  # Probability of generating errors
    chunk_size: int = 8192  # Size of data chunks
    ssl_cert: Optional[str] = None  # Path to SSL certificate
    ssl_key: Optional[str] = None   # Path to SSL private key
    rate_limit: int = 100  # Requests per second
    burst_limit: int = 200  # Max burst size
    max_header_size: int = 8192  # Maximum size per header
    max_headers: int = 100  # Maximum number of headers

class ResponseTemplate:
    """Template for mock responses."""
    def __init__(self, 
                 status: int = 200,
                 headers: Optional[Dict[str, str]] = None,
                 body: Optional[bytes] = None,
                 latency: Optional[float] = None,
                 chunk_size: Optional[int] = None,
                 use_chunked_encoding: bool = False):
        self.status = status
        self.headers = headers or {}
        self.body = body or b'{"status": "ok"}'
        self.latency = latency
        self.chunk_size = chunk_size
        self.use_chunked_encoding = use_chunked_encoding
        
        # Ensure required headers are present
        if "content-type" not in {k.lower() for k in self.headers}:
            self.headers["Content-Type"] = "application/json"
        if "server" not in {k.lower() for k in self.headers}:
            self.headers["Server"] = "MockServer/1.0"

class MockHttpsServer:
    """Mock HTTPS server for testing proxy behavior."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.routes: Dict[str, Callable] = {}
        self.server: Optional[asyncio.Server] = None
        self._start_time = datetime.now()
        self._request_count = 0
        self._bytes_received = 0
        self._bytes_sent = 0
        self._active_connections = 0
        self._error_count = 0
        self._rate_limiter = {}
        self._header_pattern = re.compile(r'^[\x21-\x7E]+$')  # Printable ASCII chars only
        self._method_pattern = re.compile(r'^[A-Z]+$')  # Valid HTTP methods

    def route(self, path: str):
        """Decorator to register route handlers."""
        def decorator(handler: Callable):
            self.routes[path] = handler
            return handler
        return decorator

    async def default_handler(self, request: Dict[str, Any]) -> ResponseTemplate:
        """Default request handler."""
        return ResponseTemplate(
            status=200,
            body=json.dumps({
                "method": request["method"],
                "path": request["path"],
                "headers": request["headers"],
                "timestamp": datetime.now().isoformat()
            }).encode()
        )

    def _is_rate_limited(self, peer: tuple) -> bool:
        """Check if request should be rate limited."""
        now = time.time()
        if peer not in self._rate_limiter:
            self._rate_limiter[peer] = {"count": 0, "window": now}
        
        # Reset window if needed
        if now - self._rate_limiter[peer]["window"] >= 1:
            self._rate_limiter[peer] = {"count": 0, "window": now}
        
        self._rate_limiter[peer]["count"] += 1
        
        # Check limits
        if (self._rate_limiter[peer]["count"] > self.config.rate_limit or
            len(self._rate_limiter) > self.config.burst_limit):
            return True
        
        return False

    def _validate_headers(self, headers: Dict[str, str], method: str) -> Optional[int]:
        """Validate headers and return error status if invalid."""
        # Check header count
        if len(headers) > self.config.max_headers:
            return 431  # Request Header Fields Too Large
        
        # Check header sizes and format
        for name, value in headers.items():
            # Check size
            if len(name) + len(value) > self.config.max_header_size:
                return 431
            
            # Validate header name format
            if not self._header_pattern.match(name):
                return 400
            
            # Check for newlines (header injection)
            if '\n' in value or '\r' in value:
                return 400
        
        # Validate required headers
        if method != "CONNECT" and "host" not in {k.lower() for k in headers}:
            return 400
        
        # Check conflicting headers
        has_content_length = "content-length" in {k.lower() for k in headers}
        has_transfer_encoding = "transfer-encoding" in {k.lower() for k in headers}
        if has_content_length and has_transfer_encoding:
            return 400
        
        return None

    def _get_status_phrase(self, code: int) -> str:
        """Get the correct reason phrase for a status code."""
        try:
            return HTTPStatus(code).phrase
        except ValueError:
            return "Unknown"

    async def _read_request_body(self, headers: Dict[str, str], reader: asyncio.StreamReader) -> bytes:
        """Read request body based on headers."""
        body = b""
        try:
            if "content-length" in headers:
                length = int(headers["content-length"])
                if length > 0:
                    body = await reader.readexactly(length)
            elif "transfer-encoding" in headers and headers["transfer-encoding"].lower() == "chunked":
                while True:
                    size_line = await reader.readline()
                    if not size_line:
                        break
                    chunk_size = int(size_line.strip().decode(), 16)
                    if chunk_size == 0:
                        await reader.readline()  # Read final CRLF
                        break
                    chunk = await reader.readexactly(chunk_size)
                    body += chunk
                    await reader.readline()  # Read CRLF after chunk
        except (ValueError, asyncio.IncompleteReadError):
            return None
        return body

    async def handle_client(self, reader: asyncio.StreamReader, 
                          writer: asyncio.StreamWriter):
        """Handle client connection."""
        self._active_connections += 1
        peer = writer.get_extra_info('peername')
        logger.debug(f"New connection from {peer}")
        
        try:
            while True:
                # Read request line
                request_line = await reader.readline()
                if not request_line:
                    break
                    
                try:
                    # Parse request
                    method, path, http_version = request_line.decode().strip().split(" ")
                    
                    # Validate method
                    if not self._method_pattern.match(method):
                        await self._send_error(writer, 400, "Invalid method")
                        break
                    
                    # Validate HTTP version
                    if not http_version.startswith("HTTP/"):
                        await self._send_error(writer, 400, "Invalid HTTP version")
                        break
                
                except (ValueError, UnicodeDecodeError):
                    await self._send_error(writer, 400, "Invalid request line")
                    break
                
                # Read headers
                headers = {}
                header_bytes = 0
                while True:
                    header_line = await reader.readline()
                    if header_line == b"\r\n":
                        break
                    try:
                        name, value = header_line.decode().strip().split(": ", 1)
                        headers[name.lower()] = value
                        header_bytes += len(header_line)
                    except (ValueError, UnicodeDecodeError):
                        await self._send_error(writer, 400, "Invalid header format")
                        return
                
                # Validate headers
                error_status = self._validate_headers(headers, method)
                if error_status:
                    await self._send_error(writer, error_status, "Header validation failed")
                    break
                
                # Check rate limiting
                if self._is_rate_limited(peer):
                    await self._send_rate_limit_response(writer)
                    break
                
                # Read body
                body = await self._read_request_body(headers, reader)
                if body is None:
                    await self._send_error(writer, 400, "Invalid message body")
                    break
                
                self._request_count += 1
                self._bytes_received += len(request_line) + header_bytes + len(body)
                
                # Protocol downgrade prevention
                if ("1.0" in http_version and 
                    any(h in headers for h in ("upgrade", "http2-settings"))):
                    await self._send_error(writer, 400, "Protocol downgrade not allowed")
                    break
                
                # Prepare request object
                request = {
                    "method": method,
                    "path": path,
                    "http_version": http_version.split("/")[1],
                    "headers": headers,
                    "body": body,
                    "peer": peer
                }
                
                # Handle request
                try:
                    if random.random() < self.config.error_rate:
                        response = ResponseTemplate(
                            status=500,
                            body=b'{"error": "Random server error"}'
                        )
                        self._error_count += 1
                    else:
                        handler = self.routes.get(path, self.default_handler)
                        response = await handler(request)
                    
                    await self._send_response(writer, response)
                    
                except Exception as e:
                    logger.error(f"Error handling request: {str(e)}")
                    await self._send_error(writer, 500, "Internal server error")
                    self._error_count += 1
                
                # Close if not keep-alive
                if headers.get("connection", "").lower() != "keep-alive":
                    break
                    
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
        finally:
            writer.close()
            await writer.wait_closed()
            self._active_connections -= 1
            logger.debug(f"Connection closed from {peer}")

    async def _send_response(self, writer: asyncio.StreamWriter, response: ResponseTemplate):
        """Send response with proper formatting."""
        try:
            # Get correct reason phrase
            status_phrase = self._get_status_phrase(response.status)
            status_line = f"HTTP/1.1 {response.status} {status_phrase}\r\n"
            writer.write(status_line.encode())
            
            # Add standard headers if not present
            headers = response.headers.copy()
            if "date" not in {k.lower() for k in headers}:
                headers["Date"] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
            
            # Handle content length and transfer encoding
            if response.use_chunked_encoding:
                headers["Transfer-Encoding"] = "chunked"
                if "content-length" in headers:
                    del headers["content-length"]
            else:
                headers["Content-Length"] = str(len(response.body))
                if "transfer-encoding" in headers:
                    del headers["transfer-encoding"]
            
            # Send headers
            for name, value in headers.items():
                header_line = f"{name}: {value}\r\n"
                writer.write(header_line.encode())
            writer.write(b"\r\n")
            
            # Send body
            if response.use_chunked_encoding:
                # Send in chunks
                chunk_size = response.chunk_size or self.config.chunk_size
                for i in range(0, len(response.body), chunk_size):
                    chunk = response.body[i:i + chunk_size]
                    writer.write(f"{len(chunk):X}\r\n".encode())
                    writer.write(chunk)
                    writer.write(b"\r\n")
                    await writer.drain()
                    await asyncio.sleep(0.001)
                writer.write(b"0\r\n\r\n")
            else:
                writer.write(response.body)
            
            await writer.drain()
            self._bytes_sent += (len(status_line) + 
                               sum(len(k) + len(v) + 4 for k, v in headers.items()) + 
                               len(response.body))
            
        except Exception as e:
            logger.error(f"Error sending response: {str(e)}")
            raise

    async def _send_error(self, writer: asyncio.StreamWriter, status: int, message: str):
        """Send error response."""
        response = ResponseTemplate(
            status=status,
            body=json.dumps({"error": message}).encode()
        )
        await self._send_response(writer, response)

    async def _send_rate_limit_response(self, writer: asyncio.StreamWriter):
        """Send rate limit response."""
        headers = {
            "Retry-After": "1",
            "X-RateLimit-Limit": str(self.config.rate_limit),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(time.time() + 1))
        }
        response = ResponseTemplate(status=429, headers=headers)
        await self._send_response(writer, response)

    async def start(self):
        """Start the mock server."""
        ssl_context = None
        if self.config.ssl_cert and self.config.ssl_key:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(self.config.ssl_cert, self.config.ssl_key)

        self.server = await asyncio.start_server(
            self.handle_client,
            'localhost',
            self.config.port,
            ssl=ssl_context
        )
        
        for sock in self.server.sockets:
            logger.info(f"Mock server running on {sock.getsockname()}")

    async def stop(self):
        """Stop the mock server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("Mock server stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        uptime = (datetime.now() - self._start_time).total_seconds()
        return {
            "uptime": uptime,
            "requests": {
                "total": self._request_count,
                "errors": self._error_count,
                "active_connections": self._active_connections,
                "requests_per_second": self._request_count / uptime if uptime > 0 else 0
            },
            "bytes": {
                "received": self._bytes_received,
                "sent": self._bytes_sent,
                "throughput": (self._bytes_sent + self._bytes_received) / uptime if uptime > 0 else 0
            }
        }
