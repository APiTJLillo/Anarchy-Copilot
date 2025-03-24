"""HTTP request/response handling for HTTPS interception."""
import h11
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
import asyncio
from async_timeout import timeout
from sqlalchemy import text
from urllib.parse import urlparse

from database import AsyncSessionLocal
from api.proxy.database_models import ProxyHistoryEntry
from proxy.core.tls import get_tls_context, CertificateManager
from ..types import ConnectionManagerProtocol
from proxy.interceptor import InterceptedRequest, InterceptedResponse, ProxyInterceptor

logger = logging.getLogger("proxy.core")

class ProxyResponse:
    """Represents an HTTP response from the proxy server."""
    
    def __init__(self, status_code: int = 200, headers: Dict[str, str] = None, body: bytes = None):
        self.status_code = status_code
        self.headers = headers or {}
        self.body = body or b""
        
    def to_h11(self) -> Tuple[h11.Response, h11.Data, h11.EndOfMessage]:
        """Convert to h11 response objects."""
        response = h11.Response(
            status_code=self.status_code,
            headers=[(k.encode(), v.encode()) for k, v in self.headers.items()]
        )
        data = h11.Data(data=self.body)
        end = h11.EndOfMessage()
        return response, data, end

class HttpRequestHandler:
    """Handles HTTP requests and responses."""
    
    def __init__(self, connection_id: str, connection_manager: ConnectionManagerProtocol):
        self.connection_id = connection_id
        self.connection_manager = connection_manager
        self.transport: Optional[asyncio.Transport] = None
        self.remote_transport: Optional[asyncio.Transport] = None
        self.remote_protocol: Optional[asyncio.Protocol] = None
        self.request_buffer = bytearray()
        self.response_buffer = bytearray()
        self.current_request: Dict[str, Any] = {}
        self._current_request: Optional[InterceptedRequest] = None
        self._current_response: Optional[InterceptedResponse] = None
        self._interceptor: Optional[ProxyInterceptor] = None
        self._h11_client = h11.Connection(h11.SERVER)
        self._h11_server = h11.Connection(h11.CLIENT)
        self._request_complete = False
        self._response_complete = False
        
    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle new client connection."""
        self.transport = transport
        self.connection_manager.create_connection(self.connection_id, transport)
        logger.debug(f"New HTTP connection {self.connection_id}")
        
    def data_received(self, data: bytes) -> None:
        """Handle received data from client."""
        self.request_buffer.extend(data)
        
        # Try to parse complete request
        if self._is_complete_request():
            self._handle_request()
            
    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle client disconnection."""
        if exc:
            logger.error(f"Connection {self.connection_id} lost with error: {exc}")
            self.connection_manager.update_connection(self.connection_id, "error", str(exc))
            
        # Close remote connection if still open
        if self.remote_transport and not self.remote_transport.is_closing():
            self.remote_transport.close()
            
        # Clean up connection tracking
        self.connection_manager.close_connection(self.connection_id)
        
    def _is_complete_request(self) -> bool:
        """Check if we have received a complete HTTP request."""
        try:
            # Look for end of headers
            if b"\r\n\r\n" not in self.request_buffer:
                return False
                
            # Parse headers
            headers_end = self.request_buffer.index(b"\r\n\r\n") + 4
            headers = self.request_buffer[:headers_end].decode('utf-8')
            
            # Parse request line
            request_line = headers.split('\r\n')[0]
            method, path, version = request_line.split(' ')
            
            # Store current request info
            self.current_request = {
                "method": method,
                "path": path,
                "version": version,
                "headers": headers,
                "headers_end": headers_end
            }
            
            # Check for content-length
            content_length = 0
            for line in headers.split('\r\n')[1:]:
                if line.lower().startswith('content-length:'):
                    content_length = int(line.split(':')[1].strip())
                    break
                    
            # Check if we have complete body
            return len(self.request_buffer) >= headers_end + content_length
            
        except Exception as e:
            logger.error(f"Error parsing request: {e}")
            return False
            
    def _handle_request(self) -> None:
        """Handle complete HTTP request."""
        try:
            # Parse URL
            url = urlparse(self.current_request["path"])
            host = url.netloc or url.path
            
            # Update connection info
            self.connection_manager.update_connection(self.connection_id, "host", host)
            self.connection_manager.update_connection(
                self.connection_id,
                "request",
                {
                    "method": self.current_request["method"],
                    "path": self.current_request["path"],
                    "version": self.current_request["version"]
                }
            )
            
            # Create remote connection if needed
            if not self.remote_transport:
                self._create_remote_connection(host)
                
            # Forward request
            if self.remote_transport and not self.remote_transport.is_closing():
                self.remote_transport.write(self.request_buffer)
                self.connection_manager.record_event(
                    self.connection_id,
                    "request",
                    "client-proxy",
                    "success",
                    len(self.request_buffer)
                )
                
            # Clear buffer
            self.request_buffer.clear()
            self.current_request.clear()
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            self.connection_manager.update_connection(self.connection_id, "error", str(e))
            if self.transport:
                self.transport.close()
                
    def _create_remote_connection(self, host: str) -> None:
        """Create connection to remote server."""
        # TODO: Implement remote connection establishment
        # This will involve:
        # 1. Resolving host
        # 2. Creating transport
        # 3. Setting up forwarding between client and remote
        pass
    
    async def _ensure_interceptor(self) -> None:
        """Ensure interceptor is initialized."""
        if self._interceptor is None:
            self._interceptor = ProxyInterceptor()
            await self._interceptor.initialize()
    
    async def handle_client_data(self, data: bytes) -> Optional[bytes]:
        """Handle data received from the client."""
        try:
            self._h11_client.receive_data(data)
            while True:
                event = self._h11_client.next_event()
                if event is h11.NEED_DATA:
                    break
                    
                if isinstance(event, h11.Request):
                    await self._start_request(event)
                elif isinstance(event, h11.Data):
                    self._request_buffer.extend(event.data)
                elif isinstance(event, h11.EndOfMessage):
                    await self._complete_request()
                    
            return None
            
        except Exception as e:
            logger.error(f"Error handling client data: {e}")
            await self.connection_manager.record_event(
                self.connection_id, 
                "request", 
                "client-proxy", 
                "error",
                len(data)
            )
            raise
    
    async def handle_server_data(self, data: bytes) -> Optional[bytes]:
        """Handle data received from the server."""
        try:
            self._h11_server.receive_data(data)
            while True:
                event = self._h11_server.next_event()
                if event is h11.NEED_DATA:
                    break
                    
                if isinstance(event, h11.Response):
                    await self._start_response(event)
                elif isinstance(event, h11.Data):
                    self._response_buffer.extend(event.data)
                elif isinstance(event, h11.EndOfMessage):
                    await self._complete_response()
                    
            return None
            
        except Exception as e:
            logger.error(f"Error handling server data: {e}")
            await self.connection_manager.record_event(
                self.connection_id,
                "response",
                "server-proxy",
                "error",
                len(data)
            )
            raise
    
    async def _start_request(self, event: h11.Request) -> None:
        """Start processing a new request."""
        self._current_request = InterceptedRequest(
            method=event.method.decode(),
            url=event.target.decode(),
            headers={k.decode(): v.decode() for k, v in event.headers},
            body=bytearray()
        )
        self._request_buffer = bytearray()
        self._request_complete = False
        
        await self.connection_manager.record_event(
            self.connection_id,
            "request",
            "client-proxy",
            "pending"
        )
    
    async def _complete_request(self) -> None:
        """Complete processing of the current request."""
        if not self._current_request:
            return
            
        self._current_request.body = bytes(self._request_buffer)
        self._request_complete = True
        
        await self._ensure_interceptor()
        if self._interceptor:
            try:
                await self._interceptor.handle_request(self._current_request)
            except Exception as e:
                logger.error(f"Error in request interceptor: {e}")
                
        await self.connection_manager.record_event(
            self.connection_id,
            "request",
            "client-proxy",
            "success",
            len(self._request_buffer)
        )
        
        # Store request in database
        async with AsyncSessionLocal() as session:
            entry = ProxyHistoryEntry(
                connection_id=self.connection_id,
                request_method=self._current_request.method,
                request_url=self._current_request.url,
                request_headers=self._current_request.headers,
                request_body=self._current_request.body,
                timestamp=datetime.utcnow()
            )
            session.add(entry)
            await session.commit()
    
    async def _start_response(self, event: h11.Response) -> None:
        """Start processing a new response."""
        self._current_response = InterceptedResponse(
            status_code=event.status_code,
            headers={k.decode(): v.decode() for k, v in event.headers},
            body=bytearray()
        )
        self._response_buffer = bytearray()
        self._response_complete = False
        
        await self.connection_manager.record_event(
            self.connection_id,
            "response",
            "server-proxy",
            "pending"
        )
    
    async def _complete_response(self) -> None:
        """Complete processing of the current response."""
        if not self._current_response:
            return
            
        self._current_response.body = bytes(self._response_buffer)
        self._response_complete = True
        
        await self._ensure_interceptor()
        if self._interceptor:
            try:
                await self._interceptor.handle_response(self._current_response)
            except Exception as e:
                logger.error(f"Error in response interceptor: {e}")
                
        await self.connection_manager.record_event(
            self.connection_id,
            "response",
            "server-proxy",
            "success",
            len(self._response_buffer)
        )
        
        # Update database entry with response
        if self._current_request:
            async with AsyncSessionLocal() as session:
                entry = await session.execute(
                    text("SELECT * FROM proxy_history WHERE connection_id = :conn_id ORDER BY timestamp DESC LIMIT 1"),
                    {"conn_id": self.connection_id}
                )
                entry = entry.first()
                if entry:
                    entry.response_status = self._current_response.status_code
                    entry.response_headers = self._current_response.headers
                    entry.response_body = self._current_response.body
                    entry.completed_at = datetime.utcnow()
                    await session.commit()
    
    def close(self) -> None:
        """Close the handler and cleanup resources."""
        if self._interceptor:
            asyncio.create_task(self._interceptor.cleanup())
        self._h11_client = h11.Connection(h11.SERVER)
        self._h11_server = h11.Connection(h11.CLIENT)
        self._current_request = None
        self._current_response = None
        self._request_buffer = bytearray()
        self._response_buffer = bytearray()
        self._request_complete = False
        self._response_complete = False
