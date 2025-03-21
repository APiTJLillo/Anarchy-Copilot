"""HTTP request/response handling for HTTPS interception."""
import h11
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
import asyncio
from async_timeout import timeout
from sqlalchemy import text

from database import AsyncSessionLocal
from api.proxy.database_models import ProxyHistoryEntry
from proxy.core.tls import get_tls_context, CertificateManager
from ..tls.connection_manager import connection_mgr
from proxy.interceptor import InterceptedRequest, InterceptedResponse, ProxyInterceptor

logger = logging.getLogger("proxy.core")

class ProxyResponse:
    """Represents an HTTP response from the proxy server."""
    
    def __init__(self, status_code: int = 200, headers: Dict[str, str] = None, body: bytes = None):
        self.status_code = status_code
        self.headers = headers or {}
        self.body = body or b""
        
    def to_h11(self) -> Tuple[h11.Response, h11.Data, h11.EndOfMessage]:
        """Convert to h11 events for sending."""
        headers = [(k.encode(), v.encode()) for k, v in self.headers.items()]
        if self.body and b"content-length" not in [k.lower() for k, _ in headers]:
            headers.append((b"content-length", str(len(self.body)).encode()))
            
        response = h11.Response(status_code=self.status_code, headers=headers)
        data = h11.Data(data=self.body) if self.body else None
        endofmessage = h11.EndOfMessage()
        
        if data:
            return response, data, endofmessage
        return response, endofmessage

class HttpRequestHandler:
    """Handles HTTP requests and responses during HTTPS interception."""

    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.client_conn = h11.Connection(h11.SERVER)  # For client->proxy
        self.server_conn = h11.Connection(h11.CLIENT)  # For proxy->server
        
        # Request/response state
        self._current_request: Optional[Dict[str, Any]] = None
        self._current_response: Optional[Dict[str, Any]] = None
        self._current_request_body = bytearray()
        self._current_response_body = bytearray()
        self._current_request_start_time: Optional[datetime] = None
        self._history_entry_id: Optional[int] = None
        self._interceptor = None  # Will be initialized in _ensure_interceptor
        self._current_intercepted_request: Optional[InterceptedRequest] = None

    async def _ensure_interceptor(self) -> None:
        """Ensure interceptor is initialized."""
        if not self._interceptor:
            logger.debug(f"[{self.connection_id}] Initializing interceptor")
            # Get the first registered interceptor
            interceptor_cls = next(iter(ProxyInterceptor._registry))
            self._interceptor = interceptor_cls(self.connection_id)
            # Test the interceptor
            try:
                await self._interceptor._ensure_db()
                session_id = await self._interceptor._get_active_session()
                if session_id:
                    logger.info(f"[{self.connection_id}] Interceptor initialized with session {session_id}")
                else:
                    logger.warning(f"[{self.connection_id}] No active session found for interceptor")
            except Exception as e:
                logger.error(f"[{self.connection_id}] Failed to initialize interceptor: {e}")
                self._interceptor = None
                raise

    async def handle_client_data(self, data: bytes) -> Optional[bytes]:
        """Process data received from the client."""
        try:
            self.client_conn.receive_data(data)
            events = []
            
            while True:
                event = self.client_conn.next_event()
                if event is h11.NEED_DATA:
                    break
                    
                if isinstance(event, h11.Request):
                    await self._start_request(event)
                elif isinstance(event, h11.Data) and self._current_request:
                    self._current_request_body.extend(event.data)
                elif isinstance(event, h11.EndOfMessage) and self._current_request:
                    await self._complete_request()
                elif event is h11.ConnectionClosed:
                    return None

            return self.server_conn.data_to_send()
                    
        except Exception as e:
            logger.error(f"[{self.connection_id}] Error handling client data: {e}")
            return None

    async def handle_server_data(self, data: bytes) -> Optional[bytes]:
        """Process data received from the server."""
        try:
            self.server_conn.receive_data(data)
            
            while True:
                event = self.server_conn.next_event()
                if event is h11.NEED_DATA:
                    break
                    
                if isinstance(event, h11.Response):
                    await self._start_response(event)
                elif isinstance(event, h11.Data) and self._current_response:
                    self._current_response_body.extend(event.data)
                elif isinstance(event, h11.EndOfMessage) and self._current_response:
                    await self._complete_response()
                elif event is h11.ConnectionClosed:
                    return None

            return self.client_conn.data_to_send()
                    
        except Exception as e:
            logger.error(f"[{self.connection_id}] Error handling server data: {e}")
            return None

    async def _start_request(self, event: h11.Request) -> None:
        """Initialize a new request."""
        self._current_request_start_time = datetime.utcnow()
        self._current_request = {
            "method": event.method.decode(),
            "url": event.target.decode(),
            "request_headers": dict(event.headers),
            "request_body": bytearray(),
            "is_modified": False,
            "tls_info": get_tls_context(None)  # Will be updated later
        }
        self._current_request_body.clear()
        
        # Update connection metrics
        connection_mgr.update_connection(
            self.connection_id, 
            "requests_processed",
            connection_mgr._active_connections[self.connection_id]["requests_processed"] + 1
        )

    async def _complete_request(self) -> None:
        """Process a complete HTTP request."""
        if not self._current_request:
            return
            
        # Store the request body
        self._current_request["request_body"] = bytes(self._current_request_body)
        
        # Ensure interceptor is initialized
        try:
            await self._ensure_interceptor()
        except Exception as e:
            logger.error(f"[{self.connection_id}] Failed to ensure interceptor: {e}")
            return
        
        # Create intercepted request
        self._current_intercepted_request = InterceptedRequest(
            method=self._current_request["method"],
            url=self._current_request["url"],
            headers=self._current_request["request_headers"],
            body=self._current_request["request_body"],
            connection_id=self.connection_id
        )
        
        # Let interceptor process request
        try:
            if self._interceptor:
                await self._interceptor.intercept(self._current_intercepted_request)
                logger.debug(f"[{self.connection_id}] Successfully intercepted request")
            else:
                logger.warning(f"[{self.connection_id}] No interceptor available for request")
        except Exception as e:
            logger.error(f"[{self.connection_id}] Failed to intercept request: {e}", exc_info=True)

    async def _start_response(self, event: h11.Response) -> None:
        """Initialize a new response."""
        self._current_response = {
            "status_code": event.status_code,
            "headers": dict(event.headers),
            "body": bytearray(),
            "is_modified": False
        }
        self._current_response_body.clear()

    async def _complete_response(self) -> None:
        """Process a complete HTTP response."""
        if not self._current_response or not self._current_request:
            return

        # Store the response body
        self._current_response["body"] = bytes(self._current_response_body)
        
        # Create intercepted response
        if self._current_intercepted_request:
            intercepted_response = InterceptedResponse(
                status_code=self._current_response["status_code"],
                headers=self._current_response["headers"],
                body=self._current_response["body"],
                connection_id=self.connection_id
            )
            
            # Let interceptor process response
            try:
                if self._interceptor:
                    await self._interceptor.intercept(intercepted_response, self._current_intercepted_request)
                    logger.debug(f"[{self.connection_id}] Successfully intercepted response")
                else:
                    logger.warning(f"[{self.connection_id}] No interceptor available for response")
            except Exception as e:
                logger.error(f"[{self.connection_id}] Failed to intercept response: {e}", exc_info=True)
        
        # Clear state
        self._current_request = None
        self._current_response = None
        self._current_request_body.clear()
        self._current_response_body.clear()
        self._current_intercepted_request = None

    def close(self) -> None:
        """Clean up handler resources."""
        self._current_request = None
        self._current_response = None
        self._current_request_body.clear()
        self._current_response_body.clear()
        self._current_intercepted_request = None
        if self._interceptor:
            asyncio.create_task(self._interceptor.close())

    async def handle_request(self, request) -> None:
        """Handle a Request object directly."""
        try:
            # Ensure method and target are strings
            method = request.method.decode() if isinstance(request.method, bytes) else request.method
            target = request.target.decode() if isinstance(request.target, bytes) else request.target
            
            # Create h11 Request event
            h11_request = h11.Request(
                method=method.encode(),
                target=target.encode(),
                headers=[(k.encode(), v.encode()) for k, v in request.headers.items()]
            )
            
            # Process request
            await self._start_request(h11_request)
            
            # Add request body if present
            if request.body:
                body_data = request.body if isinstance(request.body, bytes) else request.body.encode()
                self._current_request_body.extend(body_data)
            
            # Complete request processing
            await self._complete_request()
            
            logger.debug(f"[{self.connection_id}] Successfully handled request: {method} {target}")
            
        except Exception as e:
            logger.error(f"[{self.connection_id}] Error handling request: {e}")
            raise
