"""HTTP request/response handling for HTTPS interception."""
import h11
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
import asyncio
from async_timeout import timeout as async_timeout
from sqlalchemy import text

from database import AsyncSessionLocal
from api.proxy.database_models import ProxyHistoryEntry
from proxy.server.tls.context import TlsContextFactory
from ..tls.connection_manager import connection_mgr

logger = logging.getLogger("proxy.core")

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
            "tls_info": TlsContextFactory.get_connection_info(None)  # Will be updated later
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
        
        # Create initial history entry
        await self._create_history_entry()

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
        
        # Store transaction in database
        await self._store_transaction()
        
        # Clear state
        self._current_request = None
        self._current_response = None
        self._current_request_body.clear()
        self._current_response_body.clear()

    async def _create_history_entry(self) -> None:
        """Create initial history entry for request."""
        try:
            async with AsyncSessionLocal() as db:
                # Get active session
                result = await db.execute(
                    text("SELECT * FROM proxy_sessions WHERE is_active = true ORDER BY start_time DESC LIMIT 1")
                )
                active_session = result.first()
                if active_session:
                    entry = ProxyHistoryEntry(
                        session_id=active_session.id,
                        timestamp=self._current_request_start_time,
                        method=self._current_request["method"],
                        url=self._current_request["url"],
                        request_headers=self._current_request["request_headers"],
                        request_body=self._current_request["request_body"].decode('utf-8', errors='ignore'),
                        tags=["HTTPS"]
                    )
                    db.add(entry)
                    await db.commit()
                    await db.refresh(entry)
                    self._history_entry_id = entry.id
                    logger.debug(f"[{self.connection_id}] Created history entry {entry.id} for session {active_session.id}")
                else:
                    logger.warning(f"[{self.connection_id}] No active session found for history entry")
        except Exception as e:
            logger.error(f"[{self.connection_id}] Failed to create history entry: {e}")

    async def _store_transaction(self) -> None:
        """Store the complete HTTP transaction in the database."""
        if not self._current_request or not self._current_response:
            return

        try:
            async with AsyncSessionLocal() as db:
                if self._history_entry_id:
                    entry = await db.get(ProxyHistoryEntry, self._history_entry_id)
                    if entry:
                        end_time = datetime.utcnow()
                        duration = (end_time - self._current_request_start_time).total_seconds() \
                            if self._current_request_start_time else None

                        entry.response_status = self._current_response["status_code"]
                        entry.response_headers = self._current_response["headers"]
                        entry.response_body = self._current_response["body"].decode('utf-8', errors='ignore')
                        entry.duration = duration
                        entry.is_intercepted = self._current_request["is_modified"]
                        entry.tls_info = self._current_request.get("tls_info")
                        
                        await db.commit()
        except Exception as e:
            logger.error(f"[{self.connection_id}] Failed to store transaction: {e}")

    def close(self) -> None:
        """Clean up handler resources."""
        self._current_request = None
        self._current_response = None
        self._current_request_body.clear()
        self._current_response_body.clear()

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
