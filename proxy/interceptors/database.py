"""Database interceptor for storing proxy traffic."""
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any, Union
from sqlalchemy import text

from proxy.interceptor import ProxyInterceptor, InterceptedRequest, InterceptedResponse
from database import AsyncSessionLocal
from proxy.session import get_active_sessions

# Configure detailed logging
logger = logging.getLogger("proxy.core")
logger.setLevel(logging.DEBUG)

class DatabaseInterceptor(ProxyInterceptor):
    """Interceptor for storing proxy traffic in database."""
    
    def __init__(self, connection_id: str):
        """Initialize the database interceptor."""
        super().__init__()
        self._connection_id = connection_id
        logger.info(f"[{connection_id}] Initializing DatabaseInterceptor")
        self._current_session_id = None
        self._last_request_id = None
        self._pending_requests = {}  # Store request entries by connection_id
        self._db = None
        self._http_buffer = bytearray()
        self._is_parsing_http = False
        self._current_request = None

    async def _ensure_db(self):
        """Ensure database session is initialized."""
        if not self._db:
            logger.debug(f"[{self._connection_id}] Creating new database session")
            self._db = AsyncSessionLocal()
            # Test the connection
            try:
                await self._db.execute(text("SELECT 1"))
                logger.debug(f"[{self._connection_id}] Database connection test successful")
            except Exception as e:
                logger.error(f"[{self._connection_id}] Database connection test failed: {e}")
                self._db = None
                raise
        return self._db

    async def _get_active_session(self) -> Optional[int]:
        """Get the current active session ID."""
        if not self._current_session_id:
            sessions = await get_active_sessions()
            if sessions:
                self._current_session_id = sessions[0]['id']
                logger.debug(f"[{self._connection_id}] Found active session {self._current_session_id}")
            else:
                logger.warning(f"[{self._connection_id}] No active session found")
        return self._current_session_id

    async def store_raw_data(self, method: str, data: bytes) -> None:
        """Store raw data transfer in the database."""
        try:
            # First try to parse as HTTP
            if not self._is_parsing_http:
                self._http_buffer.extend(data)
                if self._try_parse_http():
                    self._is_parsing_http = True
                    logger.debug(f"[{self._connection_id}] Successfully parsed HTTP message")
                    return

            # If not HTTP or already tried parsing, store as raw
            logger.debug(f"[{self._connection_id}] Storing raw data: {len(data)} bytes, method: {method}")
            session_id = await self._get_active_session()
            if not session_id:
                logger.warning(f"[{self._connection_id}] No active session found for storing raw data")
                return

            db = await self._ensure_db()
            logger.debug(f"[{self._connection_id}] Creating new history entry")
            entry = {
                "session_id": session_id,
                "timestamp": datetime.utcnow(),
                "method": method,
                "url": f"raw://data",
                "request_headers": json.dumps({}),  # Empty dict for raw data
                "request_body": data.decode('utf-8', errors='ignore'),
                "tags": json.dumps(["raw"]),
                "is_intercepted": True
            }
            logger.debug(f"[{self._connection_id}] Executing INSERT query for raw data")
            result = await db.execute(
                text("""
                    INSERT INTO proxy_history (
                        session_id, timestamp, method, url, 
                        request_headers, request_body, tags, is_intercepted
                    ) VALUES (
                        :session_id, :timestamp, :method, :url,
                        :request_headers, :request_body, :tags, :is_intercepted
                    ) RETURNING id
                """),
                entry
            )
            
            # Get the row before committing
            row = result.fetchone()
            if row:
                self._last_request_id = row[0]
                logger.debug(f"[{self._connection_id}] Successfully stored raw data entry with ID {self._last_request_id}")
            else:
                logger.error(f"[{self._connection_id}] Failed to get ID for raw data entry")
                
            # Now commit the transaction
            logger.debug(f"[{self._connection_id}] Committing transaction")
            await db.commit()
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error storing raw data: {e}", exc_info=True)

    def _try_parse_http(self) -> bool:
        """Try to parse buffered data as HTTP message."""
        try:
            # Check for HTTP request line
            if b'\r\n' not in self._http_buffer:
                return False

            first_line = self._http_buffer.split(b'\r\n', 1)[0].decode('utf-8', errors='ignore')
            
            # Check if it's a valid HTTP request or response
            if ' ' not in first_line:
                return False

            if first_line.startswith(('GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH', 'CONNECT')):
                return True
            elif first_line.startswith('HTTP/'):
                return True

            return False
        except Exception:
            return False

    async def intercept_request(self, request: InterceptedRequest) -> InterceptedRequest:
        """Store intercepted request in the database."""
        try:
            logger.debug(f"[{self._connection_id}] Intercepting request: {request.method} {request.url}")
            session_id = await self._get_active_session()
            if not session_id:
                logger.warning(f"[{self._connection_id}] No active session found for request interception")
                return request

            db = await self._ensure_db()
            logger.debug(f"[{self._connection_id}] Creating new history entry for request")
            entry = {
                "session_id": session_id,
                "timestamp": datetime.utcnow(),
                "method": request.method,
                "url": request.url,
                "host": request.parsed_url.netloc,
                "path": request.parsed_url.path,
                "request_headers": json.dumps(dict(request.headers) if request.headers else {}),
                "request_body": request.body.decode('utf-8', errors='ignore') if request.body else None,
                "tags": json.dumps(["request"]),
                "is_intercepted": True
            }
            logger.debug(f"[{self._connection_id}] Executing INSERT query for request")
            result = await db.execute(
                text("""
                    INSERT INTO proxy_history (
                        session_id, timestamp, method, url, host, path,
                        request_headers, request_body, tags, is_intercepted
                    ) VALUES (
                        :session_id, :timestamp, :method, :url, :host, :path,
                        :request_headers, :request_body, :tags, :is_intercepted
                    ) RETURNING id
                """),
                entry
            )
            
            # Get the row before committing
            row = result.fetchone()
            if row:
                self._last_request_id = row[0]
                self._pending_requests[self._connection_id] = entry
                self._current_request = request  # Store for matching with response
                logger.debug(f"[{self._connection_id}] Successfully stored request entry with ID {self._last_request_id}")
            else:
                logger.error(f"[{self._connection_id}] Failed to get ID for stored request")
                
            # Now commit the transaction
            logger.debug(f"[{self._connection_id}] Committing transaction")
            await db.commit()
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error intercepting request: {e}", exc_info=True)
            # Don't block the request even if storage fails
        return request

    async def intercept_response(self, response: InterceptedResponse, request: Optional[InterceptedRequest] = None) -> InterceptedResponse:
        """Store response in database."""
        try:
            # Use provided request or stored current request
            request_to_use = request or self._current_request
            
            if self._last_request_id:
                db = await self._ensure_db()
                logger.debug(f"[{self._connection_id}] Updating history entry {self._last_request_id} with response")
                
                # Calculate duration if request is available
                duration = None
                if request_to_use and hasattr(request_to_use, 'timestamp') and request_to_use.timestamp:
                    duration = (datetime.utcnow() - request_to_use.timestamp).total_seconds()
                
                # Verify response data before update
                update_data = {
                    "id": self._last_request_id,
                    "status_code": response.status_code,
                    "response_headers": json.dumps(dict(response.headers)),
                    "response_body": response.body.decode('utf-8', errors='ignore') if response.body else None,
                    "duration": duration
                }
                
                # Verify the request entry still exists
                logger.debug(f"[{self._connection_id}] Verifying request entry exists")
                verify = await db.execute(
                    text("SELECT id FROM proxy_history WHERE id = :id"),
                    {"id": self._last_request_id}
                )
                if not verify.fetchone():
                    logger.error(f"[{self._connection_id}] Request entry {self._last_request_id} not found")
                    return response
                
                logger.debug(f"[{self._connection_id}] Executing UPDATE query for response")
                await db.execute(
                    text("""
                        UPDATE proxy_history 
                        SET status_code = :status_code,
                            response_headers = :response_headers,
                            response_body = :response_body,
                            duration = :duration
                        WHERE id = :id
                    """),
                    update_data
                )
                logger.debug(f"[{self._connection_id}] Committing transaction")
                await db.commit()
                logger.debug(f"[{self._connection_id}] Successfully updated entry with response")
                
                # Clean up after successful response storage
                self._pending_requests.pop(self._connection_id, None)
                self._current_request = None
                self._is_parsing_http = False
                self._http_buffer.clear()
            else:
                logger.warning(f"[{self._connection_id}] No request ID found for response")
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error intercepting response: {e}", exc_info=True)
            
        return response

    async def intercept(self, data: Union[InterceptedRequest, InterceptedResponse], request: Optional[InterceptedRequest] = None) -> Union[InterceptedRequest, InterceptedResponse]:
        """Route to appropriate intercept method based on type."""
        logger.debug(f"[{self._connection_id}] Intercepting data of type {type(data)}")
        try:
            if isinstance(data, InterceptedRequest):
                # Add timestamp for duration calculation
                data.timestamp = datetime.utcnow()
                return await self.intercept_request(data)
            elif isinstance(data, InterceptedResponse):
                return await self.intercept_response(data, request or self._current_request)
            raise TypeError(f"Unexpected type: {type(data)}")
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error in intercept: {e}", exc_info=True)
            return data

    async def close(self) -> None:
        """Clean up any resources."""
        logger.debug(f"[{self._connection_id}] Closing database interceptor")
        if self._db:
            await self._db.close()
            self._db = None
        self._pending_requests.clear()
        self._current_request = None
        self._is_parsing_http = False
        self._http_buffer.clear()
