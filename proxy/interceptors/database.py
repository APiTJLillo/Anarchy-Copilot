"""Database interceptor for storing proxy traffic."""
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any, Union, Tuple
from sqlalchemy import text
import asyncio
import base64
import mimetypes
from urllib.parse import urlparse
import ssl
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes

from proxy.interceptor import ProxyInterceptor, InterceptedRequest, InterceptedResponse
from database import AsyncSessionLocal
from proxy.session import get_active_sessions
from proxy.core.tls import get_tls_context, CertificateManager

logger = logging.getLogger("proxy.core")

class DatabaseInterceptor(ProxyInterceptor):
    """Stores intercepted proxy traffic in database."""
    
    # Content types that should be stored as text
    TEXT_CONTENT_TYPES = {
        'text/',
        'application/json',
        'application/xml',
        'application/javascript',
        'application/x-www-form-urlencoded'
    }
    
    def __init__(self, connection_id: str):
        """Initialize database interceptor.
        
        Args:
            connection_id: Unique connection identifier
        """
        super().__init__()
        self._connection_id = connection_id
        self._session_id = None
        self._last_request_id = None
        self._current_request = None
        self._pending_requests: Dict[str, Dict[str, Any]] = {}
        self._http_buffer = bytearray()
        self._is_parsing_http = False
        self._http2_preface_seen = False
        self._lock = asyncio.Lock()  # Lock for synchronizing database operations
        self._tls_context = None  # Will be set later when needed
        logger.info(f"[{self._connection_id}] Initializing DatabaseInterceptor")

    def _is_text_content(self, headers: Dict[str, str]) -> bool:
        """Check if content should be stored as text based on Content-Type header."""
        content_type = headers.get('Content-Type', '').lower().split(';')[0]
        return any(content_type.startswith(text_type) for text_type in self.TEXT_CONTENT_TYPES)

    def _process_body(self, body: Optional[bytes], headers: Dict[str, str], is_encrypted: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """Process body data for storage, returning both raw and decrypted versions.
        
        Args:
            body: Raw body bytes
            headers: HTTP headers dictionary
            is_encrypted: Whether the body is TLS encrypted
            
        Returns:
            Tuple of (raw_body, decrypted_body) as strings
        """
        if not body:
            return None, None
            
        try:
            # Store raw data always in base64
            raw_body = base64.b64encode(body).decode('ascii')
            
            # Check if this looks like an HTTP message
            try:
                if body.startswith(b'HTTP/') or any(body.startswith(method) for method in [b'GET ', b'POST ', b'PUT ', b'DELETE ', b'HEAD ', b'OPTIONS ', b'PATCH ']):
                    # This is an HTTP message, try to decode as text
                    try:
                        return raw_body, body.decode('utf-8')
                    except UnicodeDecodeError:
                        return raw_body, raw_body
            except:
                pass
                
            # For text content, try to decode as UTF-8
            if self._is_text_content(headers):
                try:
                    decrypted = body.decode('utf-8')
                    return raw_body, decrypted
                except UnicodeDecodeError:
                    # If UTF-8 fails, try other common encodings
                    for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                        try:
                            decrypted = body.decode(encoding)
                            return raw_body, decrypted
                        except UnicodeDecodeError:
                            continue
                    # If all decodings fail, return base64
                    return raw_body, raw_body
            
            # For binary content, store as base64
            return raw_body, raw_body
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error processing body: {e}")
            return None, None

    async def _get_session_id(self) -> Optional[int]:
        """Get current session ID."""
        if not self._session_id:
            sessions = await get_active_sessions()
            if sessions:
                self._session_id = sessions[0]['id']
                logger.debug(f"[{self._connection_id}] Found active session {self._session_id}")
            else:
                logger.warning(f"[{self._connection_id}] No active session found")
        return self._session_id

    async def store_raw_data(self, method: str, data: bytes, is_encrypted: bool = False) -> None:
        """Store raw data transfer.
        
        Args:
            method: Data transfer method/direction
            data: Raw bytes to store
            is_encrypted: Whether the data is TLS encrypted
        """
        try:
            session_id = await self._get_session_id()
            if not session_id:
                return

            # Process both raw and decrypted data
            raw_body, decrypted_body = self._process_body(
                data,
                {},  # No headers for raw data
                is_encrypted
            )

            entry = {
                "session_id": session_id,
                "timestamp": datetime.utcnow(),
                "method": method,
                "url": "raw://data",
                "request_body": raw_body if method == 'client->target' else None,
                "decrypted_request": decrypted_body if method == 'client->target' else None,
                "response_body": raw_body if method == 'target->client' else None,
                "decrypted_response": decrypted_body if method == 'target->client' else None,
                "tags": json.dumps(["raw", "encrypted"] if is_encrypted else ["raw"]),
                "is_intercepted": True,
                "is_encrypted": is_encrypted
            }

            async with AsyncSessionLocal() as db:
                async with db.begin():
                    await db.execute(
                        text("""
                            INSERT INTO proxy_history (
                                session_id, timestamp, method, url,
                                request_body, decrypted_request,
                                response_body, decrypted_response,
                                tags, is_intercepted, is_encrypted
                            ) VALUES (
                                :session_id, :timestamp, :method, :url,
                                :request_body, :decrypted_request,
                                :response_body, :decrypted_response,
                                :tags, :is_intercepted, :is_encrypted
                            )
                        """),
                        entry
                    )

        except Exception as e:
            logger.error(f"[{self._connection_id}] Error storing raw data: {e}")

    async def _store_decrypted_request(self, request: InterceptedRequest) -> None:
        """Store decrypted HTTP request."""
        try:
            session_id = await self._get_session_id()
            if not session_id:
                return

            is_encrypted = request.url.startswith("https://")
            raw_body, decrypted_body = self._process_body(
                request.body,
                request.headers,
                is_encrypted
            )

            entry = {
                "session_id": session_id,
                "timestamp": datetime.utcnow(),
                "method": request.method,
                "url": request.url,
                "host": request.parsed_url.netloc,
                "path": request.parsed_url.path,
                "request_headers": json.dumps(dict(request.headers)),
                "request_body": raw_body,
                "decrypted_request": decrypted_body,
                "tags": json.dumps(["request", "encrypted"] if is_encrypted else ["request"]),
                "is_intercepted": True,
                "is_encrypted": is_encrypted,
                "tls_version": request.tls_version if hasattr(request, 'tls_version') else None,
                "cipher_suite": request.cipher_suite if hasattr(request, 'cipher_suite') else None,
                "certificate_info": json.dumps(request.certificate_info) if hasattr(request, 'certificate_info') else None
            }

            async with AsyncSessionLocal() as db:
                async with db.begin():
                    result = await db.execute(
                        text("""
                            INSERT INTO proxy_history (
                                session_id, timestamp, method, url, host, path,
                                request_headers, request_body, decrypted_request,
                                tags, is_intercepted, is_encrypted,
                                tls_version, cipher_suite, certificate_info
                            ) VALUES (
                                :session_id, :timestamp, :method, :url, :host, :path,
                                :request_headers, :request_body, :decrypted_request,
                                :tags, :is_intercepted, :is_encrypted,
                                :tls_version, :cipher_suite, :certificate_info
                            ) RETURNING id
                        """),
                        entry
                    )
                    row = result.fetchone()
                    if row:
                        self._last_request_id = row[0]
                        self._pending_requests[self._connection_id] = entry
                        self._current_request = request
                        logger.debug(f"[{self._connection_id}] Stored request with ID {self._last_request_id}")

        except Exception as e:
            logger.error(f"[{self._connection_id}] Error storing request: {e}")

    async def _store_decrypted_response(self, response: InterceptedResponse, request: Optional[InterceptedRequest] = None) -> None:
        """Store decrypted HTTP response."""
        try:
            if not self._last_request_id:
                return

            # Use provided request or current
            req = request or self._current_request
            if not req:
                return

            is_encrypted = req.url.startswith("https://")
            raw_body, decrypted_body = self._process_body(
                response.body,
                response.headers,
                is_encrypted
            )

            update_data = {
                "id": self._last_request_id,
                "status_code": response.status_code,
                "response_headers": json.dumps(dict(response.headers)),
                "response_body": raw_body,
                "decrypted_response": decrypted_body,
                "tags": json.dumps(["request", "response", "encrypted"] if is_encrypted else ["request", "response"])
            }

            async with AsyncSessionLocal() as db:
                async with db.begin():
                    await db.execute(
                        text("""
                            UPDATE proxy_history 
                            SET status_code = :status_code,
                                response_headers = :response_headers,
                                response_body = :response_body,
                                decrypted_response = :decrypted_response,
                                tags = :tags
                            WHERE id = :id
                        """),
                        update_data
                    )
                    logger.debug(f"[{self._connection_id}] Updated history entry {self._last_request_id} with response")

                    # Clean up after successful update
                    self._pending_requests.pop(self._connection_id, None)
                    self._current_request = None

        except Exception as e:
            logger.error(f"[{self._connection_id}] Error storing response: {e}")

    async def _intercept_request(self, request: InterceptedRequest) -> InterceptedRequest:
        """Store request in database."""
        try:
            is_encrypted = request.url.startswith("https://")
            # Store raw data with encryption flag
            if hasattr(request, 'raw_data'):
                await self.store_raw_data('client->target', request.raw_data, is_encrypted)
            # Store processed request
            await self._store_decrypted_request(request)
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error in request interception: {e}")

        return request

    async def _intercept_response(self, response: InterceptedResponse, request: Optional[InterceptedRequest] = None) -> InterceptedResponse:
        """Store response in database."""
        if not isinstance(response, InterceptedResponse):
            return response

        try:
            is_encrypted = request.url.startswith("https://") if request else False
            # Store raw data with encryption flag
            if hasattr(response, 'raw_data'):
                await self.store_raw_data('target->client', response.raw_data, is_encrypted)
            # Store processed response
            await self._store_decrypted_response(response, request)
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error in response interception: {e}")

        return response

    async def close(self) -> None:
        """Clean up database resources."""
        logger.debug(f"[{self._connection_id}] Closing database interceptor")
        self._pending_requests.clear()
        self._current_request = None
        self._http_buffer.clear()

    async def store_request(self, data: bytes) -> None:
        """Store decrypted request data."""
        try:
            session_id = await self._get_session_id()
            if not session_id:
                return

            # Try to parse HTTP request
            try:
                request_lines = data.split(b'\r\n')
                request_line = request_lines[0].decode('utf-8')
                method, path, version = request_line.split(' ')
                
                # Parse headers
                headers = {}
                i = 1
                while i < len(request_lines):
                    line = request_lines[i].decode('utf-8').strip()
                    if not line:
                        break
                    name, value = line.split(':', 1)
                    headers[name.strip()] = value.strip()
                    i += 1
                    
                # Get body if present
                body = b'\r\n'.join(request_lines[i+1:]) if i+1 < len(request_lines) else None
                
                # Process body
                raw_body, decrypted_body = self._process_body(data, headers, True)
                
                # Store request
                entry = {
                    "session_id": session_id,
                    "timestamp": datetime.utcnow(),
                    "method": method,
                    "url": path,
                    "request_headers": json.dumps(headers),
                    "request_body": raw_body,
                    "decrypted_request": decrypted_body,
                    "tags": json.dumps(["request", "decrypted"]),
                    "is_intercepted": True,
                    "is_encrypted": True
                }

                async with AsyncSessionLocal() as db:
                    async with db.begin():
                        result = await db.execute(
                            text("""
                                INSERT INTO proxy_history (
                                    session_id, timestamp, method, url,
                                    request_headers, request_body, decrypted_request,
                                    tags, is_intercepted, is_encrypted
                                ) VALUES (
                                    :session_id, :timestamp, :method, :url,
                                    :request_headers, :request_body, :decrypted_request,
                                    :tags, :is_intercepted, :is_encrypted
                                ) RETURNING id
                            """),
                            entry
                        )
                        row = result.fetchone()
                        if row:
                            self._last_request_id = row[0]
                            logger.debug(f"[{self._connection_id}] Stored decrypted request with ID {self._last_request_id}")
                            
            except Exception as e:
                logger.error(f"[{self._connection_id}] Failed to parse HTTP request: {e}")
                # Store as raw data
                await self.store_raw_data("client->target", data, True)
                
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error storing request: {e}")

    async def store_response(self, data: bytes) -> None:
        """Store decrypted response data."""
        try:
            if not self._last_request_id:
                logger.warning(f"[{self._connection_id}] No request ID for response")
                await self.store_raw_data("target->client", data, True)
                return

            # Try to parse HTTP response
            try:
                response_lines = data.split(b'\r\n')
                status_line = response_lines[0].decode('utf-8')
                version, status_code, *reason = status_line.split(' ')
                status_code = int(status_code)
                
                # Parse headers
                headers = {}
                i = 1
                while i < len(response_lines):
                    line = response_lines[i].decode('utf-8').strip()
                    if not line:
                        break
                    name, value = line.split(':', 1)
                    headers[name.strip()] = value.strip()
                    i += 1
                    
                # Get body if present
                body = b'\r\n'.join(response_lines[i+1:]) if i+1 < len(response_lines) else None
                
                # Process body
                raw_body, decrypted_body = self._process_body(data, headers, True)

                # Update history entry
                update_data = {
                    "id": self._last_request_id,
                    "status_code": status_code,
                    "response_headers": json.dumps(headers),
                    "response_body": raw_body,
                    "decrypted_response": decrypted_body,
                    "tags": json.dumps(["request", "response", "decrypted"])
                }

                async with AsyncSessionLocal() as db:
                    async with db.begin():
                        await db.execute(
                            text("""
                                UPDATE proxy_history 
                                SET status_code = :status_code,
                                    response_headers = :response_headers,
                                    response_body = :response_body,
                                    decrypted_response = :decrypted_response,
                                    tags = :tags
                                WHERE id = :id
                            """),
                            update_data
                        )
                        logger.debug(f"[{self._connection_id}] Updated history entry {self._last_request_id} with decrypted response")
                        
            except Exception as e:
                logger.error(f"[{self._connection_id}] Failed to parse HTTP response: {e}")
                # Store as raw data
                await self.store_raw_data("target->client", data, True)
                
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error storing response: {e}")

    async def store_tls_info(self, side: str, info: dict) -> None:
        """Store TLS connection information."""
        try:
            session_id = await self._get_session_id()
            if not session_id:
                return

            async with AsyncSessionLocal() as db:
                async with db.begin():
                    await db.execute(
                        text("""
                            INSERT INTO proxy_tls_info (
                                session_id,
                                connection_id,
                                timestamp,
                                side,
                                version,
                                cipher_suite,
                                host
                            ) VALUES (
                                :session_id,
                                :connection_id,
                                :timestamp,
                                :side,
                                :version,
                                :cipher,
                                :host
                            )
                        """),
                        {
                            "session_id": session_id,
                            "connection_id": self._connection_id,
                            "timestamp": datetime.utcnow(),
                            "side": side,
                            "version": info.get("version"),
                            "cipher": info.get("cipher"),
                            "host": info.get("host")
                        }
                    )
                    logger.debug(f"[{self._connection_id}] Stored TLS info for {side} side")

        except Exception as e:
            logger.error(f"[{self._connection_id}] Error storing TLS info: {e}")
