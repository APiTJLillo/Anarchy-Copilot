"""Base HTTP handling methods for TLS handler."""
import asyncio
import logging
from typing import Dict, Optional, Protocol, Any, TYPE_CHECKING, TypeVar
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from proxy.interceptor import InterceptedRequest, InterceptedResponse

logger = logging.getLogger("proxy.core")

T = TypeVar('T', bound='BaseHandlerProtocol')

class BaseHandlerProtocol(Protocol):
    """Protocol for handler required attributes."""
    @property
    def _connection_id(self) -> str: ...
    
    @property 
    def _database_interceptor(self) -> Any: ...
    
    @property
    def _is_request(self) -> bool: ...
    
    @_is_request.setter
    def _is_request(self, value: bool) -> None: ...
    
    @property
    def _http2_preface_seen(self) -> bool: ...
    
    @property
    def _header_block(self) -> Optional[bytes]: ...
    
    @property
    def _current_request(self) -> Optional['InterceptedRequest']: ...
    
    @_current_request.setter
    def _current_request(self, value: Optional['InterceptedRequest']) -> None: ...
    
    @abstractmethod
    async def _ensure_interceptor(self) -> None:
        """Initialize the database interceptor."""
        ...

class BaseHttpHandler(ABC):
    """Base class providing HTTP handling methods."""

    _connection_id: str
    _database_interceptor: Any
    _is_request: bool
    _http2_preface_seen: bool
    _header_block: Optional[bytes]
    _current_request: Optional['InterceptedRequest']

    @abstractmethod
    async def _ensure_interceptor(self) -> None:
        """Initialize the database interceptor."""
        pass

    async def _handle_request(self, 
                          method: str, 
                          target: str, 
                          headers: Dict[str, str],
                          body: Optional[bytes], 
                          protocol: asyncio.Protocol) -> None:
        """Handle an HTTP request."""
        try:
            from proxy.interceptor import InterceptedRequest
            intercepted_request = InterceptedRequest(
                method=method,
                url=target,
                headers=headers,
                body=body if body is not None else b'',
                connection_id=self._connection_id
            )
            
            self._current_request = intercepted_request
            self._is_request = False
            
            await self._ensure_interceptor()
            if self._database_interceptor:
                await self._database_interceptor.intercept(intercepted_request)
                logger.info(f"[{self._connection_id}] Successfully intercepted HTTP request: {method} {target}")
                
            if not self._http2_preface_seen:
                message = bytearray()
                if self._header_block is not None:
                    message.extend(self._header_block)
                    message.extend(b'\r\n\r\n')
                    if body:
                        message.extend(body)
                    protocol.data_received(bytes(message))
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error handling request: {e}")

    async def _handle_response(self, 
                           status_code: int, 
                           headers: Dict[str, str],
                           body: Optional[bytes], 
                           protocol: asyncio.Protocol) -> None:
        """Handle an HTTP response."""
        try:
            from proxy.interceptor import InterceptedResponse
            intercepted_response = InterceptedResponse(
                status_code=status_code,
                headers=headers,
                body=body if body is not None else b'',
                connection_id=self._connection_id
            )
            
            await self._ensure_interceptor()
            if self._database_interceptor:
                await self._database_interceptor.intercept(intercepted_response, self._current_request)
                logger.info(f"[{self._connection_id}] Successfully intercepted HTTP response: {status_code}")
            
            self._current_request = None
            self._is_request = True
            
            if not self._http2_preface_seen:
                message = bytearray()
                if self._header_block is not None:
                    message.extend(self._header_block)
                    message.extend(b'\r\n\r\n')
                    if body:
                        message.extend(body)
                    protocol.data_received(bytes(message))
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error handling response: {e}")

    async def _handle_raw_data(self, data: bytes, protocol: asyncio.Protocol) -> None:
        """Handle raw (non-HTTP) data."""
        try:
            if not self._http2_preface_seen:
                await self._ensure_interceptor()
                if self._database_interceptor:
                    method = "client->target" if self._is_request else "target->client"
                    await self._database_interceptor.store_raw_data(method, data)
                    logger.debug(f"[{self._connection_id}] Stored raw data: {len(data)} bytes")
            protocol.data_received(data)
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error handling raw data: {e}")
