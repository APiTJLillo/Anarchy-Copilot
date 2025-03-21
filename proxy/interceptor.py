"""
Request and response interceptors for the Anarchy Copilot proxy module.

This module provides base classes for intercepting and modifying HTTP requests
and responses as they pass through the proxy.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from http import HTTPStatus
from uuid import uuid4
import json
import urllib.parse
import logging
from datetime import datetime
from sqlalchemy import select, desc

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from starlette.requests import Request
from starlette.responses import Response
from database import AsyncSessionLocal

@dataclass
class InterceptedRequest:
    """Represents an HTTP request intercepted by the proxy."""
    method: str
    url: str
    headers: Dict[str, str]
    body: Optional[bytes] = None
    id: str = field(default_factory=lambda: str(uuid4()))
    connection_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    timestamp: Optional[datetime] = None
    
    @property
    def parsed_url(self) -> urllib.parse.ParseResult:
        """Parse the URL into its components."""
        return urllib.parse.urlparse(self.url)
    
    @property
    def query_params(self) -> Dict[str, List[str]]:
        """Parse query parameters into a dictionary."""
        return urllib.parse.parse_qs(self.parsed_url.query)
    
    def set_header(self, name: str, value: str) -> None:
        """Set a header value, creating it if it doesn't exist."""
        self.headers[name] = value
    
    def get_header(self, name: str, default: Any = None) -> Any:
        """Get a header value."""
        return self.headers.get(name, default)
    
    def to_dict(self) -> dict:
        """Convert request to a dictionary format."""
        return {
            'method': self.method,
            'url': self.url,
            'headers': dict(self.headers),
            'body': self.body.decode('utf-8', errors='ignore') if self.body else None
        }

@dataclass
class InterceptedResponse:
    """Represents an HTTP response intercepted by the proxy."""
    status_code: int
    headers: Dict[str, str]
    body: Optional[bytes] = None
    id: str = field(default_factory=lambda: str(uuid4()))
    connection_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def status(self) -> HTTPStatus:
        """Get the HTTP status enum for the status code."""
        return HTTPStatus(self.status_code)
    
    def set_header(self, name: str, value: str) -> None:
        """Set a header value, creating it if it doesn't exist."""
        self.headers[name] = value
    
    def get_header(self, name: str, default: Any = None) -> Any:
        """Get a header value."""
        return self.headers.get(name, default)
    
    def to_dict(self) -> dict:
        """Convert response to a dictionary format."""
        return {
            'status_code': self.status_code,
            'status': self.status.phrase,
            'headers': dict(self.headers),
            'body': self.body.decode('utf-8', errors='ignore') if self.body else None
        }

class RequestInterceptor(ABC):
    """Base class for request interceptors."""
    
    @abstractmethod
    async def intercept(self, request: InterceptedRequest) -> InterceptedRequest:
        """Process an intercepted request.
        
        Args:
            request: The intercepted HTTP request
            
        Returns:
            The modified request
            
        This method should modify the request in place if needed and return it.
        """
        pass

class ResponseInterceptor(ABC):
    """Base class for response interceptors."""
    
    @abstractmethod
    async def intercept(self, response: InterceptedResponse, request: InterceptedRequest) -> InterceptedResponse:
        """Process an intercepted response.
        
        Args:
            response: The intercepted HTTP response
            request: The original request that generated this response
            
        Returns:
            The modified response
            
        This method should modify the response in place if needed and return it.
        """
        pass

class ProxyInterceptor(RequestInterceptor, ResponseInterceptor):
    """Base class for interceptors that can handle both requests and responses."""
    
    # Registry of interceptor classes
    _registry = set()

    def __init_subclass__(cls, **kwargs):
        """Register interceptor subclasses."""
        super().__init_subclass__(**kwargs)
        ProxyInterceptor._registry.add(cls)

    def __init__(self, connection_id: Optional[str] = None):
        """Initialize interceptor.
        
        Args:
            connection_id: Optional unique ID for the connection. Only required for per-connection interceptors.
        """
        self.connection_id = connection_id or str(uuid4())  # Generate a default ID if none provided

    async def _ensure_db(self) -> None:
        """Ensure database connection is ready."""
        pass

    async def _get_active_session(self) -> Optional[int]:
        """Get active session ID."""
        return None

    async def intercept(self, request: Union[InterceptedRequest, InterceptedResponse], 
                       original_request: Optional[InterceptedRequest] = None) -> None:
        """Intercept and process request/response.
        
        Args:
            request: The request or response to intercept
            original_request: The original request if intercepting a response
        """
        pass

    async def close(self) -> None:
        """Clean up resources."""
        pass

    async def intercept_request(self, request: InterceptedRequest) -> InterceptedRequest:
        """Process an intercepted request.
        
        Override this method to implement request interception.
        """
        return request

    async def intercept_response(self, response: InterceptedResponse, request: InterceptedRequest) -> InterceptedResponse:
        """Process an intercepted response.
        
        Override this method to implement response interception.
        """
        return response

class JSONModifyInterceptor(RequestInterceptor):
    """Example interceptor that modifies JSON request bodies."""
    
    def __init__(self, modifications: Dict[str, Any]):
        """Initialize with a dictionary of modifications to apply."""
        self.modifications = modifications
    
    async def intercept(self, request: InterceptedRequest) -> InterceptedRequest:
        """Modify JSON request bodies by applying the configured changes."""
        content_type = request.get_header('Content-Type', '')
        if 'application/json' in content_type and request.body:
            try:
                data = json.loads(request.body)
                data.update(self.modifications)
                request.body = json.dumps(data).encode('utf-8')
                request.headers['Content-Length'] = str(len(request.body))
            except json.JSONDecodeError:
                pass
        return request

class SecurityHeadersInterceptor(ResponseInterceptor):
    """Example interceptor that adds security headers to responses."""
    
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'"
    }
    
    async def intercept(self, response: InterceptedResponse, request: InterceptedRequest) -> InterceptedResponse:
        """Add security headers to the response."""
        for header, value in self.SECURITY_HEADERS.items():
            if header not in response.headers:
                response.set_header(header, value)
        return response
