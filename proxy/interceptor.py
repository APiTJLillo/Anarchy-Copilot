"""
Request and response interceptors for the Anarchy Copilot proxy module.

This module provides base classes for intercepting and modifying HTTP requests
and responses as they pass through the proxy.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from http import HTTPStatus
import json
import urllib.parse

@dataclass
class InterceptedRequest:
    """Represents an HTTP request intercepted by the proxy."""
    method: str
    url: str
    headers: Dict[str, str]
    body: Optional[bytes] = None
    
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

    async def intercept(self, req_or_resp: InterceptedRequest | InterceptedResponse, request: Optional[InterceptedRequest] = None) -> InterceptedRequest | InterceptedResponse:
        """Route to appropriate intercept method based on argument type."""
        if isinstance(req_or_resp, InterceptedRequest):
            return await self.intercept_request(req_or_resp)
        elif isinstance(req_or_resp, InterceptedResponse):
            if request is None:
                raise ValueError("Request parameter is required for response interception")
            return await self.intercept_response(req_or_resp, request)
        raise TypeError(f"Unexpected type: {type(req_or_resp)}")

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
