"""Error handling for proxy connections."""
import asyncio
import socket
import ssl
from enum import Enum
from typing import Dict, Optional, Type

class ErrorCode(Enum):
    """Error codes for proxy errors."""
    UNKNOWN = 500
    CONNECTION_ERROR = 502
    TUNNEL_ERROR = 504
    SSL_ERROR = 525
    HANDSHAKE_ERROR = 526
    TIMEOUT_ERROR = 408

class ProxyError(Exception):
    """Base class for proxy errors."""
    
    def __init__(self, message: str, code: int = 500):
        self.message = message
        self.code = code
        super().__init__(message)
        
    def __str__(self) -> str:
        return f"{self.message} (code: {self.code})"

class ConnectionError(ProxyError):
    """Connection related errors."""
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.CONNECTION_ERROR.value)

class TunnelError(ProxyError):
    """Tunnel establishment errors."""
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.TUNNEL_ERROR.value)

class SSLError(ProxyError):
    """SSL/TLS related errors."""
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.SSL_ERROR.value)

class HandshakeError(ProxyError):
    """TLS handshake errors."""
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.HANDSHAKE_ERROR.value)

class TimeoutError(ProxyError):
    """Timeout related errors."""
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.TIMEOUT_ERROR.value)

def get_error_for_exception(exc: Exception) -> ProxyError:
    """Convert standard exceptions to proxy errors."""
    if isinstance(exc, ProxyError):
        return exc
        
    error_map = {
        ConnectionError: (ErrorCode.CONNECTION_ERROR, "Connection failed"),
        ssl.SSLError: (ErrorCode.SSL_ERROR, "SSL error occurred"),
        TimeoutError: (ErrorCode.TIMEOUT_ERROR, "Operation timed out"),
    }
    
    error_code, default_message = error_map.get(type(exc), (ErrorCode.UNKNOWN, "Unknown error occurred"))
    return ProxyError(message=str(exc) or default_message, code=error_code.value)

def format_error_response(error: ProxyError) -> bytes:
    """Format error as HTTP response."""
    status_line = f"HTTP/1.1 {error.code} {error.message}\r\n"
    headers = [
        "Content-Type: text/plain",
        f"Content-Length: {len(error.message)}",
        "Connection: close"
    ]
    return f"{status_line}{chr(13).join(headers)}\r\n\r\n{error.message}".encode()
