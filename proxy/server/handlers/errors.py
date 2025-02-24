"""Error handling for proxy connections."""
import asyncio
import socket
import ssl
from enum import Enum
from typing import Dict, Optional, Type

class ProxyErrorCode(Enum):
    """Error codes for proxy operations."""
    GENERAL_ERROR = 500
    BAD_GATEWAY = 502
    GATEWAY_TIMEOUT = 504
    CONNECTION_REFUSED = 503
    SSL_ERROR = 525
    HANDSHAKE_ERROR = 526
    TUNNEL_FAILED = 527
    PROTOCOL_ERROR = 528

class ProxyError(Exception):
    """Base exception for proxy errors."""
    def __init__(self, 
                 code: ProxyErrorCode,
                 message: str,
                 details: Optional[Dict] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)

class ConnectionError(ProxyError):
    """Error establishing connection."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(ProxyErrorCode.CONNECTION_REFUSED, message, details)

class TunnelError(ProxyError):
    """Error in tunnel operation."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(ProxyErrorCode.TUNNEL_FAILED, message, details)

class SSLError(ProxyError):
    """SSL/TLS related errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(ProxyErrorCode.SSL_ERROR, message, details)

class HandshakeError(ProxyError):
    """TLS handshake errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(ProxyErrorCode.HANDSHAKE_ERROR, message, details)

class TimeoutError(ProxyError):
    """Timeout errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(ProxyErrorCode.GATEWAY_TIMEOUT, message, details)

# Error response templates
ERROR_RESPONSES: Dict[ProxyErrorCode, str] = {
    ProxyErrorCode.GENERAL_ERROR: "Internal Server Error",
    ProxyErrorCode.BAD_GATEWAY: "Bad Gateway",
    ProxyErrorCode.GATEWAY_TIMEOUT: "Gateway Timeout",
    ProxyErrorCode.CONNECTION_REFUSED: "Service Unavailable",
    ProxyErrorCode.SSL_ERROR: "SSL Handshake Failed",
    ProxyErrorCode.HANDSHAKE_ERROR: "TLS Handshake Failed",
    ProxyErrorCode.TUNNEL_FAILED: "Tunnel Connection Failed",
    ProxyErrorCode.PROTOCOL_ERROR: "Protocol Error"
}

def format_error_response(error: ProxyError) -> bytes:
    """Format error as HTTP response."""
    status_text = ERROR_RESPONSES.get(error.code, "Unknown Error")
    message = f"{error.message}\n\nDetails: {error.details}" if error.details else error.message
    
    response = (
        f"HTTP/1.1 {error.code.value} {status_text}\r\n"
        f"Content-Type: text/plain\r\n"
        f"Content-Length: {len(message)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
        f"{message}"
    )
    return response.encode()

def get_error_for_exception(exc: Exception) -> ProxyError:
    """Convert standard exceptions to proxy errors."""
    error_mapping: Dict[Type[Exception], Type[ProxyError]] = {
        ConnectionRefusedError: ConnectionError,
        asyncio.TimeoutError: TimeoutError,
        ssl.SSLError: SSLError,
        ssl.SSLCertVerificationError: SSLError,
        ConnectionError: ConnectionError,
        socket.error: ConnectionError
    }

    error_class = error_mapping.get(type(exc), ProxyError)
    if error_class == ProxyError:
        return ProxyError(
            ProxyErrorCode.GENERAL_ERROR,
            str(exc),
            {"exception_type": exc.__class__.__name__}
        )
    
    return error_class(str(exc), {"exception_type": exc.__class__.__name__})
