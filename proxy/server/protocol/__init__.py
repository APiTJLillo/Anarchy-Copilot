"""Protocol package for proxy server."""
# First import base types and interfaces
from .base_types import (
    TlsCapableProtocol,
    TlsContextProvider,
    TlsHandlerBase
)

# Then import core implementations
from .ssl_transport import SslTransport
from .base import BaseProxyProtocol
from .tls_handler import TlsHandler

# Define what should be available at package level
__all__ = [
    # Base types and interfaces
    'TlsCapableProtocol',
    'TlsContextProvider',
    'TlsHandlerBase',
    
    # Core implementations
    'SslTransport',
    'BaseProxyProtocol',
    'TlsHandler',
]

# Import protocol implementations last to avoid circular imports
from .https_intercept import HttpsInterceptProtocol
__all__.append('HttpsInterceptProtocol')
