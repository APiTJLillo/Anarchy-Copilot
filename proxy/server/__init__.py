"""Proxy server package.

This package provides the core proxy server functionality, including:
- HTTP/HTTPS request handling
- Certificate management for HTTPS interception
- Logging and monitoring
- Request/response interception
"""

from .certificates import CertificateAuthority
from .handlers import proxy_middleware, ProxyResponse
from .logging_middleware import LoggingMiddleware, LoggingMiddlewareFactory
from .proxy_server import ProxyServer
from .tunneling import TunnelManager

__version__ = '1.0.0'

__all__ = [
    'CertificateAuthority',
    'LoggingMiddleware',
    'LoggingMiddlewareFactory',
    'ProxyServer',
    'TunnelManager',
    'proxy_middleware',
    'ProxyResponse',
    '__version__'
]

# Initialize package-level logger
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
