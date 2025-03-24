"""Proxy server package.

This package provides the core proxy server functionality, including:
- HTTP/HTTPS request handling
- Certificate management for HTTPS interception
- Logging and monitoring
- Request/response interception
"""

from .server import app as proxy_app
from .websocket import ws_manager, create_router

__version__ = '1.0.0'

__all__ = [
    'proxy_app',
    'ws_manager',
    'create_router',
    '__version__'
]

# Initialize package-level logger
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.info("Proxy package initialized successfully")
