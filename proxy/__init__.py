"""Proxy module for intercepting and modifying HTTP traffic."""

from .config import ProxyConfig
from .interceptor import (
    InterceptedRequest,
    InterceptedResponse,
    RequestInterceptor,
    ResponseInterceptor,
    ProxyInterceptor,
    JSONModifyInterceptor,
    SecurityHeadersInterceptor,
)
from .analysis.analyzer import TrafficAnalyzer
from .session import ProxySession

# Import ProxyServer lazily to avoid circular imports
def get_proxy_server():
    """Get the ProxyServer class.
    
    Returns:
        ProxyServer class, imported lazily to avoid circular imports
    """
    from .core import ProxyServer
    return ProxyServer

__all__ = [
    'ProxyConfig',
    'get_proxy_server',
    'InterceptedRequest',
    'InterceptedResponse',
    'RequestInterceptor',
    'ResponseInterceptor',
    'ProxyInterceptor',
    'JSONModifyInterceptor',
    'SecurityHeadersInterceptor',
    'ProxySession',
    'TrafficAnalyzer',
]
