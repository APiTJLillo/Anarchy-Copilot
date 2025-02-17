"""Proxy module for intercepting and modifying HTTP traffic."""

from .config import ProxyConfig
from .core import ProxyServer
from .interceptor import (
    InterceptedRequest,
    InterceptedResponse,
    RequestInterceptor,
    ResponseInterceptor,
    ProxyInterceptor,
    JSONModifyInterceptor,
    SecurityHeadersInterceptor,
)
from .analysis.analyzer import TrafficAnalyzer  # Import from correct path
from .session import ProxySession

__all__ = [
    'ProxyConfig',
    'ProxyServer',
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
