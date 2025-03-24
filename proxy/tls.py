"""TLS functionality for the proxy.

This module re-exports TLS functionality from core.tls.
"""

from proxy.core.tls import (
    get_tls_context,
    CertificateManager,
    CertificateValidationError,
    RateLimitExceededError,
    CANotInitializedError,
    CertificateStats,
    CertificateHealth,
)

__all__ = [
    'get_tls_context',
    'CertificateManager',
    'CertificateValidationError',
    'RateLimitExceededError',
    'CANotInitializedError',
    'CertificateStats',
    'CertificateHealth',
] 