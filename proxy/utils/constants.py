"""Constants for proxy configuration."""
from typing import List, Dict, Any, cast
from .config_types import (
    NetworkConfigDict,
    SSLConfigDict,
    MemoryConfigDict,
    EnvVarsDict,
    HTTPStatusDict
)

class NetworkConfig:
    """Network configuration settings."""
    def __init__(self):
        self._config: NetworkConfigDict = {
            # Buffer sizes
            'BUFFER_SIZE': 8192,
            'SOCKET_BUFFER_SIZE': 131072,  # 128KB

            # Timeouts
            'CONNECT_TIMEOUT': 30.0,
            'SSL_HANDSHAKE_TIMEOUT': 30.0,
            'KEEPALIVE_TIME': 60.0,
            'INACTIVITY_TIMEOUT': 300.0,
            'CLIENT_HELLO_TIMEOUT': 10.0,
            'RESPONSE_WAIT': 0.1,
            'PIPE_TIMEOUT': 3600.0,
            'CLEANUP_TIMEOUT': 5.0,

            # Retry settings
            'MAX_RETRIES': 3,
            'MAX_ERRORS': 5,
            'RETRY_DELAY': 0.5,

            # Memory monitoring
            'MEMORY_SAMPLE_INTERVAL': 10.0,
            'MEMORY_GROWTH_THRESHOLD': 10 * 1024 * 1024,  # 10MB
            'MEMORY_CLEANUP_THRESHOLD': 100 * 1024 * 1024,  # 100MB
            'THROUGHPUT_LOG_INTERVAL': 60.0
        }

    def __getattr__(self, name: str) -> Any:
        """Get config value."""
        return self._config[name]

class SSLConfig:
    """SSL/TLS configuration settings."""
    def __init__(self):
        self._config: SSLConfigDict = {
            # TLS handshake settings
            'MAX_HANDSHAKE_ATTEMPTS': 3,
            'HANDSHAKE_RETRY_DELAY': 1.0,
            'MIN_HANDSHAKE_DELAY': 0.1,
            'MAX_HANDSHAKE_DELAY': 5.0,
            'TLS_RECORD_HEADER_SIZE': 5,

            # TLS protocol settings
            'ALPN_PROTOCOLS': ['h2', 'http/1.1'],
            'MIN_TLS_VERSION': 2,  # TLS 1.2
            'CIPHER_PREFERENCE': 'HIGH:!aNULL:!kRSA:!PSK:!SRP:!MD5:!RC4',
            
            # Fatal SSL errors that should terminate handshake attempts
            'FATAL_SSL_ERRORS': [
                "NO_PROTOCOLS_AVAILABLE",
                "CERTIFICATE_VERIFY_FAILED", 
                "UNKNOWN_PROTOCOL",
                "BAD_CERTIFICATE",
                "CERTIFICATE_UNKNOWN",
                "WRONG_VERSION_NUMBER",
                "SSLV3_ALERT_CERTIFICATE_UNKNOWN",
                "TLSV1_ALERT_PROTOCOL_VERSION",
                "TLSV1_ALERT_INTERNAL_ERROR",
                "UNEXPECTED_EOF_WHILE_READING"
            ]
        }

    def __getattr__(self, name: str) -> Any:
        """Get config value."""
        return self._config[name]

class MemoryConfig:
    """Memory management configuration."""
    def __init__(self):
        self._config: MemoryConfigDict = {
            'GC_THRESHOLD': 1000,              # Operations before forcing garbage collection
            'CONTEXT_CACHE_SIZE': 100,         # Maximum number of cached SSL contexts
            'HIGH_MEMORY_THRESHOLD_MB': 500    # Memory threshold for forced GC (500MB)
        }

    def __getattr__(self, name: str) -> Any:
        """Get config value."""
        return self._config[name]

class EnvVars:
    """Environment variable names."""
    def __init__(self):
        self._config: EnvVarsDict = {
            'HOST_ENV_VAR': 'ANARCHY_PROXY_HOST',
            'PORT_ENV_VAR': 'ANARCHY_PROXY_PORT',
            'CERT_PATH_ENV_VAR': 'CA_CERT_PATH',
            'KEY_PATH_ENV_VAR': 'CA_KEY_PATH',
            'SSL_KEY_LOG_ENV_VAR': 'SSLKEYLOGFILE'  # For debugging with Wireshark
        }

    def __getattr__(self, name: str) -> Any:
        """Get config value."""
        return self._config[name]

class HTTPStatus:
    """HTTP status codes and messages."""
    def __init__(self):
        self._config: HTTPStatusDict = {
            'OK': (200, b'HTTP/1.1 200 Connection Established\r\n\r\n'),
            'BAD_REQUEST': (400, b'HTTP/1.1 400 Bad Request\r\n\r\n'),
            'NOT_IMPLEMENTED': (501, b'HTTP/1.1 501 Not Implemented\r\n\r\n'),
            'BAD_GATEWAY': (502, b'HTTP/1.1 502 SSL Error\r\n\r\n'),
            'GATEWAY_TIMEOUT': (504, b'HTTP/1.1 504 Gateway Timeout\r\n\r\n'),
            'INTERNAL_ERROR': (500, b'HTTP/1.1 500 Internal Server Error\r\n\r\n')
        }

    def __getattr__(self, name: str) -> Any:
        """Get config value."""
        return self._config[name]

# Create and export global instances
network_config = NetworkConfig()
ssl_config = SSLConfig()
memory_config = MemoryConfig()
env_vars = EnvVars()
http_status = HTTPStatus()

__all__ = [
    'NetworkConfig', 'SSLConfig', 'MemoryConfig', 'EnvVars', 'HTTPStatus',
    'network_config', 'ssl_config', 'memory_config', 'env_vars', 'http_status'
]
