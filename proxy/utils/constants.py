"""Constants used throughout the proxy server."""
from typing import List

# Network Constants
class NetworkConfig:
    BUFFER_SIZE: int = 8192                # Size of data chunks to transfer
    SOCKET_BUFFER_SIZE: int = 131072       # 128KB socket buffer size
    DEFAULT_PORT: int = 8081               # Default proxy server port
    BACKLOG: int = 100                     # Maximum pending connections

    # Keepalive settings
    KEEPALIVE_TIME: int = 60              # Start sending keepalive after 60 seconds
    KEEPALIVE_INTERVAL: int = 10          # Send keepalive every 10 seconds
    KEEPALIVE_COUNT: int = 5              # Drop connection after 5 failed keepalives
    
    # Timeouts
    SHUTDOWN_TIMEOUT: float = 30.0        # Maximum time to wait for graceful shutdown
    CLEANUP_TIMEOUT: float = 5.0          # Maximum time to wait for resource cleanup
    CONNECT_TIMEOUT: float = 30.0         # Timeout for initial connection
    SSL_HANDSHAKE_TIMEOUT: float = 30.0   # Timeout for SSL/TLS handshake
    INACTIVITY_TIMEOUT: float = 300.0     # Connection inactivity timeout (5 minutes)

# Memory Management
class MemoryConfig:
    GC_THRESHOLD: int = 1000              # Operations before forcing garbage collection
    CONTEXT_CACHE_SIZE: int = 100         # Maximum number of cached SSL contexts
    HIGH_MEMORY_THRESHOLD_MB: int = 500   # Memory threshold for forced GC (500MB)

# SSL/TLS Configuration
class SSLConfig:
    # Fatal SSL errors that should terminate connection attempts
    FATAL_SSL_ERRORS: List[str] = [
        "NO_PROTOCOLS_AVAILABLE",
        "CERTIFICATE_VERIFY_FAILED",
        "UNKNOWN_PROTOCOL",
        "BAD_CERTIFICATE",
        "CERTIFICATE_UNKNOWN",
        "WRONG_VERSION_NUMBER",
        "SSLV3_ALERT_CERTIFICATE_UNKNOWN",
        "TLSV1_ALERT_PROTOCOL_VERSION",
        "TLSV1_ALERT_INTERNAL_ERROR"
    ]

    # Default paths (can be overridden by environment variables)
    DEFAULT_CERT_PATH: str = "certs/ca.crt"
    DEFAULT_KEY_PATH: str = "certs/ca.key"

    # Supported ALPN protocols in order of preference
    ALPN_PROTOCOLS: List[str] = ['h2', 'http/1.1']

    # Maximum TLS handshake attempts
    MAX_HANDSHAKE_ATTEMPTS: int = 3
    HANDSHAKE_RETRY_DELAY: float = 0.5    # Base delay between retries (exponential backoff)

# Environment Variables
class EnvVars:
    HOST_ENV_VAR: str = 'ANARCHY_PROXY_HOST'
    PORT_ENV_VAR: str = 'ANARCHY_PROXY_PORT'
    CERT_PATH_ENV_VAR: str = 'CA_CERT_PATH'
    KEY_PATH_ENV_VAR: str = 'CA_KEY_PATH'
    SSL_KEY_LOG_ENV_VAR: str = 'SSLKEYLOGFILE'  # For debugging with Wireshark

# HTTP Status Codes and Messages
class HTTPStatus:
    OK: tuple = (200, b'HTTP/1.1 200 Connection Established\r\n\r\n')
    BAD_REQUEST: tuple = (400, b'HTTP/1.1 400 Bad Request\r\n\r\n')
    NOT_IMPLEMENTED: tuple = (501, b'HTTP/1.1 501 Not Implemented\r\n\r\n')
    BAD_GATEWAY: tuple = (502, b'HTTP/1.1 502 SSL Error\r\n\r\n')
    GATEWAY_TIMEOUT: tuple = (504, b'HTTP/1.1 504 Gateway Timeout\r\n\r\n')
    INTERNAL_ERROR: tuple = (500, b'HTTP/1.1 500 Internal Server Error\r\n\r\n')
