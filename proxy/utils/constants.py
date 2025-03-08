"""Constants used throughout the proxy server."""
from typing import List

# Network Constants
class NetworkConfig:
    BUFFER_SIZE: int = 8192                # Size of data chunks to transfer
    SOCKET_BUFFER_SIZE: int = 131072       # 128KB socket buffer size
    DEFAULT_PORT: int = 8081               # Default proxy server port
    BACKLOG: int = 100                     # Maximum pending connections

    # Keepalive settings
    KEEPALIVE_TIME: int = 30              # Start sending keepalive after 30 seconds
    KEEPALIVE_INTERVAL: int = 5           # Send keepalive every 5 seconds
    KEEPALIVE_COUNT: int = 3              # Drop connection after 3 failed keepalives
    
    # Timeouts
    SHUTDOWN_TIMEOUT: float = 30.0        # Maximum time to wait for graceful shutdown
    CLEANUP_TIMEOUT: float = 5.0          # Maximum time to wait for resource cleanup
    CONNECT_TIMEOUT: float = 30.0         # Increased timeout for initial connection
    SSL_HANDSHAKE_TIMEOUT: float = 30.0   # Increased timeout for SSL/TLS handshake
    INACTIVITY_TIMEOUT: float = 120.0     # Connection inactivity timeout (2 minutes)
    PIPE_TIMEOUT: float = 180.0           # Maximum time for a pipe operation (3 minutes)
    CLIENT_HELLO_TIMEOUT: float = 10.0    # Increased timeout for ClientHello
    RESPONSE_WAIT: float = 0.5            # Wait time after sending 200 before handshake

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
        "TLSV1_ALERT_INTERNAL_ERROR",
        "UNEXPECTED_EOF_WHILE_READING",
        "SSL3_GET_RECORD_WRONG_PACKET_TYPE",
        "SSLV3_ALERT_BAD_CERTIFICATE",
        "TLSV1_ALERT_ACCESS_DENIED",
        "TLSV1_ALERT_DECODE_ERROR",
        "SSL_HANDSHAKE_FAILURE"
    ]

    # TLS Options
    TLS_OPTIONS = [
        'OP_NO_SSLv2',
        'OP_NO_SSLv3',
        'OP_NO_COMPRESSION',
        'OP_CIPHER_SERVER_PREFERENCE',
        'OP_SINGLE_DH_USE',
        'OP_SINGLE_ECDH_USE',
        'OP_NO_TICKET',
        'OP_NO_RENEGOTIATION'
    ]

    # Default paths (can be overridden by environment variables)
    DEFAULT_CERT_PATH: str = "certs/ca.crt"
    DEFAULT_KEY_PATH: str = "certs/ca.key"

    # Supported ALPN protocols in order of preference
    ALPN_PROTOCOLS: List[str] = ['h2', 'http/1.1']

    # Connection retry settings
    MAX_HANDSHAKE_ATTEMPTS: int = 5       # Increased retry attempts
    HANDSHAKE_RETRY_DELAY: float = 0.5    # Base delay between retries (exponential backoff)
    MIN_HANDSHAKE_DELAY: float = 0.1      # Minimum delay between handshake steps
    MAX_HANDSHAKE_DELAY: float = 2.0      # Maximum delay between handshake steps

    # TLS cipher suites in order of preference
    CIPHER_LIST: str = (
        'ECDHE-ECDSA-AES256-GCM-SHA384:'
        'ECDHE-RSA-AES256-GCM-SHA384:'
        'ECDHE-ECDSA-CHACHA20-POLY1305:'
        'ECDHE-RSA-CHACHA20-POLY1305:'
        'ECDHE-ECDSA-AES128-GCM-SHA256:'
        'ECDHE-RSA-AES128-GCM-SHA256'
    )

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
