"""Type definitions for configuration."""
from typing import List, TypedDict, Tuple

class NetworkConfigDict(TypedDict):
    """Network configuration settings."""
    BUFFER_SIZE: int
    SOCKET_BUFFER_SIZE: int
    CONNECT_TIMEOUT: float
    SSL_HANDSHAKE_TIMEOUT: float
    KEEPALIVE_TIME: float
    INACTIVITY_TIMEOUT: float
    CLIENT_HELLO_TIMEOUT: float
    RESPONSE_WAIT: float
    PIPE_TIMEOUT: float
    CLEANUP_TIMEOUT: float
    MAX_RETRIES: int
    MAX_ERRORS: int
    RETRY_DELAY: float
    MEMORY_SAMPLE_INTERVAL: float
    MEMORY_GROWTH_THRESHOLD: int
    MEMORY_CLEANUP_THRESHOLD: int
    THROUGHPUT_LOG_INTERVAL: float

class SSLConfigDict(TypedDict):
    """SSL/TLS configuration settings."""
    MAX_HANDSHAKE_ATTEMPTS: int
    HANDSHAKE_RETRY_DELAY: float
    MIN_HANDSHAKE_DELAY: float
    MAX_HANDSHAKE_DELAY: float
    TLS_RECORD_HEADER_SIZE: int
    ALPN_PROTOCOLS: List[str]
    MIN_TLS_VERSION: int
    CIPHER_PREFERENCE: str
    FATAL_SSL_ERRORS: List[str]

class MemoryConfigDict(TypedDict):
    """Memory management configuration."""
    GC_THRESHOLD: int
    CONTEXT_CACHE_SIZE: int
    HIGH_MEMORY_THRESHOLD_MB: int

class EnvVarsDict(TypedDict):
    """Environment variable names."""
    HOST_ENV_VAR: str
    PORT_ENV_VAR: str
    CERT_PATH_ENV_VAR: str
    KEY_PATH_ENV_VAR: str
    SSL_KEY_LOG_ENV_VAR: str

class HTTPStatusDict(TypedDict):
    """HTTP status codes and messages."""
    OK: Tuple[int, bytes]
    BAD_REQUEST: Tuple[int, bytes]
    NOT_IMPLEMENTED: Tuple[int, bytes]
    BAD_GATEWAY: Tuple[int, bytes]
    GATEWAY_TIMEOUT: Tuple[int, bytes]
    INTERNAL_ERROR: Tuple[int, bytes]