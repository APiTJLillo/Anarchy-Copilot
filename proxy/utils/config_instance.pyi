"""Type stubs for config instances."""
from typing import List, Tuple

class NetworkConfig:
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

class SSLConfig:
    MAX_HANDSHAKE_ATTEMPTS: int
    HANDSHAKE_RETRY_DELAY: float
    MIN_HANDSHAKE_DELAY: float
    MAX_HANDSHAKE_DELAY: float
    TLS_RECORD_HEADER_SIZE: int
    ALPN_PROTOCOLS: List[str]
    MIN_TLS_VERSION: int
    CIPHER_PREFERENCE: str
    FATAL_SSL_ERRORS: List[str]

class MemoryConfig:
    GC_THRESHOLD: int
    CONTEXT_CACHE_SIZE: int
    HIGH_MEMORY_THRESHOLD_MB: int

class EnvVars:
    HOST_ENV_VAR: str
    PORT_ENV_VAR: str
    CERT_PATH_ENV_VAR: str
    KEY_PATH_ENV_VAR: str
    SSL_KEY_LOG_ENV_VAR: str

class HTTPStatus:
    OK: Tuple[int, bytes]
    BAD_REQUEST: Tuple[int, bytes]
    NOT_IMPLEMENTED: Tuple[int, bytes]
    BAD_GATEWAY: Tuple[int, bytes]
    GATEWAY_TIMEOUT: Tuple[int, bytes]
    INTERNAL_ERROR: Tuple[int, bytes]

network_config: NetworkConfig
ssl_config: SSLConfig
memory_config: MemoryConfig
env_vars: EnvVars
http_status: HTTPStatus