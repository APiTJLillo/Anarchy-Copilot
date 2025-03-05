"""Type definitions for proxy server constants."""
from typing import List, Tuple, TypedDict
from typing_extensions import Final

class NetworkConfig:
    BUFFER_SIZE: Final[int]
    SOCKET_BUFFER_SIZE: Final[int]
    DEFAULT_PORT: Final[int]
    BACKLOG: Final[int]
    KEEPALIVE_TIME: Final[int]
    KEEPALIVE_INTERVAL: Final[int]
    KEEPALIVE_COUNT: Final[int]
    SHUTDOWN_TIMEOUT: Final[float]
    CLEANUP_TIMEOUT: Final[float]
    CONNECT_TIMEOUT: Final[float]
    SSL_HANDSHAKE_TIMEOUT: Final[float]
    INACTIVITY_TIMEOUT: Final[float]

class MemoryConfig:
    GC_THRESHOLD: Final[int]
    CONTEXT_CACHE_SIZE: Final[int]
    HIGH_MEMORY_THRESHOLD_MB: Final[int]

class SSLConfig:
    FATAL_SSL_ERRORS: Final[List[str]]
    DEFAULT_CERT_PATH: Final[str]
    DEFAULT_KEY_PATH: Final[str]
    ALPN_PROTOCOLS: Final[List[str]]
    MAX_HANDSHAKE_ATTEMPTS: Final[int]
    HANDSHAKE_RETRY_DELAY: Final[float]

class EnvVars:
    HOST_ENV_VAR: Final[str]
    PORT_ENV_VAR: Final[str]
    CERT_PATH_ENV_VAR: Final[str]
    KEY_PATH_ENV_VAR: Final[str]
    SSL_KEY_LOG_ENV_VAR: Final[str]

class HTTPStatus:
    OK: Final[Tuple[int, bytes]]
    BAD_REQUEST: Final[Tuple[int, bytes]]
    NOT_IMPLEMENTED: Final[Tuple[int, bytes]]
    BAD_GATEWAY: Final[Tuple[int, bytes]]
    GATEWAY_TIMEOUT: Final[Tuple[int, bytes]]
    INTERNAL_ERROR: Final[Tuple[int, bytes]]

# Type alias for HTTP status
HTTPStatusType = Tuple[int, bytes]

# Type definitions for configuration
class ProxyConfig(TypedDict):
    host: str
    port: int
    cert_path: str
    key_path: str
    backlog: int
    debug: bool

# Default configuration
DEFAULT_CONFIG: Final[ProxyConfig]
