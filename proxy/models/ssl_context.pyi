from typing import Dict, Optional
import ssl
from pathlib import Path

class SSLContextManager:
    cert_path: str
    key_path: str
    _ssl_contexts: Dict[str, ssl.SSLContext]
    _active_connections: Dict[str, int]
    _initialized: bool
    _init_error: Optional[str]
    _max_cached_contexts: int
    _operation_count: int

    def __init__(self, cert_path: str, key_path: str) -> None: ...
    
    def _initialize_certificates(self) -> None: ...
    def _cleanup_unused_contexts(self) -> None: ...
    def cleanup_resources(self) -> None: ...
    def _load_cert_with_fallback(self) -> ssl.SSLContext: ...
    
    def create_client_context(self, hostname: str) -> ssl.SSLContext: ...
    def create_server_context(self, hostname: str) -> ssl.SSLContext: ...
    
    def _configure_context(self, 
                         context: ssl.SSLContext, 
                         server_side: bool = True,
                         hostname: Optional[str] = None) -> None: ...
    
    def remove_context(self, hostname: str) -> None: ...
    
    @property
    def is_initialized(self) -> bool: ...
    
    @property
    def initialization_error(self) -> Optional[str]: ...
    
    def get_active_connection_count(self, hostname: str) -> int: ...
    def get_total_contexts(self) -> int: ...
    
    def __enter__(self) -> 'SSLContextManager': ...
    def __exit__(self, exc_type: Optional[type], 
                 exc_val: Optional[Exception], 
                 exc_tb: Optional[object]) -> None: ...
