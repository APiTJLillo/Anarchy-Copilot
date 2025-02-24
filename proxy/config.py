"""Proxy server configuration."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

@dataclass
class ProxyConfig:
    """Proxy server configuration."""
    host: str = "127.0.0.1"
    port: int = 8080
    
    # Connection settings
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_timeout: int = 30
    history_size: int = 1000

    # Feature flags
    websocket_support: bool = True
    http2_support: bool = False
    intercept_responses: bool = True
    intercept_requests: bool = True

    # SSL/TLS settings
    ca_cert_path: Optional[Path] = None
    ca_key_path: Optional[Path] = None

    # Filtering
    allowed_hosts: List[str] = None
    excluded_hosts: List[str] = None

    def __post_init__(self):
        """Initialize default lists."""
        if self.allowed_hosts is None:
            self.allowed_hosts = []
        if self.excluded_hosts is None:
            self.excluded_hosts = []

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ProxyConfig':
        """Create config from dictionary."""
        # Only include keys that match our fields
        filtered_dict = {
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        }
        return cls(**filtered_dict)

    def update(self, new_config: dict) -> None:
        """Update config with new values."""
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
