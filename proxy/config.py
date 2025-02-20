"""
Configuration management for the Anarchy Copilot proxy module.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from pathlib import Path
import ssl

@dataclass
class ProxyConfig:
    """Configuration settings for the proxy server."""
    
    # Network settings
    host: str = "127.0.0.1"
    port: int = 8080
    
    # SSL/TLS settings
    ca_cert_path: Optional[Path] = None
    ca_key_path: Optional[Path] = None
    ssl_version: int = ssl.PROTOCOL_TLS_SERVER
    
    # Scope settings
    allowed_hosts: Set[str] = field(default_factory=set)
    excluded_hosts: Set[str] = field(default_factory=set)
    
    # Performance settings
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_timeout: int = 30
    connection_timeout: int = 30
    read_timeout: int = 30
    write_timeout: int = 30
    
    # Storage settings
    history_size: int = 1000
    storage_path: Optional[Path] = None
    
    # Feature flags
    intercept_requests: bool = True
    intercept_responses: bool = True
    websocket_support: bool = True
    http2_support: bool = True
    
    # Plugin settings
    enabled_plugins: List[str] = field(default_factory=list)
    plugin_options: Dict[str, Dict] = field(default_factory=dict)
    
    # Security settings
    verify_ssl: bool = True
    strip_proxy_headers: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.port, int) or self.port <= 0 or self.port > 65535:
            raise ValueError("Port must be between 1 and 65535")
        
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("Host must be a non-empty string")
        
        if self.max_connections <= 0:
            raise ValueError("Max connections must be positive")

        if self.max_keepalive_connections <= 0:
            raise ValueError("Max keepalive connections must be positive")
            
        if self.keepalive_timeout <= 0:
            raise ValueError("Keepalive timeout must be positive")
        
        if self.connection_timeout <= 0:
            raise ValueError("Connection timeout must be positive")
        
        if self.read_timeout <= 0:
            raise ValueError("Read timeout must be positive")
        
        if self.write_timeout <= 0:
            raise ValueError("Write timeout must be positive")
        
        if self.history_size <= 0:
            raise ValueError("History size must be positive")
        
        if self.ca_cert_path and not isinstance(self.ca_cert_path, Path):
            self.ca_cert_path = Path(self.ca_cert_path)
            
        if self.ca_key_path and not isinstance(self.ca_key_path, Path):
            self.ca_key_path = Path(self.ca_key_path)
            
        if self.storage_path and not isinstance(self.storage_path, Path):
            self.storage_path = Path(self.storage_path)
            
        # Ensure both cert and key are provided if either is provided
        if bool(self.ca_cert_path) != bool(self.ca_key_path):
            raise ValueError("Both CA certificate and key must be provided together")
    
    def is_in_scope(self, host: str) -> bool:
        """Check if a host is within the configured scope."""
        # Only check excluded hosts, allow everything else
        if host in self.excluded_hosts:
            return False
        return True
    
    def to_dict(self) -> dict:
        """Convert configuration to a dictionary format."""
        return {
            "network": {
                "host": self.host,
                "port": self.port,
            },
            "ssl": {
                "ca_cert_path": str(self.ca_cert_path) if self.ca_cert_path else None,
                "ca_key_path": str(self.ca_key_path) if self.ca_key_path else None,
                "ssl_version": self.ssl_version,
                "verify_ssl": self.verify_ssl,
            },
            "scope": {
                "allowed_hosts": list(self.allowed_hosts),
                "excluded_hosts": list(self.excluded_hosts),
            },
            "performance": {
                "max_connections": self.max_connections,
                "max_keepalive_connections": self.max_keepalive_connections,
                "keepalive_timeout": self.keepalive_timeout,
                "connection_timeout": self.connection_timeout,
                "read_timeout": self.read_timeout,
                "write_timeout": self.write_timeout,
            },
            "storage": {
                "history_size": self.history_size,
                "storage_path": str(self.storage_path) if self.storage_path else None,
            },
            "features": {
                "intercept_requests": self.intercept_requests,
                "intercept_responses": self.intercept_responses,
                "websocket_support": self.websocket_support,
                "http2_support": self.http2_support,
            },
            "plugins": {
                "enabled": self.enabled_plugins,
                "options": self.plugin_options,
            },
            "security": {
                "verify_ssl": self.verify_ssl,
                "strip_proxy_headers": self.strip_proxy_headers,
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ProxyConfig':
        """Create a configuration instance from a dictionary."""
        network = data.get("network", {})
        ssl_config = data.get("ssl", {})
        scope = data.get("scope", {})
        performance = data.get("performance", {})
        storage = data.get("storage", {})
        features = data.get("features", {})
        plugins = data.get("plugins", {})
        security = data.get("security", {})
        
        return cls(
            host=network.get("host", "127.0.0.1"),
            port=network.get("port", 8080),
            ca_cert_path=Path(ssl_config["ca_cert_path"]) if ssl_config.get("ca_cert_path") else None,
            ca_key_path=Path(ssl_config["ca_key_path"]) if ssl_config.get("ca_key_path") else None,
            ssl_version=ssl_config.get("ssl_version", ssl.PROTOCOL_TLS_SERVER),
            allowed_hosts=set(scope.get("allowed_hosts", [])),
            excluded_hosts=set(scope.get("excluded_hosts", [])),
            max_connections=performance.get("max_connections", 100),
            max_keepalive_connections=performance.get("max_keepalive_connections", 20),
            keepalive_timeout=performance.get("keepalive_timeout", 30),
            connection_timeout=performance.get("connection_timeout", 30),
            read_timeout=performance.get("read_timeout", 30),
            write_timeout=performance.get("write_timeout", 30),
            history_size=storage.get("history_size", 1000),
            storage_path=Path(storage["storage_path"]) if storage.get("storage_path") else None,
            intercept_requests=features.get("intercept_requests", True),
            intercept_responses=features.get("intercept_responses", True),
            websocket_support=features.get("websocket_support", True),
            http2_support=features.get("http2_support", True),
            enabled_plugins=plugins.get("enabled", []),
            plugin_options=plugins.get("options", {}),
            verify_ssl=security.get("verify_ssl", True),
            strip_proxy_headers=security.get("strip_proxy_headers", True),
        )
