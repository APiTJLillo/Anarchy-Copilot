"""Module containing initialized config instances."""
from .constants import NetworkConfig, SSLConfig, MemoryConfig, EnvVars, HTTPStatus

# Create global instances
network_config = NetworkConfig()
ssl_config = SSLConfig()
memory_config = MemoryConfig()
env_vars = EnvVars()
http_status = HTTPStatus()

__all__ = [
    'network_config', 'ssl_config', 'memory_config', 'env_vars', 'http_status'
]