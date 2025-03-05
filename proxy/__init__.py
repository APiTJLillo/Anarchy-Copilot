"""Anarchy HTTPS Proxy Server.

A high-performance HTTPS proxy server with SSL/TLS interception capabilities
for security testing and debugging.
"""

import os
from typing import Optional, Tuple

from .models import ProxyServer
from .main import run_server
from .utils.constants import NetworkConfig, SSLConfig

__version__ = '1.0.0'

async def create_proxy_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    cert_path: Optional[str] = None,
    key_path: Optional[str] = None
) -> ProxyServer:
    """Create a new proxy server instance.
    
    Args:
        host: Host address to bind to (default: '0.0.0.0')
        port: Port to listen on (default: 8081)
        cert_path: Path to SSL certificate (default: from environment or certs/ca.crt)
        key_path: Path to SSL private key (default: from environment or certs/ca.key)
    
    Returns:
        ProxyServer: Configured proxy server instance
    """
    return ProxyServer(
        host=host,
        port=port,
        cert_path=cert_path,
        key_path=key_path
    )

def get_default_paths() -> Tuple[str, str]:
    """Get default certificate paths.
    
    Returns:
        Tuple[str, str]: (certificate path, key path)
    """
    cert_path = os.getenv('CA_CERT_PATH', SSLConfig.DEFAULT_CERT_PATH)
    key_path = os.getenv('CA_KEY_PATH', SSLConfig.DEFAULT_KEY_PATH)
    return cert_path, key_path

__all__ = [
    'ProxyServer',
    'run_server',
    'create_proxy_server',
    'get_default_paths',
    '__version__'
]
