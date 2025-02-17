"""Proxy management API package."""
from typing import Optional
from fastapi import APIRouter

from proxy.core import ProxyServer
from proxy.config import ProxyConfig

# Create router with proxy tag
router = APIRouter(tags=["proxy"])

# Global proxy server instance
proxy_server: Optional[ProxyServer] = None

def reset_state() -> None:
    """Reset all proxy state to initial values."""
    global proxy_server
    proxy_server = None

# Import and setup other modules
from .models import *  # noqa: F403
from .utils import cleanup_port, try_close_sockets, find_processes_using_port  # noqa: F401
from .endpoints import *  # noqa: F403

__all__ = [
    'router',
    'proxy_server',
    'reset_state',
    'ProxyServer',
    'ProxyConfig',
    'cleanup_port',
    'try_close_sockets',
    'find_processes_using_port'
]

# Add all models and endpoints to __all__
from .models import __all__ as models_all
from .endpoints import __all__ as endpoints_all
__all__.extend(models_all)
__all__.extend(endpoints_all)
