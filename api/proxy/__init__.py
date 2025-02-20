"""Proxy management API package."""
import logging
from typing import Optional, cast

# Third party imports
from fastapi import APIRouter

# Initialize logging
logger = logging.getLogger(__name__)

# Create router first - needed by endpoints
router = APIRouter(tags=["proxy"])

# Import non-dependent utilities
from .utils import cleanup_port, try_close_sockets, find_processes_using_port  # noqa: F401

# Initialize state
proxy_server = None

def reset_state() -> None:
    """Reset all proxy state to initial values."""
    global proxy_server
    proxy_server = None

def get_proxy_server():
    """Get proxy server instance."""
    return proxy_server

# Import models after basic setup
from .models import *  # noqa: F403
from .analysis_models import *  # noqa: F403

# Import endpoints last, after all dependencies are ready
from .endpoints import *  # noqa: F403

# Define exports
__all__ = [
    'router',
    'proxy_server',
    'reset_state',
    'get_proxy_server',
    'cleanup_port',
    'try_close_sockets',
    'find_processes_using_port'
]

# Add component exports
from .models import __all__ as models_all
from .analysis_models import __all__ as analysis_models_all
from .endpoints import __all__ as endpoints_all
__all__.extend(models_all)
__all__.extend(analysis_models_all)
__all__.extend(endpoints_all)

# Log successful initialization
logger.info("Proxy package initialized successfully")
