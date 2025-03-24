"""Proxy management API package."""
import logging
from typing import Optional, cast
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import asyncio

# Third party imports
from fastapi import APIRouter

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create router first - needed by endpoints
logger.debug("Creating proxy router")
router = APIRouter(
    tags=["proxy"],
    responses={404: {"description": "Not found"}},
    default_response_class=JSONResponse
)

# Import non-dependent utilities
from .utils import cleanup_port, try_close_sockets, find_processes_using_port  # noqa: F401

# Initialize state
proxy_server = None

# Import state management
from .state import proxy_state

# Initialize proxy state
async def initialize_proxy():
    """Initialize proxy state and establish WebSocket connection."""
    logger.info("Initializing proxy state and connections...")
    
    # Initialize proxy state first
    await proxy_state.initialize()
    logger.info("Proxy state initialized")
    
    # Import and start WebSocket connection
    from .history import ensure_dev_connection
    await ensure_dev_connection()
    
    logger.info("Proxy initialization complete")

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

# Import endpoints
logger.debug("Importing proxy endpoints")
from .endpoints import (  # noqa: F403
    health_check,
    create_session,
    start_proxy,
    stop_proxy,
    get_history,
    clear_history,
    get_analysis_results,
    clear_analysis_results,
    get_proxy_status,
    create_rule,
    update_rule,
    delete_rule,
    list_rules,
    reorder_rules,
    get_connections
)

# Log all registered routes
logger.debug("Proxy router routes after endpoint imports:")
for route in router.routes:
    logger.debug(f"  {route.path} [{','.join(route.methods)}] - endpoint: {route.endpoint.__name__ if hasattr(route.endpoint, '__name__') else str(route.endpoint)}")
    # Log route details
    logger.debug(f"  Route details:")
    logger.debug(f"    Name: {route.name}")
    logger.debug(f"    Path: {route.path}")
    logger.debug(f"    Methods: {route.methods}")
    logger.debug(f"    Endpoint: {route.endpoint.__name__ if hasattr(route.endpoint, '__name__') else str(route.endpoint)}")
    logger.debug(f"    Dependencies: {route.dependencies}")
    if hasattr(route, "app"):
        logger.debug(f"    Router: {route.app}")

# Import WebSocket router after endpoints
from .websocket import router as websocket_router
router.include_router(websocket_router)

# Define exports
__all__ = [
    'router',
    'proxy_server',
    'reset_state',
    'get_proxy_server',
    'cleanup_port',
    'try_close_sockets',
    'find_processes_using_port',
    'initialize_proxy'
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
