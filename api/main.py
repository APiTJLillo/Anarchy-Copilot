"""Main FastAPI application."""
import asyncio
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.config import settings
from .proxy.state import proxy_state
from .proxy import router as proxy_router, initialize_proxy
from .health import router as health_router
from .proxy.websocket import router as websocket_router
from .projects import router as projects_router
from .users import router as users_router
from .proxy.websocket import connection_manager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configure health logger
health_logger = logging.getLogger("health")
health_logger.setLevel(logging.DEBUG)

def log_websocket_routes(router):
    """Log WebSocket routes for debugging."""
    logger.info("=== WebSocket Routes Configuration ===")
    prefix = router.prefix if hasattr(router, 'prefix') else ''
    parent_prefix = "/api/proxy/ws"  # The prefix where the router is mounted
    
    for route in router.routes:
        # Check if it's a WebSocket route
        if str(route.endpoint).startswith("<function") and "websocket" in str(route.endpoint).lower():
            # Calculate the full path including all prefixes
            full_path = f"{parent_prefix}{prefix}{route.path}"
            logger.info(f"WebSocket Route Details:")
            logger.info(f"  Base Path: {route.path}")
            logger.info(f"  Full Path: {full_path}")
            logger.info(f"  Handler: {route.endpoint.__name__}")
            logger.info(f"  Protocol: WebSocket")
            # Log runtime access URL for debugging
            logger.info(f"  Runtime URL: ws://localhost:8000{full_path}")
            logger.info("---")

# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    debug=settings.debug
)

# Configure CORS with more permissive settings for development
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8083",
    "ws://localhost:3000",
    "ws://localhost:8000",
    "ws://localhost:8083",
    "http://dev:8000",
    "ws://dev:8000",
    "http://proxy:8083",
    "ws://proxy:8083",
    "http://proxy:8000",
    "ws://proxy:8000",
    "*"  # Allow all origins in development
]

logger.info(f"Configuring CORS with origins: {origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=[
        "Origin", "X-Requested-With", "Content-Type", "Accept", "Authorization",
        "Access-Control-Allow-Origin", "Access-Control-Allow-Credentials",
        "Sec-WebSocket-Protocol", "Sec-WebSocket-Key", "Sec-WebSocket-Version",
        "Sec-WebSocket-Extensions", "Upgrade", "Connection",
        "x-connection-type", "x-proxy-version"
    ],
    expose_headers=["*"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Add routers first
logger.info("Adding routers")
app.include_router(proxy_router, prefix="/api/proxy")  # Mount proxy router at /api/proxy
app.include_router(websocket_router, prefix="/api/proxy/ws")  # Mount WebSocket router at /api/proxy/ws
app.include_router(projects_router)  # Projects router already has /api/projects prefix
app.include_router(users_router, prefix="/api/users")  # Mount users router at /api/users
app.include_router(health_router, prefix="/api/health", tags=["health"])  # Mount health router at /api/health

# Log WebSocket routes
logger.info("Logging WebSocket routes")
log_websocket_routes(websocket_router)

@app.on_event("startup")
async def startup_event():
    """Initialize application state on startup."""
    logger.info("Starting application initialization...")
    
    try:
        # Initialize proxy state
        await initialize_proxy()
        logger.info("Application initialization complete")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up application state on shutdown."""
    logger.info("Starting application shutdown...")
    
    try:
        # Reset proxy state
        proxy_state.is_running = False
        
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        raise

# Log proxy router routes before mounting
logger.debug("Proxy router routes before mounting:")
for route in proxy_router.routes:
    logger.debug(f"  {route.path} [{','.join(route.methods)}] - endpoint: {route.endpoint.__name__ if hasattr(route.endpoint, '__name__') else str(route.endpoint)}")

# Log all routes for debugging
@app.on_event("startup")
async def log_routes():
    """Log all registered routes."""
    logger.debug("All registered routes after mounting:")
    for route in app.routes:
        logger.debug(f"  {route.path} [{','.join(route.methods)}] - endpoint: {route.endpoint.__name__ if hasattr(route.endpoint, '__name__') else str(route.endpoint)}")
        if hasattr(route, "app"):
            logger.debug(f"    Router prefix: {route.app.prefix if hasattr(route.app, 'prefix') else 'No prefix'}")
        # Log route details
        logger.debug(f"    Route details:")
        logger.debug(f"      Name: {route.name}")
        logger.debug(f"      Path: {route.path}")
        logger.debug(f"      Methods: {route.methods}")
        logger.debug(f"      Endpoint: {route.endpoint.__name__ if hasattr(route.endpoint, '__name__') else str(route.endpoint)}")
        logger.debug(f"      Dependencies: {route.dependencies}")
        if hasattr(route, "app"):
            logger.debug(f"      Router: {route.app}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Anarchy Copilot API"}
