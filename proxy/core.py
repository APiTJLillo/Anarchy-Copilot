"""Core proxy server implementation."""
import asyncio
from fastapi import FastAPI, Request
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Union
import uvicorn
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.middleware.base import BaseHTTPMiddleware

from .config import ProxyConfig
from .server.proxy_server import ProxyServer  # Import directly from the module
from .server import (
    LoggingMiddleware,
    proxy_middleware,
    ProxyResponse,
    LoggingMiddlewareFactory
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("proxy.core")
logger.setLevel(logging.DEBUG)

# Create FastAPI apps
app = FastAPI(
    title="Anarchy Copilot Proxy",
    description="HTTPS-capable intercepting proxy for security testing",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

api_router = FastAPI(
    title="Anarchy Copilot API",
    description="API endpoints for proxy management",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Mount API router
app.mount("/api", api_router)

# Load configuration
host = os.getenv("PROXY_HOST", "0.0.0.0")
port = int(os.getenv("PROXY_PORT", "8080"))
ca_cert = os.getenv("CA_CERT_PATH", "/app/certs/ca.crt")
ca_key = os.getenv("CA_KEY_PATH", "/app/certs/ca.key")

proxy_config = ProxyConfig(
    host=host,
    port=port,
    ca_cert_path=Path(ca_cert),
    ca_key_path=Path(ca_key),
    history_size=1000,
    intercept_requests=True,
    intercept_responses=True,
    websocket_support=True
)

logger.info("=== Starting Anarchy Copilot Proxy Server ===")
logger.info("Loaded proxy configuration: host=%s, port=%d, cert=%s, key=%s", 
           host, port, ca_cert, ca_key)

# Global proxy server instance
proxy_server = None

class ProxyASGIMiddleware(BaseHTTPMiddleware):
    """ASGI middleware for handling proxy requests."""

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Union[Response, ProxyResponse]:
        """Handle proxy requests."""
        if request.url.path.startswith("/api/"):
            # Let API requests through
            return await call_next(request)

        # Add raw socket send to scope for tunneling
        try:
            transport = request.scope.get('extensions', {}).get('transport')
            if transport:
                raw_socket = transport.get_extra_info('socket')
                if raw_socket:
                    request.scope['_raw_socket'] = raw_socket
                    request.scope['_raw_send'] = request.scope.get('send')
        except Exception as e:
            logger.error(f"Error getting raw socket: {e}")

        # Handle request via proxy middleware
        try:
            response = await proxy_middleware(request, call_next, proxy_server)
            
            if response is None:
                # Tunnel handling complete
                return Response(status_code=200)
            
            return response

        except Exception as e:
            logger.error(f"Error in proxy middleware: {e}", exc_info=True)
            return ProxyResponse(
                status_code=500,
                headers={'Content-Type': 'text/plain'},
                body=f"Internal server error: {str(e)}".encode()
            )

# Add middleware in correct order
app.add_middleware(LoggingMiddlewareFactory)
app.add_middleware(ProxyASGIMiddleware)

# API endpoints
@api_router.get("/")
async def read_root():
    """Root endpoint that responds with API and proxy status."""
    return {
        "service": "Anarchy Copilot",
        "proxy_status": "running" if proxy_server and proxy_server.is_running else "stopped",
        "api_status": "online",
        "version": "1.0.0"
    }

# Startup event
@app.on_event("startup")
async def proxy_startup():
    """Start proxy server and verify configuration."""
    global proxy_server
    logger.info("Starting proxy server...")
    
    try:
        if proxy_server is None:
            proxy_server = ProxyServer(proxy_config, add_default_interceptors=True)
            await proxy_server.start()
            logger.info("Proxy server startup complete")
    except Exception as e:
        logger.error("Failed to start proxy server: %s", str(e), exc_info=True)
        raise RuntimeError(f"Proxy server startup failed: {str(e)}")

@api_router.on_event("startup")
async def api_startup():
    """Initialize API endpoints on startup."""
    logger.info("Initializing API endpoints...")
    try:
        logger.info("API endpoints ready")
    except Exception as e:
        logger.error("Failed to initialize API: %s", str(e), exc_info=True)
        raise RuntimeError(f"API initialization failed: {str(e)}")

# Shutdown events
@app.on_event("shutdown")
async def proxy_shutdown():
    """Stop proxy server on shutdown."""
    global proxy_server
    logger.info("Stopping proxy server...")
    try:
        # Stop proxy server
        if proxy_server:
            await proxy_server.stop()
            logger.info("Proxy server stopped")
        
        # Wait for sockets to fully close
        await asyncio.sleep(1.0)
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)
    finally:
        proxy_server = None

@api_router.on_event("shutdown")
async def api_shutdown():
    """Cleanup API resources on shutdown."""
    logger.info("Cleaning up API resources...")
    try:
        logger.info("API cleanup complete")
    except Exception as e:
        logger.error("API cleanup error: %s", str(e), exc_info=True)

# Re-export ProxyServer to avoid circular imports
__all__ = ["ProxyServer"]

# For direct execution
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="debug",
        reload=True
    )
