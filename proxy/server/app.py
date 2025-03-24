"""Main FastAPI application for the proxy server."""
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .handlers import proxy_middleware
from ..websocket import ws_manager, create_router
from ..websocket.client_manager import dev_connection_manager

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=[
        "Origin", "X-Requested-With", "Content-Type", "Accept", "Authorization",
        "Access-Control-Allow-Origin", "Access-Control-Allow-Credentials",
        "Sec-WebSocket-Protocol", "Sec-WebSocket-Key", "Sec-WebSocket-Version",
        "Sec-WebSocket-Extensions", "Upgrade", "Connection",
        "x-connection-type", "x-proxy-version"
    ],
)

# Create WebSocket router with manager instance
router = create_router(ws_manager)

# Add WebSocket routes
app.include_router(router, prefix="/api/proxy/ws")

# Add proxy middleware
app.middleware("http")(proxy_middleware)

@app.on_event("startup")
async def startup_event():
    """Initialize proxy server on startup."""
    try:
        # Start WebSocket connection in background task
        await dev_connection_manager.ensure_connection()
    except Exception as e:
        app.logger.error(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up proxy server on shutdown."""
    try:
        await dev_connection_manager.cleanup()
    except Exception as e:
        app.logger.error(f"Error during shutdown: {e}")
        raise 