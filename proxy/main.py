"""Main entry point for the proxy server."""

import sys
import uvicorn
import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from proxy.server import app as proxy_app
from proxy.utils.logging import logger
from proxy.websocket.client_manager import dev_connection_manager

# Create main FastAPI application
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=[
        "Origin", "Content-Type", "Accept", "Authorization",
        "Access-Control-Allow-Origin", "Access-Control-Allow-Credentials",
        "Sec-WebSocket-Protocol", "Sec-WebSocket-Key", "Sec-WebSocket-Version",
        "Sec-WebSocket-Extensions", "Upgrade", "Connection",
        "x-connection-type", "x-proxy-version"
    ],
)

# Mount proxy application under /proxy path to avoid conflicts
app.mount("/proxy", proxy_app)

@app.on_event("startup")
async def startup_event():
    """Initialize WebSocket connection on startup."""
    try:
        logger.info("Starting proxy server...")
        # Initialize WebSocket connection
        await dev_connection_manager.ensure_connection()
        logger.info("WebSocket connection initialized")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up WebSocket connection on shutdown."""
    try:
        logger.info("Shutting down proxy server...")
        await dev_connection_manager.cleanup()
        logger.info("WebSocket connection cleaned up")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "Anarchy Copilot Proxy"}

def main():
    """Run the proxy server."""
    try:
        uvicorn.run(
            "proxy.main:app",
            host="0.0.0.0",
            port=8083,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start proxy server: {e}")
        raise

if __name__ == '__main__':
    main()
