import os
import signal
from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
import asyncio
import logging
from typing import Optional
from proxy.models import ProxyServer, ServerState

logger = logging.getLogger(__name__)

app = FastAPI()
class SharedState:
    def __init__(self):
        self.proxy_server: Optional[ProxyServer] = None
        self.proxy_task: Optional[asyncio.Task] = None
        self.is_shutting_down = False

state = SharedState()

@app.on_event("startup")
async def startup_event():
    global state
    try:
        # Initialize the proxy server with environment variables
        state.proxy_server = ProxyServer()
        
        # Start the proxy server in the background
        state.proxy_task = asyncio.create_task(state.proxy_server.start())
        logger.info(f"Proxy server starting on port {state.proxy_server.port}")

        # Set up signal handlers for graceful shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop = asyncio.get_running_loop()
                loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
                logger.info(f"Signal handler set up for {sig.name}")
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                logger.warning(f"Could not set up signal handler for {sig.name}")
    except Exception as e:
        logger.error(f"Failed to start proxy server: {str(e)}")
        raise

async def shutdown():
    """Clean shutdown of the proxy server."""
    if state.is_shutting_down:
        return
        
    state.is_shutting_down = True
    logger.info("Initiating graceful shutdown...")
    
    if state.proxy_server:
        state.proxy_server.close()
        
    if state.proxy_task:
        state.proxy_task.cancel()
        try:
            await state.proxy_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error during proxy shutdown: {str(e)}")
    
    logger.info("Proxy server shut down successfully")

@app.on_event("shutdown")
async def shutdown_event():
    await shutdown()

@app.get("/health")
async def health_check():
    global state
    if state.proxy_server is None or state.proxy_task is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Proxy server not initialized"}
        )
    if state.proxy_task.done():
        exception = state.proxy_task.exception()
        if exception:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "message": str(exception)}
            )
    return {
        "status": "healthy",
        "message": "Proxy server is running",
        "port": state.proxy_server.port if state.proxy_server else None,
        "shutting_down": state.is_shutting_down
    }
