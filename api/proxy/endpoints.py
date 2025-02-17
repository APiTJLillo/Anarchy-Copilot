"""Proxy API endpoint handlers."""
import asyncio
import logging
from typing import Dict, List, Any, Optional, cast
from fastapi import HTTPException, Body, Depends, Path, WebSocket, BackgroundTasks
from starlette.responses import JSONResponse

from proxy.core import ProxyServer
from proxy.config import ProxyConfig
from . import router

__all__ = [
    "get_proxy_status",
    "get_analysis_results",
    "clear_analysis_results",
    "test_endpoint",
    "stop_proxy",
    "start_proxy",
    "websocket_endpoint"
]

logger = logging.getLogger(__name__)

# Initialize state
proxy_server: Optional[ProxyServer] = None

# Import models after state initialization to avoid circular imports
from .models import (
    ProxySettings, 
    InterceptedRequest, 
    InterceptedResponse,
    HistoryEntryResponse
)
from .utils import cleanup_port

def get_proxy_state() -> Optional[ProxyServer]:
    """Get current proxy server instance."""
    global proxy_server
    if proxy_server and proxy_server.is_running:
        return proxy_server
    return None

def assert_server_running(server: Optional[ProxyServer]) -> ProxyServer:
    """Assert server is running and return it."""
    if not server or not server.is_running:
        raise HTTPException(
            status_code=400,
            detail="Proxy server is not running"
        )
    return cast(ProxyServer, server)  # Cast for type checker

def require_proxy() -> ProxyServer:
    """Assert that proxy server is running and return it."""
    return assert_server_running(get_proxy_state())

@router.get("/status", response_model=Dict[str, Any])
async def get_proxy_status() -> Dict[str, Any]:
    """Get current proxy server status."""
    logger.debug("Proxy status endpoint called")
    
    server = get_proxy_state()
    is_running = bool(server and server.is_running)
    
    status = {
        "isRunning": is_running,
        "settings": {
            "host": "127.0.0.1",
            "port": 8080,
            "interceptRequests": False,
            "interceptResponses": False,
            "allowedHosts": [],
            "excludedHosts": [],
            "maxConnections": 100,
            "maxKeepaliveConnections": 20,
            "keepaliveTimeout": 30
        }
    }
    
    if server and is_running:
        status["settings"].update({
            "interceptRequests": server.config.intercept_requests,
            "interceptResponses": server.config.intercept_responses,
            "allowedHosts": list(server.config.allowed_hosts),
            "excludedHosts": list(server.config.excluded_hosts)
        })
    
    logger.debug(f"Proxy status: {status}")
    return status

@router.get("/analysis/results", response_model=List[Dict[str, Any]])
async def get_analysis_results():
    """Get proxy analysis results."""
    logger.debug("Analysis results endpoint called")
    return []

@router.delete("/analysis/results")
async def clear_analysis_results():
    """Clear all analysis results."""
    logger.debug("Clear analysis results endpoint called")
    return {"message": "Analysis results cleared successfully"}

@router.get("/test")
async def test_endpoint() -> Dict[str, str]:
    """Test endpoint to verify routing."""
    logger.debug("Test endpoint called")
    return {"status": "ok", "message": "Test endpoint working"}

@router.post("/stop", status_code=201)
async def stop_proxy(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Stop proxy server."""
    server = get_proxy_state()
    
    if not server:
        return {"message": "Proxy server is already stopped"}

    global proxy_server
    server_ref = server
    proxy_server = None

    async def do_stop(server: ProxyServer) -> None:
        try:
            await asyncio.wait_for(server.stop(), timeout=5.0)
            logger.info("Proxy server stopped successfully")
        except Exception as e:
            logger.warning(f"Non-critical error during stop: {e}")

    background_tasks.add_task(do_stop, server_ref)
    return {"message": "Proxy server stopped successfully"}

@router.post("/start", status_code=201)
async def start_proxy(settings: ProxySettings = Body(...)) -> Dict[str, str]:
    """Start proxy server."""
    global proxy_server
    
    # Stop existing proxy if running
    if get_proxy_state():
        await stop_proxy(BackgroundTasks())
    
    # Ensure port is free
    max_cleanup_retries = 3
    for attempt in range(max_cleanup_retries):
        if await cleanup_port(settings.port, logger):
            break
        if attempt == max_cleanup_retries - 1:
            raise HTTPException(status_code=500, detail=f"Failed to free port {settings.port}")
        await asyncio.sleep(2)
    
    try:
        config = ProxyConfig(
            host=settings.host,
            port=settings.port,
            intercept_requests=settings.interceptRequests,
            intercept_responses=settings.interceptResponses,
            allowed_hosts=set(settings.allowedHosts),
            excluded_hosts=set(settings.excludedHosts),
            max_connections=settings.maxConnections,
            max_keepalive_connections=settings.maxKeepaliveConnections,
            keepalive_timeout=settings.keepaliveTimeout
        )

        proxy_server = ProxyServer(config, add_default_interceptors=False)
        await proxy_server.start()
        logger.info("Proxy server started successfully")
        return {"message": "Proxy server started successfully"}
        
    except Exception as e:
        logger.error(f"Failed to start proxy: {e}")
        proxy_server = None
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/intercept")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time proxy interception."""
    await websocket.accept()
    try:
        await websocket.send_json({"type": "connected"})
        
        while True:
            data = await websocket.receive_json()
            if "type" not in data:
                await websocket.send_json({"type": "error", "message": "Missing type"})
                continue
                
            try:
                if data["type"] == "request":
                    await websocket.send_json({
                        "type": "request_processed",
                        "requestId": data.get("requestId"),
                        "success": True
                    })
                elif data["type"] == "response":
                    await websocket.send_json({
                        "type": "response_processed",
                        "requestId": data.get("requestId"),
                        "success": True
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown type: {data['type']}"
                    })
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if not websocket.client_state.DISCONNECTED:
            await websocket.close(code=1011, reason=str(e))
    finally:
        if not websocket.client_state.DISCONNECTED:
            await websocket.close()
