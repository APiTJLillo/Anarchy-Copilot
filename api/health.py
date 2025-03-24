"""Health check and version information endpoints."""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from version import __version__
from api.proxy.connection_manager import connection_manager

router = APIRouter()

@router.get("/health", tags=["system"])
async def health_check() -> Dict[str, str]:
    """Get system health status and version information."""
    return {
        "status": "healthy",
        "version": __version__,
        "api": "online"
    }

@router.get("/status", tags=["system"])
async def status_check() -> Dict[str, str]:
    """Simple status check endpoint."""
    return {
        "status": "healthy"
    }

@router.get("/version", tags=["system"])
async def version_info() -> Dict[str, str]:
    """Get detailed version information."""
    return {
        "version": __version__,
        "name": "Anarchy Copilot",
        "api_compatibility": f"^{__version__.split('.')[0]}.0.0"
    }

@router.get("/websocket-status", tags=["system"])
async def websocket_status() -> JSONResponse:
    """Get WebSocket connection status."""
    try:
        stats = connection_manager.get_stats()
        active_connections = connection_manager.get_active_connections()
        
        # Convert active connections to serializable format
        connections = [{
            "id": conn.id,
            "type": conn.type,
            "connectedAt": conn.connected_at.isoformat(),
            "lastActivity": conn.last_activity.isoformat(),
            "messageCount": conn.message_count,
            "errorCount": conn.error_count
        } for conn in active_connections]
        
        response_data = {
            "status": "ok",
            "websocket": {
                "ui": {
                    "connected": stats["ui"]["connected"],
                    "connectionCount": stats["ui"]["connection_count"],
                    "lastMessage": stats["ui"]["last_message"].isoformat() if stats["ui"]["last_message"] else None,
                    "messageCount": stats["ui"]["message_count"],
                    "errorCount": stats["ui"]["error_count"],
                    "active_connections": stats["ui"]["active_connections"],
                    "connection_history": stats["ui"]["connection_history"]
                },
                "internal": {
                    "connected": stats["internal"]["connected"],
                    "connectionCount": stats["internal"]["connection_count"],
                    "lastMessage": stats["internal"]["last_message"].isoformat() if stats["internal"]["last_message"] else None,
                    "messageCount": stats["internal"]["message_count"],
                    "errorCount": stats["internal"]["error_count"],
                    "active_connections": stats["internal"]["active_connections"],
                    "connection_history": stats["internal"]["connection_history"]
                },
                "connections": connections
            }
        }
        
        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
