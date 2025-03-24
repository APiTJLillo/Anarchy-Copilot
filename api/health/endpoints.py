"""Health monitoring endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from database.session import get_db
from datetime import datetime
import psutil
import logging
from typing import List, Dict, Any
from ..proxy.connection import connection_manager
from ..proxy.state import proxy_state

# Create router with prefix
router = APIRouter()
logger = logging.getLogger("health")
logger.setLevel(logging.DEBUG)

@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get system health status."""
    logger.debug("Received request to /status endpoint")
    try:
        # Get proxy service status
        proxy_status = "healthy" if proxy_state.is_running else "down"
        logger.debug(f"Proxy status: {proxy_status}")
        
        response = {
            "status": "healthy",
            "version": proxy_state.version["version"],
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.debug(f"Returning response: {response}")
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/services")
async def get_service_status() -> Dict[str, Any]:
    """Get status of all services."""
    logger.debug("Received request to /services endpoint")
    try:
        # Get proxy service status
        proxy_status = "healthy" if proxy_state.is_running else "down"
        logger.debug(f"Proxy status: {proxy_status}")
        
        response = {
            "status": "healthy",
            "version": proxy_state.version["version"],
            "services": [
                {
                    "name": "API Service",
                    "status": "healthy",
                    "lastCheck": datetime.utcnow().isoformat(),
                    "details": f"Version: {proxy_state.version['version']}"
                },
                {
                    "name": "Proxy Service",
                    "status": proxy_status,
                    "lastCheck": datetime.utcnow().isoformat(),
                    "details": "Running" if proxy_status == "healthy" else "Stopped"
                }
            ]
        }
        logger.debug(f"Returning response: {response}")
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Failed to get service status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_system_metrics() -> Dict[str, Any]:
    """Get system metrics."""
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Get network I/O
        net_io = psutil.net_io_counters()
        network = {
            "in": round(net_io.bytes_recv / (1024 * 1024), 2),  # MB/s
            "out": round(net_io.bytes_sent / (1024 * 1024), 2)  # MB/s
        }
        
        return JSONResponse(content={
            "cpu": cpu_percent,
            "memory": memory_percent,
            "disk": disk_percent,
            "network": network
        })
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/websocket-status")
async def get_websocket_status() -> Dict[str, Any]:
    """Get WebSocket connection status."""
    try:
        # Get connection manager stats
        stats = connection_manager.get_stats()
        
        # Helper function to format datetime objects
        def format_datetime(dt):
            if dt is None:
                return None
            return dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)
        
        # Helper function to process connection history
        def process_connection_history(history):
            return [{
                **entry,
                'timestamp': format_datetime(entry.get('timestamp'))
            } for entry in history]
        
        # Helper function to process active connections
        def process_active_connections(connections):
            return [{
                **conn,
                'connected_since': format_datetime(conn.get('connected_since')),
                'last_activity': format_datetime(conn.get('last_activity')),
                'connection_history': process_connection_history(conn.get('connection_history', []))
            } for conn in connections]
        
        # Process UI stats
        ui_stats = {
            "connected": stats["ui"]["connected"],
            "connectionCount": stats["ui"]["connection_count"],
            "lastMessage": format_datetime(stats["ui"]["last_message"]),
            "messageCount": stats["ui"]["message_count"],
            "errorCount": stats["ui"]["error_count"],
            "active_connections": process_active_connections(stats["ui"]["active_connections"]),
            "connection_history": process_connection_history(stats["ui"]["connection_history"])
        }
        
        # Process internal stats
        internal_stats = {
            "connected": stats["internal"]["connected"],
            "connectionCount": stats["internal"]["connection_count"],
            "lastMessage": format_datetime(stats["internal"]["last_message"]),
            "messageCount": stats["internal"]["message_count"],
            "errorCount": stats["internal"]["error_count"],
            "active_connections": process_active_connections(stats["internal"]["active_connections"]),
            "connection_history": process_connection_history(stats["internal"]["connection_history"])
        }
        
        # Get active connections
        active_connections = [{
            "id": conn.id,
            "type": conn.type,
            "connectedAt": format_datetime(conn.connected_at),
            "lastActivity": format_datetime(conn.last_activity),
            "messageCount": conn.message_count,
            "errorCount": conn.error_count
        } for conn in connection_manager.get_active_connections()]
        
        response_data = {
            "status": "ok",
            "websocket": {
                "ui": ui_stats,
                "internal": internal_stats,
                "connections": active_connections
            }
        }
        
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Failed to get WebSocket status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connections")
async def get_active_connections() -> List[Dict[str, Any]]:
    """Get list of active WebSocket connections."""
    try:
        connections = connection_manager.get_active_connections()
        return JSONResponse(content=[
            {
                "id": conn.id,
                "type": conn.type,
                "connectedAt": conn.connected_at.isoformat(),
                "lastActivity": conn.last_activity.isoformat()
            }
            for conn in connections
        ])
    except Exception as e:
        logger.error(f"Failed to get active connections: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 