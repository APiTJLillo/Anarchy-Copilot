"""Health monitoring endpoints."""
from fastapi import APIRouter, Depends
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
                    "lastCheck": datetime.utcnow(),
                    "details": f"Version: {proxy_state.version['version']}"
                },
                {
                    "name": "Proxy Service",
                    "status": proxy_status,
                    "lastCheck": datetime.utcnow(),
                    "details": "Running" if proxy_status == "healthy" else "Stopped"
                }
            ]
        }
        logger.debug(f"Returning response: {response}")
        return response
    except Exception as e:
        logger.error(f"Failed to get service status: {e}")
        return {
            "status": "down",
            "version": "unknown",
            "services": []
        }

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
        
        return {
            "cpu": cpu_percent,
            "memory": memory_percent,
            "disk": disk_percent,
            "network": network
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {
            "cpu": 0,
            "memory": 0,
            "disk": 0,
            "network": {"in": 0, "out": 0}
        }

@router.get("/websocket-status")
async def get_websocket_status() -> Dict[str, Any]:
    """Get WebSocket connection status."""
    try:
        # Get connection manager stats
        stats = connection_manager.get_stats()
        
        # Get connection details by type
        ui_connections = [conn for conn in connection_manager.get_active_connections() if conn.type == "ui"]
        internal_connections = [conn for conn in connection_manager.get_active_connections() if conn.type == "internal"]
        
        return {
            "ui": {
                "connected": len(ui_connections) > 0,
                "connectionCount": len(ui_connections),
                "lastMessage": stats["ui"]["last_message"],
                "messageCount": stats["ui"]["message_count"],
                "errorCount": stats["ui"]["error_count"]
            },
            "internal": {
                "connected": len(internal_connections) > 0,
                "connectionCount": len(internal_connections),
                "lastMessage": stats["internal"]["last_message"],
                "messageCount": stats["internal"]["message_count"],
                "errorCount": stats["internal"]["error_count"]
            },
            "connections": [
                {
                    "id": conn.id,
                    "type": conn.type,
                    "connectedAt": conn.connected_at,
                    "lastActivity": conn.last_activity,
                    "messageCount": conn.message_count,
                    "errorCount": conn.error_count
                }
                for conn in connection_manager.get_active_connections()
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get WebSocket status: {e}")
        return {
            "ui": {
                "connected": False,
                "connectionCount": 0,
                "lastMessage": datetime.utcnow(),
                "messageCount": 0,
                "errorCount": 0
            },
            "internal": {
                "connected": False,
                "connectionCount": 0,
                "lastMessage": datetime.utcnow(),
                "messageCount": 0,
                "errorCount": 0
            },
            "connections": []
        }

@router.get("/connections")
async def get_active_connections() -> List[Dict[str, Any]]:
    """Get list of active WebSocket connections."""
    try:
        connections = connection_manager.get_active_connections()
        return [
            {
                "id": conn.id,
                "type": conn.type,
                "connectedAt": conn.connected_at,
                "lastActivity": conn.last_activity
            }
            for conn in connections
        ]
    except Exception as e:
        logger.error(f"Failed to get active connections: {e}")
        return [] 