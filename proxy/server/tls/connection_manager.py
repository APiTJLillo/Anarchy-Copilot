"""Connection tracking and metrics for HTTPS interception."""
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from uuid import UUID
import logging
import asyncio

from api.proxy.models import ConnectionInfo
from ..types import ConnectionManagerProtocol

logger = logging.getLogger("proxy.core")

class ConnectionManager(ConnectionManagerProtocol):
    """Manages active HTTPS connections and their metrics."""
    
    def __init__(self):
        self._active_connections: Dict[str, Dict[str, Any]] = {}
        self._connection_count = 0
        
    def create_connection(self, connection_id: str, transport: Optional[asyncio.Transport] = None) -> None:
        """Create a new connection entry."""
        self._active_connections[connection_id] = {
            "id": connection_id,
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "bytes_sent": 0,
            "bytes_received": 0,
            "status": "connected",
            "error": None,
            "transport": transport,
            "tls_info": {}
        }
        self._connection_count += 1
        logger.debug(f"Created connection {connection_id}")
    
    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get information about a connection."""
        if connection_id not in self._active_connections:
            return None
            
        conn = self._active_connections[connection_id]
        return ConnectionInfo(
            id=conn["id"],
            created_at=conn["created_at"],
            last_activity=conn["last_activity"],
            bytes_sent=conn["bytes_sent"],
            bytes_received=conn["bytes_received"],
            status=conn["status"],
            error=conn["error"],
            tls_info=conn["tls_info"]
        )
    
    def update_connection(self, connection_id: str, key: str, value: Any) -> None:
        """Update a connection's metadata."""
        if connection_id in self._active_connections:
            self._active_connections[connection_id][key] = value
            self._active_connections[connection_id]["last_activity"] = datetime.now(timezone.utc)
    
    async def record_event(self, connection_id: str, event_type: str, direction: str, 
                         status: str, bytes_transferred: Optional[int] = None) -> None:
        """Record a connection event."""
        if connection_id not in self._active_connections:
            return
            
        conn = self._active_connections[connection_id]
        conn["last_activity"] = datetime.now(timezone.utc)
        
        if bytes_transferred:
            if direction in ["client-proxy", "proxy-client"]:
                conn["bytes_sent"] += bytes_transferred
            else:
                conn["bytes_received"] += bytes_transferred
                
        if status == "error":
            conn["status"] = "error"
            conn["error"] = f"Error during {event_type}"
        elif status == "success":
            conn["status"] = "active"
            conn["error"] = None
    
    async def close_connection(self, connection_id: str) -> None:
        """Close and cleanup a connection."""
        if connection_id not in self._active_connections:
            return
            
        conn = self._active_connections[connection_id]
        if conn["transport"]:
            conn["transport"].close()
            
        del self._active_connections[connection_id]
        logger.debug(f"Closed connection {connection_id}")
    
    def get_active_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self._active_connections)
    
    def get_total_bytes_transferred(self) -> tuple[int, int]:
        """Get total bytes sent and received across all connections."""
        total_sent = sum(conn["bytes_sent"] for conn in self._active_connections.values())
        total_received = sum(conn["bytes_received"] for conn in self._active_connections.values())
        return total_sent, total_received

# Create singleton instance
connection_mgr = ConnectionManager()
