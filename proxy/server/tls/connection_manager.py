"""Connection tracking and metrics for HTTPS interception."""
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from uuid import UUID
import logging
import asyncio

from api.proxy.models import ConnectionInfo
from api.proxy.websocket import connection_manager

logger = logging.getLogger("proxy.core")

class ConnectionManager:
    """Manages active HTTPS connections and their metrics."""
    
    def __init__(self):
        self._active_connections: Dict[str, Dict[str, Any]] = {}
        self._connection_count = 0
        
    def create_connection(self, connection_id: str, transport: Optional[asyncio.Transport] = None) -> None:
        """Create a new connection entry."""
        self._connection_count += 1
        self._active_connections[connection_id] = {
            "number": self._connection_count,
            "created_at": datetime.now(timezone.utc),
            "client_address": transport.get_extra_info('peername') if transport else None,
            "host": None,
            "port": None,
            "tls_version": None,
            "cipher": None,
            "bytes_received": 0,
            "bytes_sent": 0,
            "requests_processed": 0,
            "events": [],
            "error": None
        }
        logger.debug(f"Created new connection {self._connection_count} ({connection_id})")

    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get current connection info."""
        try:
            conn_info = self._active_connections.get(connection_id, {})
            if not conn_info:
                return None

            host = conn_info.get("host") or "pending"
            try:
                port = int(conn_info.get("port", 0))
            except (TypeError, ValueError):
                port = 0
                
            created_at = conn_info.get("created_at", datetime.now(timezone.utc))

            return ConnectionInfo(
                id=connection_id,
                host=host,
                port=port,
                start_time=created_at.timestamp(),
                status="closed" if conn_info.get("end_time") else "active",
                events=conn_info.get("events", []),
                bytes_received=conn_info.get("bytes_received", 0),
                bytes_sent=conn_info.get("bytes_sent", 0),
                requests_processed=conn_info.get("requests_processed", 0),
                error=conn_info.get("error")
            )
        except Exception as e:
            logger.error(f"Error creating connection info: {e}")
            return None

    def update_connection(self, connection_id: str, key: str, value: Any) -> None:
        """Update connection information."""
        if connection_id in self._active_connections:
            self._active_connections[connection_id][key] = value
            
    async def record_event(self, connection_id: str, event_type: str, direction: str, 
                          status: str, bytes_transferred: Optional[int] = None) -> None:
        """Record a connection event and broadcast update."""
        if connection_id in self._active_connections:
            conn_info = self._active_connections[connection_id]
            conn_info["events"].append({
                "timestamp": datetime.now(timezone.utc).timestamp(),
                "type": event_type,
                "direction": direction,
                "status": status,
                "bytes": bytes_transferred
            })
            
            if bytes_transferred:
                if direction == "received":
                    conn_info["bytes_received"] += bytes_transferred
                else:
                    conn_info["bytes_sent"] += bytes_transferred

            # Broadcast update through WebSocket
            connection_info = self.get_connection_info(connection_id)
            if connection_info:
                await connection_manager.broadcast_connection_update(connection_info)

    async def close_connection(self, connection_id: str) -> None:
        """Close and cleanup a connection."""
        if connection_id in self._active_connections:
            conn_info = self._active_connections[connection_id]
            logger.info(
                f"Connection {conn_info['number']} closed. "
                f"Processed {conn_info['requests_processed']} requests, "
                f"received {conn_info['bytes_received']} bytes"
            )

            # Update final state
            conn_info["end_time"] = datetime.now(timezone.utc)
            
            # Broadcast final state
            connection_info = self.get_connection_info(connection_id)
            if connection_info:
                await connection_manager.broadcast_connection_update(connection_info)
                await connection_manager.broadcast_connection_closed(connection_id)

            # Cleanup
            del self._active_connections[connection_id]

    def get_active_connection_count(self) -> int:
        """Get count of active connections."""
        return len(self._active_connections)

    def get_total_bytes_transferred(self) -> tuple[int, int]:
        """Get total bytes received and sent across all active connections."""
        total_received = sum(c["bytes_received"] for c in self._active_connections.values())
        total_sent = sum(c["bytes_sent"] for c in self._active_connections.values())
        return total_received, total_sent

# Global instance
connection_mgr = ConnectionManager()
