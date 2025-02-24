"""Proxy state tracking and monitoring."""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger("proxy.core")

class ProxyState:
    """Track proxy state information."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not ProxyState._initialized:
            self._connections: Dict[str, Dict[str, Any]] = {}
            self._active_sessions: Dict[str, Dict[str, Any]] = {}
            self._stats: Dict[str, Any] = {
                "start_time": datetime.now(timezone.utc),
                "total_connections": 0,
                "active_connections": 0,
                "total_bytes_sent": 0,
                "total_bytes_received": 0,
                "connection_errors": 0
            }
            self._lock = asyncio.Lock()
            ProxyState._initialized = True

    async def add_connection(self, conn_id: str, info: Dict[str, Any]) -> None:
        """Add a new connection to tracking."""
        async with self._lock:
            self._connections[conn_id] = info
            self._stats["total_connections"] += 1
            self._stats["active_connections"] += 1
            logger.debug(f"Added connection {conn_id} to state tracking")

    async def remove_connection(self, conn_id: str) -> None:
        """Remove a connection from tracking."""
        async with self._lock:
            if conn_id in self._connections:
                conn_info = self._connections.pop(conn_id)
                self._stats["active_connections"] -= 1
                # Update transfer stats
                self._stats["total_bytes_sent"] += conn_info.get("bytes_sent", 0)
                self._stats["total_bytes_received"] += conn_info.get("bytes_received", 0)
                if conn_info.get("error"):
                    self._stats["connection_errors"] += 1
                logger.debug(f"Removed connection {conn_id} from state tracking")

    async def update_connection(self, conn_id: str, key: str, value: Any) -> None:
        """Update a specific connection attribute."""
        async with self._lock:
            if conn_id in self._connections:
                self._connections[conn_id][key] = value

    async def get_connection_info(self, conn_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific connection."""
        async with self._lock:
            return self._connections.get(conn_id)

    async def get_all_connections(self) -> Dict[str, Dict[str, Any]]:
        """Get all current connection information."""
        async with self._lock:
            return self._connections.copy()

    async def get_stats(self) -> Dict[str, Any]:
        """Get current proxy statistics."""
        async with self._lock:
            # Calculate uptime
            uptime = (datetime.now(timezone.utc) - self._stats["start_time"]).total_seconds()
            stats = self._stats.copy()
            stats["uptime_seconds"] = uptime
            return stats

    def connection_exists(self, conn_id: str) -> bool:
        """Check if a connection exists."""
        return conn_id in self._connections

    async def reset_stats(self) -> None:
        """Reset statistics counters."""
        async with self._lock:
            self._stats.update({
                "total_connections": len(self._connections),
                "active_connections": len(self._connections),
                "total_bytes_sent": 0,
                "total_bytes_received": 0,
                "connection_errors": 0
            })
            logger.debug("Reset proxy state statistics")

# Global instance
proxy_state = ProxyState()
