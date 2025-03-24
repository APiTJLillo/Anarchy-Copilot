"""Global proxy state management."""
from datetime import datetime
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List

from api.proxy.models import ConnectionInfo
from .types import ConnectionManagerProtocol

logger = logging.getLogger("proxy.core")

class ProxyState:
    """Manages global proxy state and broadcasts updates."""
    
    def __init__(self):
        self._active_connections: Dict[str, Dict[str, Any]] = {}
        self._connection_manager: Optional[ConnectionManagerProtocol] = None
        self._update_callbacks: List[callable] = []
        self._startup_time = datetime.utcnow()
        self.last_heartbeat = time.time()
        self.is_running = True
        
        # Version information
        self.version = {
            "version": "0.1.0",
            "name": "Anarchy Copilot",
            "api_compatibility": "0.1.0"
        }
        
        # Health check state
        self.running = True  # Alias for is_running for compatibility
        self.initialized = False
        self.last_health_check = time.time()
        self.health_check_interval = 5.0  # seconds
        
    def set_connection_manager(self, manager: ConnectionManagerProtocol) -> None:
        """Set the connection manager instance."""
        self._connection_manager = manager
        self.initialized = True
        
    def register_update_callback(self, callback: callable) -> None:
        """Register a callback for state updates."""
        if callback not in self._update_callbacks:
            self._update_callbacks.append(callback)
            
    def unregister_update_callback(self, callback: callable) -> None:
        """Unregister an update callback."""
        if callback in self._update_callbacks:
            self._update_callbacks.remove(callback)
            
    async def broadcast_update(self, update_type: str, data: Any = None) -> None:
        """Broadcast a state update to all registered callbacks."""
        update = {
            "type": update_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        for callback in self._update_callbacks:
            try:
                await callback(update)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")
                
    def get_active_connections(self) -> List[ConnectionInfo]:
        """Get list of all active connections."""
        if not self._connection_manager:
            return []
            
        connections = []
        for conn_id in self._active_connections:
            info = self._connection_manager.get_connection_info(conn_id)
            if info:
                connections.append(info)
        return connections
        
    async def add_connection(self, connection_id: str, info: Dict[str, Any]) -> None:
        """Add a new connection to state."""
        self._active_connections[connection_id] = info
        await self.broadcast_update("connection_added", {
            "id": connection_id,
            "info": info
        })
        
    async def update_connection(self, connection_id: str, key: str, value: Any) -> None:
        """Update connection state."""
        if connection_id in self._active_connections:
            self._active_connections[connection_id][key] = value
            await self.broadcast_update("connection_updated", {
                "id": connection_id,
                "key": key,
                "value": value
            })
            
    async def remove_connection(self, connection_id: str) -> None:
        """Remove a connection from state."""
        if connection_id in self._active_connections:
            del self._active_connections[connection_id]
            await self.broadcast_update("connection_removed", {
                "id": connection_id
            })
            
    def get_status(self) -> Dict[str, Any]:
        """Get current proxy state status."""
        return {
            "version": self.version,
            "running": self.running,
            "initialized": self.initialized,
            "uptime": (datetime.utcnow() - self._startup_time).total_seconds(),
            "last_heartbeat": self.last_heartbeat,
            "last_health_check": self.last_health_check,
            "connections": {
                "active": len(self._active_connections),
                "total": len(self._active_connections)
            }
        }
        
    def update_health_check(self) -> None:
        """Update health check timestamp."""
        self.last_health_check = time.time()
        # Update running state based on heartbeat
        current_time = time.time()
        if current_time - self.last_heartbeat > self.health_check_interval * 2:
            self.running = False
            self.is_running = False
        else:
            self.running = True
            self.is_running = True

# Global instance
proxy_state = ProxyState()
