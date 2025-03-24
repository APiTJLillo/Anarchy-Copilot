"""Shared types and interfaces for the proxy server."""
from typing import Protocol, Dict, Any, Optional
import asyncio
from datetime import datetime

class ConnectionInfo(Protocol):
    """Connection information interface."""
    id: str
    created_at: datetime
    last_activity: datetime
    bytes_sent: int
    bytes_received: int
    status: str
    error: Optional[str]
    tls_info: Dict[str, Any]

class ConnectionManagerProtocol(Protocol):
    """Connection manager interface."""
    def create_connection(self, connection_id: str, transport: Optional[asyncio.Transport] = None) -> None:
        ...
    
    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        ...
    
    def update_connection(self, connection_id: str, key: str, value: Any) -> None:
        ...
    
    async def record_event(self, connection_id: str, event_type: str, direction: str, 
                         status: str, bytes_transferred: Optional[int] = None) -> None:
        ...
    
    async def close_connection(self, connection_id: str) -> None:
        ...
    
    def get_active_connection_count(self) -> int:
        ...
    
    def get_total_bytes_transferred(self) -> tuple[int, int]:
        ... 