"""Connection handling for TLS connections."""
import asyncio
import logging
import socket
import ssl
from typing import Optional, Dict, Any, Tuple
from async_timeout import timeout as async_timeout

logger = logging.getLogger("proxy.core")

class ConnectionHandler:
    """Handles remote connection establishment."""
    
    def __init__(self, connection_id: str):
        self._connection_id = connection_id
        self.transport = None
        self._closing = False
        
    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle connection established."""
        self.transport = transport
        
    def data_received(self, data: bytes) -> None:
        """Handle received data."""
        if not self._closing and self.transport:
            self.transport.write(data)
            
    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Handle connection lost."""
        self._closing = True
        if self.transport and not self.transport.is_closing():
            self.transport.close()
            
    def get_extra_info(self, name: str, default: Any = None) -> Any:
        """Get transport information."""
        if self.transport:
            return self.transport.get_extra_info(name, default)
        return default 