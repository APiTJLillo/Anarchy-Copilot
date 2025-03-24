"""Shared types and interfaces for proxy protocols."""
from typing import Protocol, Optional, Dict, Any, Tuple, Union
import asyncio
import ssl
from dataclasses import dataclass

@dataclass
class Request:
    """HTTP request representation."""
    method: str
    target: str
    headers: Dict[str, str]
    body: Optional[Union[str, bytes]] = None

class TlsCapableProtocol(Protocol):
    """Interface for protocols that can handle TLS connections."""
    
    def get_extra_info(self, name: str, default: Any = None) -> Any:
        """Get transport information."""
        ...
        
    def set_tunnel(self, tunnel: asyncio.Transport) -> None:
        """Set the tunnel transport."""
        ...
        
    def write(self, data: bytes) -> None:
        """Write data to the transport."""
        ...
        
    def close(self) -> None:
        """Close the transport."""
        ...

class TlsContextProvider(Protocol):
    """Interface for objects that can provide TLS contexts."""
    
    def get_server_context(self, hostname: str) -> ssl.SSLContext:
        """Get SSL context for server side."""
        ...
        
    def get_client_context(self, hostname: str) -> ssl.SSLContext:
        """Get SSL context for client side."""
        ...

class ConnectionCallbacks(Protocol):
    """Interface for connection lifecycle callbacks."""
    
    async def on_connection_established(self, host: str, port: int) -> None:
        """Called when a connection is established."""
        ...
        
    async def on_connection_lost(self, exc: Optional[Exception] = None) -> None:
        """Called when a connection is lost."""
        ...
        
    async def on_data_received(self, data: bytes) -> None:
        """Called when data is received."""
        ...
        
    async def on_data_sent(self, data: bytes) -> None:
        """Called when data is sent."""
        ... 