"""Base types and interfaces for proxy protocols."""
from typing import Protocol, Optional, Dict, Any, Tuple
import asyncio
import ssl

class TlsCapableProtocol(Protocol):
    """Interface for protocols that can handle TLS connections."""
    
    @property
    def transport(self) -> Optional[asyncio.Transport]:
        """Get the current transport."""
        ...
        
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

class TlsHandlerBase(Protocol):
    """Base interface for TLS handlers."""
    
    async def wrap_client(self, protocol: TlsCapableProtocol,
                         server_hostname: str,
                         alpn_protocols: Optional[list[str]] = None) -> Tuple[asyncio.Transport, TlsCapableProtocol]:
        """Wrap client connection with TLS."""
        ...
        
    async def wrap_server(self, protocol: TlsCapableProtocol,
                         server_hostname: Optional[str] = None,
                         alpn_protocols: Optional[list[str]] = None) -> Tuple[asyncio.Transport, TlsCapableProtocol]:
        """Wrap server connection with TLS."""
        ...
        
    def update_connection_stats(self, connection_id: str, **kwargs) -> None:
        """Update connection statistics."""
        ...
        
    def close_connection(self, connection_id: str) -> None:
        """Close and cleanup a connection."""
        ...
