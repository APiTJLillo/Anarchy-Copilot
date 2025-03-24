"""Base type definitions for TLS handling."""
from typing import Protocol, Optional, Union, Awaitable, Dict, Any, List, Tuple
import asyncio
import ssl

class TlsCapableProtocol(Protocol):
    """Protocol for TLS-capable connections."""

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
    """Protocol for TLS context providers."""
    def get_server_context(self, hostname: str) -> ssl.SSLContext:
        """Get server SSL context."""
        ...
    
    def get_client_context(self, hostname: str) -> ssl.SSLContext:
        """Get client SSL context."""
        ...

class TlsHandlerBase(Protocol):
    """Base protocol for TLS handlers."""
    async def wrap_client(self,
                       protocol: TlsCapableProtocol, 
                       server_hostname: str,
                       alpn_protocols: Optional[List[str]] = None
                       ) -> Tuple[asyncio.Transport, TlsCapableProtocol]:
        """Wrap client connection with TLS."""
        ...

    async def wrap_server(self,
                       protocol: TlsCapableProtocol,
                       server_hostname: str,
                       alpn_protocols: Optional[List[str]] = None
                       ) -> Tuple[asyncio.Transport, TlsCapableProtocol]:
        """Wrap server connection with TLS."""
        ...
        
    def update_connection_stats(self, connection_id: str, **kwargs) -> None:
        """Update connection statistics."""
        ...
        
    def close_connection(self, connection_id: str) -> None:
        """Close and cleanup a connection."""
        ...
