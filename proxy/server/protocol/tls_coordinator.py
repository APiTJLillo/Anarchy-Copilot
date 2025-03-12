"""Coordinates TLS handshake operations across event loops."""
import asyncio
import logging
from typing import Dict, Optional, Any, Set

logger = logging.getLogger("proxy.core")

class TlsCoordinator:
    """Coordinates TLS handshake operations across event loops."""
    
    def __init__(self):
        self._loops: Dict[str, asyncio.AbstractEventLoop] = {}
        self._active_handshakes: Set[str] = set()
        self._lock = asyncio.Lock()
        
    async def register_handshake(self, connection_id: str, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Register a new TLS handshake operation."""
        async with self._lock:
            if connection_id in self._active_handshakes:
                logger.warning(f"Handshake already registered for {connection_id}")
                return
                
            self._active_handshakes.add(connection_id)
            if loop:
                self._loops[connection_id] = loop
                
    async def finish_handshake(self, connection_id: str) -> None:
        """Mark a handshake operation as complete."""
        async with self._lock:
            self._active_handshakes.discard(connection_id)
            self._loops.pop(connection_id, None)
            
    async def get_handshake_loop(self, connection_id: str) -> Optional[asyncio.AbstractEventLoop]:
        """Get the event loop associated with a handshake."""
        async with self._lock:
            return self._loops.get(connection_id)
            
    async def is_handshake_active(self, connection_id: str) -> bool:
        """Check if a handshake is currently active."""
        async with self._lock:
            return connection_id in self._active_handshakes
            
    def cleanup(self) -> None:
        """Clean up any remaining handshake state."""
        self._active_handshakes.clear()
        self._loops.clear()

# Global coordinator instance
tls_coordinator = TlsCoordinator()