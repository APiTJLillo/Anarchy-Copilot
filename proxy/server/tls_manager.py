"""TLS manager implementation."""
import asyncio
import logging
from typing import Optional, Dict, Any, Set
from contextlib import suppress

logger = logging.getLogger("proxy.core")

class TlsManager:
    """Manages TLS operations and cleanup tasks."""
    
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Initialize TLS manager.
        
        Args:
            loop: Optional event loop to use
        """
        self._loop = loop
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._lock = asyncio.Lock()
        
    async def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Start TLS manager with the provided or current event loop."""
        if self._is_running:
            return
            
        async with self._lock:
            try:
                # Get or use provided event loop
                self._loop = loop or asyncio.get_running_loop()
                
                # Schedule cleanup task
                if not self._cleanup_task or self._cleanup_task.done():
                    self._cleanup_task = self._loop.create_task(self._cleanup_loop())
                    
                self._is_running = True
                logger.info("TLS manager started")
                
            except Exception as e:
                logger.error(f"Failed to start TLS manager: {e}")
                if self._cleanup_task:
                    self._cleanup_task.cancel()
                raise
                
    async def stop(self) -> None:
        """Stop TLS manager and cleanup resources."""
        if not self._is_running:
            return
            
        async with self._lock:
            self._is_running = False
            
            # Cancel cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._cleanup_task
                    
            logger.info("TLS manager stopped")
            
    async def _cleanup_loop(self) -> None:
        """Run periodic cleanup operations."""
        while self._is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                # Add cleanup operations here
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                
    @property
    def is_running(self) -> bool:
        """Check if TLS manager is running."""
        return self._is_running

# Global instance
tls_manager = TlsManager()