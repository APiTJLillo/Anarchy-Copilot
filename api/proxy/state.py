"""Proxy state management."""
from typing import Optional
import asyncio
import logging
from .history import ensure_dev_connection

logger = logging.getLogger(__name__)

class ProxyState:
    """Manages proxy state."""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not ProxyState._initialized:
            self._running = False
            self._lock = asyncio.Lock()
            self._proxy_server = None
            self._version = {
                "version": "0.1.0",
                "name": "Anarchy Copilot",
                "api_compatibility": "0.1.0"
            }
            ProxyState._initialized = True

    async def start(self) -> None:
        """Start the proxy."""
        async with self._lock:
            self._running = True
            # Establish WebSocket connection to dev container
            try:
                await ensure_dev_connection()
                logger.info("Established WebSocket connection to dev container")
            except Exception as e:
                logger.error(f"Failed to establish WebSocket connection to dev container: {e}")
            logger.info("Proxy started")

    async def stop(self) -> None:
        """Stop the proxy."""
        async with self._lock:
            if self._proxy_server:
                try:
                    await self._proxy_server.stop()
                except Exception as e:
                    logger.error(f"Error stopping proxy server: {e}")
                self._proxy_server = None
            self._running = False
            logger.info("Proxy stopped")

    @property
    def is_running(self) -> bool:
        """Check if proxy is running."""
        return self._running

    @property
    def running(self) -> bool:
        """Alias for is_running for backward compatibility."""
        return self._running

    @property
    def proxy_server(self):
        """Get the current proxy server instance."""
        return self._proxy_server

    @proxy_server.setter
    def proxy_server(self, server):
        """Set the current proxy server instance."""
        self._proxy_server = server

    @property
    def version(self):
        """Get version information."""
        return self._version

# Global instance
proxy_state = ProxyState() 