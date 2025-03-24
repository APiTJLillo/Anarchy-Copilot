"""Proxy state management."""
import asyncio
import logging
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional

from .connection import connection_manager, ConnectionManager, ConnectionType, ConnectionState

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = int(os.getenv('ANARCHY_WS_HEARTBEAT_INTERVAL', '15'))
HEARTBEAT_TIMEOUT = HEARTBEAT_INTERVAL * 2

class ProxyState:
    """Global proxy state."""

    def __init__(self):
        """Initialize proxy state."""
        self._is_running = False
        self._last_heartbeat = None
        self._health_check_task = None
        self._shutdown_event = asyncio.Event()
        self._version = {
            "version": "0.1.0",
            "api_version": "0.1.0",
            "proxy_version": "0.1.0"
        }
        self._connection_manager = None
        self._update_callbacks = []

    @property
    def version(self):
        """Get version information."""
        return self._version

    @property
    def is_running(self):
        """Get proxy running state."""
        return self._is_running

    @is_running.setter
    def is_running(self, value: bool):
        """Set proxy running state."""
        self._is_running = value

    @property
    def last_heartbeat(self):
        """Get last heartbeat timestamp."""
        return self._last_heartbeat

    @last_heartbeat.setter
    def last_heartbeat(self, value: float):
        """Set last heartbeat timestamp."""
        self._last_heartbeat = value

    @property
    def connection_manager(self):
        """Get the connection manager."""
        return self._connection_manager

    @connection_manager.setter
    def connection_manager(self, value: ConnectionManager):
        """Set the connection manager."""
        self._connection_manager = value

    @property
    def update_callbacks(self):
        """Get the list of update callbacks."""
        return self._update_callbacks

    @update_callbacks.setter
    def update_callbacks(self, value: list):
        """Set the list of update callbacks."""
        self._update_callbacks = value

    async def initialize(self):
        """Initialize the proxy state."""
        if self._connection_manager:
            logger.warning("[ProxyState] Already initialized")
            return

        logger.info("[ProxyState] Initializing proxy state")
        self._connection_manager = connection_manager
        self.last_heartbeat = time.time()
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("[ProxyState] Proxy state initialized")

    async def shutdown(self):
        """Shutdown the proxy state."""
        if not self._connection_manager:
            return

        logger.info("[ProxyState] Shutting down proxy state")
        self._shutdown_event.set()
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        self._connection_manager = None
        self._is_running = False
        logger.info("[ProxyState] Proxy state shut down")

    async def _health_check_loop(self):
        """Periodically check proxy health."""
        logger.info("[ProxyState] Starting health check loop")
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                time_since_last_heartbeat = current_time - self.last_heartbeat

                # Check for heartbeat timeout
                if time_since_last_heartbeat > HEARTBEAT_TIMEOUT:
                    if self.is_running:
                        logger.warning(
                            f"[ProxyState] No heartbeat detected for {time_since_last_heartbeat:.1f} seconds "
                            f"(timeout: {HEARTBEAT_TIMEOUT}s)"
                        )
                        self._is_running = False
                else:
                    if not self.is_running:
                        logger.info("[ProxyState] Heartbeat detected, proxy is running")
                        self._is_running = True

                # Check connection status
                internal_connections = self._connection_manager.get_connection_by_type(ConnectionType.INTERNAL.value)
                if not internal_connections:
                    logger.warning("[ProxyState] No internal connections")
                else:
                    logger.debug(f"[ProxyState] Active internal connections: {len(internal_connections)}")

                self.last_heartbeat = current_time
                await asyncio.sleep(HEARTBEAT_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ProxyState] Error in health check loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

        logger.info("[ProxyState] Health check loop stopped")

    async def get_status(self) -> Dict[str, Any]:
        """Get the current proxy status."""
        current_time = time.time()
        internal_connections = self._connection_manager.get_connection_by_type(ConnectionType.INTERNAL.value)

        return {
            "initialized": bool(self._connection_manager),
            "running": self.is_running,
            "last_heartbeat": self.last_heartbeat,
            "time_since_heartbeat": current_time - self.last_heartbeat if self.last_heartbeat > 0 else None,
            "internal_connections": len(internal_connections),
            "active_internal_connections": len(internal_connections),  # All connections returned by get_connection_by_type are active
            "timestamp": datetime.utcnow().isoformat()
        }

proxy_state = ProxyState()
