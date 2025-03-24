import asyncio
import gc
import logging
from typing import Set, Optional
from ..utils.logging import logger

class ServerState:
    def __init__(self):
        self.is_shutting_down = False
        self.active_connections: Set['ProxyConnection'] = set()
        self.shutdown_event = asyncio.Event()
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'bytes_transferred': 0,
            'peak_memory_mb': 0,
            'ssl_contexts_created': 0,
            'ssl_contexts_cleaned': 0,
            'start_time': None
        }
        self._stats_task = None

    async def start_stats_monitoring(self) -> None:
        """Start periodic statistics monitoring."""
        self.stats['start_time'] = asyncio.get_event_loop().time()
        self._stats_task = asyncio.create_task(self._monitor_stats())

    async def _monitor_stats(self) -> None:
        """Monitor and log server statistics periodically."""
        while not self.shutdown_event.is_set():
            try:
                memory_usage = self._get_memory_usage()
                self.stats['peak_memory_mb'] = max(
                    self.stats['peak_memory_mb'], 
                    memory_usage
                )

                uptime = int(asyncio.get_event_loop().time() - self.stats['start_time'])
                logger.info(
                    f"Server Statistics (uptime {uptime}s):\n"
                    f"  Active Connections: {len(self.active_connections)}\n"
                    f"  Total Connections: {self.stats['total_connections']}\n"
                    f"  Bytes Transferred: {self.stats['bytes_transferred']:,}\n"
                    f"  Current Memory: {memory_usage:.1f} MB\n"
                    f"  Peak Memory: {self.stats['peak_memory_mb']:.1f} MB\n"
                    f"  SSL Contexts: {self.stats['ssl_contexts_created']} created, "
                    f"{self.stats['ssl_contexts_cleaned']} cleaned"
                )

                if memory_usage > 500:  # Over 500MB
                    logger.warning(f"High memory usage detected: {memory_usage:.1f} MB")
                    gc.collect()

                await asyncio.sleep(300)  # Report every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats monitoring: {e}")
                await asyncio.sleep(60)  # Wait a bit before retrying

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback if psutil is not available
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown to complete."""
        await self.shutdown_event.wait()
