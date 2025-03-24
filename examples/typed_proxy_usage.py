"""Examples of using the proxy server with proper type hints."""

import asyncio
import ssl
from typing import Optional, AsyncIterator
from contextlib import asynccontextmanager

from proxy import (
    ProxyServer,
    create_proxy_server,
    get_default_paths,
)
from proxy.models import ServerState
from proxy.utils.constants import NetworkConfig, SSLConfig

async def basic_usage() -> None:
    """Basic usage example."""
    # Create and start a proxy server with default settings
    server = await create_proxy_server()
    try:
        await server.start()
    except KeyboardInterrupt:
        server.close()

async def custom_configuration() -> None:
    """Example with custom configuration."""
    # Get default certificate paths
    cert_path, key_path = get_default_paths()
    
    # Create server with custom settings
    server = ProxyServer(
        host='127.0.0.1',
        port=8443,
        cert_path=cert_path,
        key_path=key_path
    )
    
    try:
        await server.start()
    finally:
        server.close()
        await server.cleanup_resources()

@asynccontextmanager
async def proxy_server_context(
    host: Optional[str] = None,
    port: Optional[int] = None
) -> AsyncIterator[ProxyServer]:
    """Context manager for running a proxy server."""
    server = await create_proxy_server(host=host, port=port)
    
    # Start server in background task
    server_task = asyncio.create_task(server.start())
    
    try:
        yield server
    finally:
        server.close()
        await server.cleanup_resources()
        server_task.cancel()
        with suppress(asyncio.CancelledError):
            await server_task

async def monitor_server_stats(state: ServerState) -> None:
    """Example of monitoring server statistics."""
    while True:
        # Get current stats
        stats = state.stats
        
        print(f"""
Server Statistics:
  Active Connections: {stats['active_connections']}
  Total Connections: {stats['total_connections']}
  Bytes Transferred: {stats['bytes_transferred']:,}
  Memory Usage: {stats['peak_memory_mb']:.1f} MB
  SSL Contexts: {stats['ssl_contexts_created']} created, {stats['ssl_contexts_cleaned']} cleaned
""")
        
        await asyncio.sleep(5)

async def main() -> None:
    """Run example scenarios."""
    # Using context manager
    async with proxy_server_context(port=8443) as server:
        # Start stats monitoring
        monitor_task = asyncio.create_task(
            monitor_server_stats(server.state)
        )
        
        try:
            # Wait for Ctrl+C
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await monitor_task

if __name__ == '__main__':
    from contextlib import suppress
    
    # Set up uvloop if available
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    
    # Run example
    asyncio.run(main())
