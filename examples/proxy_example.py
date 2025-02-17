"""Example usage of the Anarchy Copilot proxy."""
from pathlib import Path
import asyncio
import re  # Add missing import

from proxy import ProxyServer, ProxyConfig

async def main():
    """Run example proxy server."""
    config = ProxyConfig(
        host="localhost",
        port=8080,
        ca_cert_path=Path("certs/ca.crt"),
        ca_key_path=Path("certs/ca.key"),
        verify_ssl=False,
        websocket_support=True,
        intercept_requests=True,
        intercept_responses=True
    )

    server = ProxyServer(config)

    try:
        await server.start()
        print("Proxy server running at http://localhost:8080")
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await server.stop()

if __name__ == "__main__":
    asyncio.run(main())
