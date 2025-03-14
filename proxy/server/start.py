"""Script to start the proxy server."""
import asyncio
import logging
from pathlib import Path
from api.config import Settings
from proxy.server.certificates import CertificateAuthority
from proxy.config import ProxyConfig
from proxy.server.proxy_server import ProxyServer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Start the proxy server."""
    try:
        # Load settings
        settings = Settings()
        
        # Initialize CA
        ca = CertificateAuthority(
            ca_cert_path=Path(settings.ca_cert_path),
            ca_key_path=Path(settings.ca_key_path)
        )
        
        # Create proxy server instance
        proxy_config = ProxyConfig(
            host=settings.proxy_host,
            port=settings.proxy_port,
            max_connections=settings.proxy_max_connections,
            max_keepalive_connections=settings.proxy_max_keepalive_connections,
            keepalive_timeout=settings.proxy_keepalive_timeout,
            ca_cert_path=settings.ca_cert_path,
            ca_key_path=settings.ca_key_path
        )
        
        proxy_server = ProxyServer(config=proxy_config, ca_instance=ca)
        await proxy_server.start()
        
        logger.info("Proxy server started successfully")
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down proxy server...")
        await proxy_server.stop()
    except Exception as e:
        logger.error(f"Failed to start proxy server: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 