import asyncio
import logging
import os
import sys
from pathlib import Path

from ..config import ProxyConfig
from .proxy_server import ProxyServer
from .certificates import CertificateAuthority

logger = logging.getLogger(__name__)

async def main():
    """Main entry point for the proxy server."""
    # Clear existing log file
    log_file = "proxy-debug.log"
    try:
        if os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.truncate(0)
    except Exception as e:
        print(f"Warning: Could not clear log file: {e}")

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'  # 'w' mode will truncate the file on open
    )

    # Load configuration
    config = ProxyConfig()
    config.host = "0.0.0.0"
    config.port = 8083
    config.history_size = 1000
    config.websocket_support = True
    
    # Set certificate paths - these should be mapped in the container
    config.ca_cert_path = Path("/app/certs/ca.crt")
    config.ca_key_path = Path("/app/certs/ca.key")

    # Initialize CA
    try:
        if not config.ca_cert_path.exists() or not config.ca_key_path.exists():
            logger.error(f"CA certificate files not found at {config.ca_cert_path} and {config.ca_key_path}")
            logger.error("Please ensure the certificate files are mounted in the container")
            sys.exit(1)
            
        ca = CertificateAuthority(
            ca_cert_path=config.ca_cert_path,
            ca_key_path=config.ca_key_path
        )
    except Exception as e:
        logger.error(f"Failed to initialize CA: {e}")
        sys.exit(1)

    # Create and start server
    server = ProxyServer(config=config, ca_instance=ca)
    try:
        await server.start()
        await asyncio.Event().wait()  # Keep the server running
    except KeyboardInterrupt:
        await server.stop()
    except Exception as e:
        logger.error(f"Server error: {e}")
        await server.stop()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 