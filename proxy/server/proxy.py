"""Main proxy server implementation."""
import asyncio
import logging
import ssl
from typing import Optional, Dict, Any

from .handlers.http import HttpRequestHandler
from .protocol.https_intercept import HttpsInterceptProtocol
from .tls.connection_manager import connection_mgr
from .state import proxy_state

logger = logging.getLogger("proxy.core")

class ProxyServer:
    """Main proxy server implementation."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.server: Optional[asyncio.AbstractServer] = None
        
        # Wire up connection manager with state
        proxy_state.set_connection_manager(connection_mgr)
        
    async def start(self) -> None:
        """Start the proxy server."""
        try:
            self.server = await asyncio.start_server(
                self._handle_connection,
                self.host,
                self.port
            )
            
            addr = self.server.sockets[0].getsockname()
            logger.info(f"Proxy server started on {addr[0]}:{addr[1]}")
            
            async with self.server:
                await self.server.serve_forever()
                
        except Exception as e:
            logger.error(f"Error starting proxy server: {e}")
            raise
            
    async def stop(self) -> None:
        """Stop the proxy server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("Proxy server stopped")
            
    async def _handle_connection(self, reader: asyncio.StreamReader, 
                               writer: asyncio.StreamWriter) -> None:
        """Handle incoming client connection."""
        try:
            # Read first bytes to determine protocol
            first_bytes = await reader.read(1024)
            if not first_bytes:
                writer.close()
                return
                
            # Check if this is a TLS connection
            is_tls = self._is_tls_connection(first_bytes)
            
            if is_tls:
                # Handle HTTPS connection
                connection_id = f"https-{id(writer)}"
                protocol = HttpsInterceptProtocol(connection_id, connection_mgr)
                transport = writer.transport
                
                # Attach protocol to transport
                transport._protocol = protocol
                protocol.connection_made(transport)
                
                # Process initial data
                protocol.data_received(first_bytes)
                
            else:
                # Handle HTTP connection
                connection_id = f"http-{id(writer)}"
                handler = HttpRequestHandler(connection_id, connection_mgr)
                transport = writer.transport
                
                # Attach handler to transport
                transport._protocol = handler
                handler.connection_made(transport)
                
                # Process initial data
                handler.data_received(first_bytes)
                
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
            writer.close()
            
    def _is_tls_connection(self, first_bytes: bytes) -> bool:
        """Determine if connection is TLS based on first bytes."""
        # TLS handshake starts with 0x16 (handshake) followed by version
        return (len(first_bytes) > 2 and
                first_bytes[0] == 0x16 and  # Handshake
                first_bytes[1] == 0x03)     # TLS version major 