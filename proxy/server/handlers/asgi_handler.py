"""ASGI handler for proxy requests."""
import logging
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..proxy_server import ProxyServer
    from ...config import ProxyConfig

from urllib.parse import unquote
from ..protocol import HttpsInterceptProtocol
from ..protocol.types import Request

logger = logging.getLogger("proxy.core")

class ASGIHandler:
    """ASGI application handler for proxy server."""
    
    def __init__(self, proxy_server: 'ProxyServer'):
        """Initialize ASGI handler."""
        self.proxy_server = proxy_server
        logger.info(f"Proxy server starting on port {proxy_server.port}")
        logger.info(f"Memory monitoring settings: {proxy_server.config.get_memory_settings()}")

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        """Handle ASGI requests."""
        client_addr = scope.get('client', ('unknown', 0))[0]
        logger.info(f"[{client_addr}] Received {scope['type']} request: {scope.get('method', 'UNKNOWN')} {scope.get('path', '/')}")
        logger.debug(f"[{client_addr}] Full request scope: {scope}")
        logger.debug(f"[{client_addr}] Headers: {scope.get('headers', [])}")

        if scope["type"] == "lifespan":
            await self._handle_lifespan(scope, receive, send)
        elif scope["type"] == "http":
            if scope["method"] == "CONNECT":
                logger.info(f"[{client_addr}] Routing CONNECT request for {scope['path']}")
                await self._handle_connect(scope, receive, send)
            else:
                await self.proxy_server.handle_request(scope, receive, send)

    async def _handle_connect(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        """Handle CONNECT requests using HttpsInterceptProtocol."""
        client_addr = scope.get('client', ('unknown', 0))[0]
        try:
            # Create protocol instance
            protocol = HttpsInterceptProtocol()
            
            # Get transport from server
            transport = scope.get('extensions', {}).get('transport')
            if not transport:
                # Try alternate methods to get transport
                server = scope.get('server')
                if hasattr(server, 'transport'):
                    transport = server.transport
                elif hasattr(server, 'handle'):
                    transport = server.handle.transport
            
            if not transport:
                raise RuntimeError("Could not get transport from ASGI scope")

            # Initialize protocol with transport
            protocol.connection_made(transport)
            
            # Parse target
            target = unquote(scope['path'])
            if not target:
                raise ValueError("No target specified in CONNECT request")

            # Create request object
            headers = [(k.decode('latin1'), v.decode('latin1')) 
                      for k, v in scope.get('headers', [])]
            request = Request(
                method='CONNECT',
                target=target,
                headers=dict(headers),
                body=b''
            )

            # Handle the request using our protocol
            await protocol.handle_request(request)

        except Exception as e:
            logger.error(f"[{client_addr}] CONNECT handling failed: {e}")
            await send({
                'type': 'http.response.start',
                'status': 502,
                'headers': [(b'content-type', b'text/plain')]
            })
            await send({
                'type': 'http.response.body',
                'body': str(e).encode(),
                'more_body': False
            })

    async def _handle_lifespan(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        """Handle lifespan events."""
        logger.debug(f"[unknown] Received lifespan request: UNKNOWN /")
        logger.debug(f"[unknown] Full request scope: {scope}")
        logger.debug(f"[unknown] Headers: {scope.get('headers', [])}")
        
        try:
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    logger.info("Received lifespan.startup event")
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    logger.info("Received lifespan.shutdown event")
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        except Exception as e:
            logger.error(f"Error handling lifespan event: {e}")
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.failed", "message": str(e)})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.failed", "message": str(e)})
            
    async def _send_error(self, send: Any, status: int, message: str) -> None:
        """Send an error response."""
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"text/plain")]
        })
        await send({
            "type": "http.response.body",
            "body": message.encode()
        }) 