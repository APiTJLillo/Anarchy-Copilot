"""ASGI application setup for proxy server."""
import h11
import logging
from typing import Callable, Dict, Any, Optional, Union
from starlette.types import Scope, Receive, Send

from .custom_protocol import TunnelProtocol
from .proxy_server import ProxyServer

logger = logging.getLogger("proxy.core")

class ASGIHandler:
    """ASGI compatibility layer for proxy server."""

    def __init__(self, proxy_server: ProxyServer):
        self.proxy_server = proxy_server

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send
    ) -> None:
        """Handle ASGI request."""
        if scope["type"] == "http":
            # Get current protocol if it's a CONNECT request
            if scope.get("method") == "CONNECT":
                # Get tunnel protocol
                server = scope.get('server')
                tunnel_protocol = None
                if server and hasattr(server, 'protocols'):
                    for protocol in server.protocols:
                        if isinstance(protocol, TunnelProtocol):
                            tunnel_protocol = protocol
                            break

                # Handle CONNECT with tunnel protocol
                if tunnel_protocol:
                    logger.debug("ASGI layer: Processing CONNECT request")
                    try:
                        # Create h11 request from ASGI scope
                        host = scope.get("path", "").replace("/", "")  # Remove any slashes
                        h11_request = h11.Request(
                            method=b"CONNECT",
                            target=host.encode(),  # Use raw host:port as target
                            headers=scope["headers"]  # Headers are already in correct format
                        )

                        # Let TunnelProtocol handle it
                        logger.debug(f"Forwarding CONNECT {host} to TunnelProtocol")
                        await tunnel_protocol.handle_request(h11_request)

                        # Drain any remaining request body
                        while True:
                            message = await receive()
                            if message["type"] == "http.disconnect":
                                break

                        return

                    except Exception as e:
                        logger.error(f"Tunnel error: {e}", exc_info=True)
                        await send({
                            "type": "http.response.start",
                            "status": 502,
                            "headers": [(b"content-type", b"text/plain")]
                        })
                        await send({
                            "type": "http.response.body",
                            "body": f"Tunnel failed: {str(e)}".encode()
                        })
                        return

                # No tunnel protocol available
                await send({
                    "type": "http.response.start",
                    "status": 502,
                    "headers": [(b"content-type", b"text/plain")]
                })
                await send({
                    "type": "http.response.body",
                    "body": b"Tunnel protocol not available"
                })
                return
            
            # Handle regular HTTP request
            response = await self.proxy_server.handle_request(scope, receive, send)
            if response:
                await self._send_response(send, response)
        elif scope["type"] == "lifespan":
            await self._handle_lifespan(scope, receive, send)
        else:
            logger.warning(f"Unsupported scope type: {scope['type']}")
            if scope["type"] == "websocket":
                await send({
                    "type": "websocket.close",
                    "code": 1000,
                    "reason": "WebSocket connections not supported",
                })
            return

    async def _handle_lifespan(
        self, 
        scope: Scope, 
        receive: Receive, 
        send: Send
    ) -> None:
        """Handle ASGI lifespan events."""
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                try:
                    # Server startup is handled by ProxyServer.start()
                    await send({"type": "lifespan.startup.complete"})
                except Exception as e:
                    await send({
                        "type": "lifespan.startup.failed",
                        "message": str(e)
                    })
            elif message["type"] == "lifespan.shutdown":
                try:
                    await self.proxy_server.stop()
                    await send({"type": "lifespan.shutdown.complete"})
                except Exception as e:
                    await send({
                        "type": "lifespan.shutdown.failed",
                        "message": str(e)
                    })
                return

    async def _send_response(self, send: Send, response: Any) -> None:
        """Send response through ASGI interface."""
        await send({
            "type": "http.response.start",
            "status": response.status_code,
            "headers": [
                (k.lower().encode("latin1"), v.encode("latin1"))
                for k, v in response.headers.items()
            ]
        })
        
        await send({
            "type": "http.response.body",
            "body": response.body
        })
