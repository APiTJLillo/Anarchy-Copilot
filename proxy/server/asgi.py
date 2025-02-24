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
            if scope.get("method") == "CONNECT":
                await self._handle_connect(scope, receive, send)
            else:
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

    async def _handle_connect(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle CONNECT requests for HTTPS tunneling."""
        # Get tunnel protocol from transport
        transport = scope.get('transport')
        protocol = None
        if transport and hasattr(transport, 'get_protocol'):
            protocol = transport.get_protocol()
        elif transport and hasattr(transport, '_protocol'):
            protocol = transport._protocol

        tunnel_protocol = protocol if isinstance(protocol, TunnelProtocol) else None

        if not tunnel_protocol:
            logger.error("Could not get tunnel protocol from connection")
            await self._send_error(send, 502, "Internal server error: missing tunnel protocol")
            return

        # Extract host and port from path
        target = scope.get("path", "").replace("/", "")  # Remove any slashes
        try:
            if ":" in target:
                host, port = target.split(":", 1)
                port = int(port)
            else:
                host = target
                port = 443  # Default to HTTPS port

            if not host or port <= 0 or port > 65535:
                raise ValueError("Invalid host or port")

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid CONNECT target: {target}")
            await self._send_error(send, 400, "Invalid CONNECT target")
            return

        # Send initial CONNECT response
        try:
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"connection", b"keep-alive"),
                    (b"proxy-connection", b"keep-alive"),
                    (b"server", b"Anarchy-Copilot-Proxy"),
                ]
            })
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": False
            })

            # Handle CONNECT with tunnel protocol
            try:
                logger.debug(f"Processing CONNECT request for {host}:{port}")
                await tunnel_protocol._handle_connect(host, port)

                # Drain any remaining request body
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        break

            except Exception as e:
                logger.error(f"Tunnel error: {e}", exc_info=True)
                # No need to send error response here since we already sent 200
                await tunnel_protocol._cleanup(error=str(e))

        except Exception as e:
            logger.error(f"Failed to send CONNECT response: {e}", exc_info=True)
            await self._send_error(send, 502, f"Failed to establish tunnel: {str(e)}")

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

    async def _send_error(self, send: Send, status: int, message: str) -> None:
        """Send error response."""
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain"),
                (b"connection", b"close")
            ]
        })
        await send({
            "type": "http.response.body",
            "body": message.encode()
        })
