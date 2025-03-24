"""Request handlers for the proxy server."""
import asyncio
import logging
import time
from typing import Optional, Union, Any, Dict, Callable
from fastapi import Request
from fastapi.responses import Response
from starlette.responses import StreamingResponse
from starlette.websockets import WebSocket

logger = logging.getLogger("proxy.core")

class ProxyResponse:
    """Wrapper for proxy responses to handle ASGI protocol."""
    
    def __init__(
        self, 
        status_code: int = 200, 
        headers: Optional[Dict[str, str]] = None,
        body: bytes = b''
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self.body = body
        self._sent_start = False

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        """ASGI application interface."""
        if not self._sent_start:
            await send({
                'type': 'http.response.start',
                'status': self.status_code,
                'headers': [
                    (k.lower().encode(), v.encode())
                    for k, v in self.headers.items()
                ]
            })
            self._sent_start = True

        await send({
            'type': 'http.response.body',
            'body': self.body,
            'more_body': False
        })

async def proxy_middleware(
    request: Request, 
    call_next: Callable,
    proxy_server: Any
) -> Union[Response, ProxyResponse]:
    """Handle proxy requests."""
    start_time = time.time()
    
    try:
        # Debug logging
        path = request.url.path
        logger.debug("Received request: %s %s", request.method, path)
        logger.debug("Headers: %s", dict(request.headers))
        
        # Define API paths that should bypass proxy
        api_paths = {"/", "/api", "/api/docs", "/api/redoc", "/api/openapi.json"}
        
        # Let API endpoints through
        if path in api_paths or path.startswith("/api/"):
            logger.debug("API request detected, bypassing proxy: %s", path)
            return await call_next(request)
            
        # Check proxy server status
        if not proxy_server or not proxy_server.is_running:
            logger.warning("Proxy server unavailable for request to: %s", path)
            error_msg = "Proxy server not initialized" if not proxy_server else "Proxy server not running"
            return ProxyResponse(
                status_code=503,
                headers={'Content-Type': 'text/plain', 'Retry-After': '5'},
                body=error_msg.encode()
            )

        try:
            # Get special handling for CONNECT/WebSocket
            if request.method == "CONNECT":
                # Handle CONNECT requests for HTTPS tunneling
                logger.debug("Handling CONNECT request for %s", request.url.path)
                await proxy_server._tunnel_manager.create_tunnel(
                    scope=request.scope,
                    receive=request.receive,
                    send=request._send
                )
                return None  # Response is handled by tunnel_manager
                
            elif request.scope.get("type") == "websocket":
                # Handle WebSocket upgrades
                logger.debug("Handling WebSocket upgrade for %s", request.url.path)
                websocket = WebSocket(scope=request.scope, receive=request.receive, send=request.send)
                await proxy_server.tunnel_manager.create_tunnel(
                    scope=request.scope,
                    receive=request.receive,
                    send=request.send,
                    websocket=websocket
                )
                return None  # Response is handled by tunnel_manager

            # Handle normal HTTP requests
            logger.debug("Handling %s request for %s", request.method, request.url.path)
            response = await proxy_server.handle_request(
                scope=request.scope,
                receive=request.receive,
                send=request.send
            )

            # Log completion
            duration = (time.time() - start_time) * 1000
            status = getattr(response, 'status_code', 'unknown')
            logger.debug(
                "Request completed: %s %s -> %s (%.2fms)",
                request.method, request.url.path, status, duration
            )

            return response

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            return ProxyResponse(
                status_code=502,
                headers={'Content-Type': 'text/plain'},
                body=f"Proxy error: {str(e)}".encode()
            )

    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(
            "Error handling request %s %s (%.2fms): %s",
            request.method,
            request.url.path,
            duration,
            str(e),
            exc_info=True
        )
        return ProxyResponse(
            status_code=500,
            headers={'Content-Type': 'text/plain'},
            body=f"Internal server error: {str(e)}".encode()
        )
