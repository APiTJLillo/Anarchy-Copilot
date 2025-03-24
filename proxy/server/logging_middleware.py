"""Logging middleware for the proxy server."""
import logging
import time
from typing import Any, Callable, Dict, Optional, Union
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("proxy.core")

class LoggingMiddleware:
    """Log requests and responses."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process an ASGI request/response cycle with logging."""
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        path = scope.get("path", "")
        method = scope.get("method", "")
        
        # Log request
        logger.debug(
            "Request: %s %s",
            method or scope.get("type", "UNKNOWN").upper(),
            path
        )

        # Wrap send to capture response
        response_started = False
        response_status = None
        response_headers = None

        async def wrapped_send(message: Message) -> None:
            nonlocal response_started, response_status, response_headers
            
            if message["type"] == "http.response.start":
                response_started = True
                response_status = message.get("status")
                response_headers = message.get("headers")
                
            elif message["type"] == "websocket.accept":
                response_started = True
                response_status = 101  # Switching Protocols
                response_headers = message.get("headers", [])

            await send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        except Exception as e:
            logger.error("Error processing request: %s", str(e), exc_info=True)
            raise
        finally:
            duration = (time.time() - start_time) * 1000
            status_text = (
                str(response_status) if response_status is not None
                else "no status"
            )
            logger.debug(
                "Response: %s %s -> %s (%.2fms)",
                method or scope.get("type", "UNKNOWN").upper(),
                path,
                status_text,
                duration
            )

class LoggingMiddlewareFactory:
    """Factory for creating logging middleware."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Create and call logging middleware."""
        middleware = LoggingMiddleware(self.app)
        await middleware(scope, receive, send)
