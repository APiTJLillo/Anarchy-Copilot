"""Middleware for handling proxy requests."""
from typing import Optional, Callable, Awaitable, Any
import logging
from .http import ProxyResponse

logger = logging.getLogger("proxy.core")

async def proxy_middleware(request: Any, call_next: Callable[[Any], Awaitable[ProxyResponse]]) -> ProxyResponse:
    """Basic proxy middleware that logs requests and handles errors."""
    try:
        # Log the incoming request
        logger.info(f"Proxy request: {request.method} {request.url}")
        
        # Call the next handler
        response = await call_next(request)
        
        # Log the response
        logger.info(f"Proxy response: {response.status_code}")
        
        return response
        
    except Exception as e:
        # Log the error
        logger.error(f"Proxy error: {e}", exc_info=True)
        
        # Return a 500 error response
        return ProxyResponse(
            status_code=500,
            headers={"content-type": "text/plain"},
            body=b"Internal proxy error"
        )
