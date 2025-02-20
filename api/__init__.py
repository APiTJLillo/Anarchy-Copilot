"""
FastAPI application initialization and configuration.
"""

__version__ = "0.1.0"

import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Global config storage
_app_config: Dict[str, Any] = {}

def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create and configure the FastAPI application.
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        Configured FastAPI application instance
    """
    global _app_config
    
    if config:
        _app_config.update(config)

    # Set up FastAPI with configuration
    default_config = {
        "title": "Anarchy Copilot API",
        "description": "API for managing bug bounty operations",
        "version": "0.1.0",
        "debug": False,
        "cors_origins": ["http://localhost:3000"]
    }

    if config:
        default_config.update(config)

    app = FastAPI(
        title=default_config["title"],
        description=default_config["description"],
        version=default_config["version"],
        debug=default_config["debug"]
    )

    logger.debug(f"Creating app with config: {default_config}")

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=default_config["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=600,
    )

    logger.debug("CORS middleware configured")

    # Add CORS and security headers handler
    @app.middleware("http")
    async def always_allow_origin(request, call_next):
        try:
            # Get response and handle None case
            response: Optional[Response] = None
            try:
                response = await call_next(request)
            except Exception as e:
                logger.error(f"Error in route handler: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"detail": f"Internal server error: {str(e)}"}
                )

            # Handle missing response
            if not response:
                return JSONResponse(
                    status_code=500,
                    content={"detail": "No response returned from handler"}
                )
            
            # Add security headers
            if isinstance(response, Response):
                # Disable HTTPS enforcement headers
                response.headers.update({
                    "strict-transport-security": "max-age=0",
                    "x-content-security-policy": "upgrade-insecure-requests 0",
                    "content-security-policy": "upgrade-insecure-requests 0"
                })

            return response

        except Exception as e:
            # Handle unexpected middleware errors
            logger.error(f"Unexpected middleware error: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Middleware error: {str(e)}"}
            )

    # Import and include the proxy router
    from .proxy.endpoints import router as proxy_router
    app.include_router(proxy_router, prefix="/api/proxy")

    @app.on_event("startup")
    async def startup_event():
        """Initialize resources on startup."""
        # Initialize any required resources
        pass

    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on shutdown."""
        # Clean up any resources
        # Import proxy_server lazily to avoid circular imports
        try:
            from .proxy import proxy_server
            if proxy_server:
                await proxy_server.stop()
        except ImportError:
            pass

    return app

def get_config() -> Dict[str, Any]:
    """Get current application configuration."""
    return _app_config

__all__ = ['create_app', 'get_config']
