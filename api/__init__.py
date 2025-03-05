"""
FastAPI application initialization and configuration.
"""

from version import __version__

import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Global config storage
_app_config: Dict[str, Any] = {}

def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    from .config import Settings
    
    # Import version info at runtime to avoid circular imports
    version = config.get("version", __version__) if config else __version__

    # Initialize settings with environment variables
    settings = Settings()

    # Store settings globally
    global _app_config
    _app_config.update(settings.model_dump())

    # Apply any override config and store globally
    if config:
        settings_dict = dict(config)
        for key, value in settings_dict.items():
            if key == "cors_origins":
                # Update input instead of computed property
                settings.cors_origins_input = ",".join(value if isinstance(value, list) else [value])
            elif hasattr(settings, key):
                setattr(settings, key, value)
        # Update global config after modifying settings
        _app_config.update(settings.model_dump())

    app = FastAPI(
        title=settings.api_title,
        description=f"API for managing bug bounty operations (Version {version})",
        version=settings.api_version,
        debug=settings.debug,
        openapi_tags=[{"name": "version", "description": version}]
    )

    logger.debug(f"Creating app with config: {_app_config}")

    # Configure CORS middleware
    logger.info(f"Configuring CORS with origins: {settings.cors_origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=600,
    )

    logger.debug("CORS middleware configured")

    # Add security headers handler
    @app.middleware("http")
    async def add_security_headers(request, call_next):
        try:
            response = await call_next(request)
            
            if isinstance(response, Response):
                # Disable HTTPS enforcement headers for development
                response.headers.update({
                    "strict-transport-security": "max-age=0",
                    "x-content-security-policy": "upgrade-insecure-requests 0",
                    "content-security-policy": "upgrade-insecure-requests 0"
                })
                
                # Add CORS headers for preflight
                if request.method == "OPTIONS":
                    origin = request.headers.get("origin")
                    if origin and origin in settings.cors_origins:
                        response.headers.update({
                            "access-control-allow-origin": origin,
                            "access-control-allow-methods": "*",
                            "access-control-allow-headers": "*",
                            "access-control-allow-credentials": "true",
                            "access-control-max-age": "600",
                        })
            
            return response

        except Exception as e:
            logger.error(f"Middleware error: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": str(e)}
            )

    # Import and include routers
    from .proxy.endpoints import router as proxy_router
    from .health import router as health_router
    from .ai.settings import router as ai_router
    
    app.include_router(proxy_router, prefix="/api/proxy")
    app.include_router(health_router, prefix="/api", tags=["system"])
    app.include_router(ai_router, prefix="/api", tags=["ai"])

    @app.on_event("startup")
    async def startup_event():
        """Initialize resources on startup."""
        # Initialize proxy settings
        settings = Settings()
        try:
            from proxy.server import proxy_server
            proxy_server.configure({
                'host': settings.proxy_host,
                'port': settings.proxy_port,
                'max_connections': settings.proxy_max_connections,
                'max_keepalive_connections': settings.proxy_max_keepalive_connections,
                'keepalive_timeout': settings.proxy_keepalive_timeout,
            })
            logger.info("Initialized proxy settings")
        except Exception as e:
            logger.error(f"Failed to initialize proxy settings: {e}")

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

__all__ = ['create_app', 'get_config', 'Settings']
