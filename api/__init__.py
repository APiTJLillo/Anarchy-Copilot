"""
FastAPI application initialization and configuration.
"""

__version__ = "0.1.0"

from typing import Dict, Any, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .health import router as health_router
from .proxy import router as proxy_router

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

    app = FastAPI(
        title="Anarchy Copilot API",
        description="API for managing bug bounty operations",
        version="0.1.0"
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for testing
        allow_credentials=False,  # Must be False when allow_origins=["*"]
        allow_methods=["*"],
        allow_headers=["*"],
        allow_origin_regex=None,
        expose_headers=["*"],
        max_age=600,
    )

    # Add CORS handler to ensure "*" is always returned
    @app.middleware("http")
    async def always_allow_origin(request, call_next):
        response = await call_next(request)
        if "access-control-allow-origin" in response.headers:
            response.headers["access-control-allow-origin"] = "*"
        return response

    # Register routers - they have their own prefixes now
    app.include_router(health_router, prefix="/api")  # Will be mounted at /api/health/*
    app.include_router(proxy_router, prefix="/api")  # Already has /api/proxy/* prefix

    # Register event handlers using decorators
    @app.on_event("startup")
    async def startup_event():
        """Initialize resources on startup."""
        # Initialize any required resources
        pass

    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on shutdown."""
        # Clean up any resources
        from .proxy import proxy_server
        if proxy_server:
            await proxy_server.stop()

    return app

# Create default application instance
app = create_app()

def get_config() -> Dict[str, Any]:
    """Get current application configuration."""
    return _app_config
