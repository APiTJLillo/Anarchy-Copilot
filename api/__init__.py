"""
FastAPI application initialization and configuration.
"""

from version import __version__

import logging
import os
from pathlib import Path
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
                # Add CORS headers for WebSocket upgrade requests
                if request.headers.get("upgrade", "").lower() == "websocket":
                    origin = request.headers.get("origin")
                    if origin and origin in settings.cors_origins:
                        response.headers.update({
                            "access-control-allow-origin": origin,
                            "access-control-allow-methods": "*",
                            "access-control-allow-headers": "*",
                            "access-control-allow-credentials": "true",
                            "access-control-max-age": "600",
                            "access-control-expose-headers": "*"
                        })
                
                # Disable HTTPS enforcement headers for development
                response.headers.update({
                    "strict-transport-security": "max-age=0",
                    "x-content-security-policy": "upgrade-insecure-requests 0",
                    "content-security-policy": "upgrade-insecure-requests 0"
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
    from .proxy.websocket import router as websocket_router
    from .ai.settings import router as ai_router
    
    app.include_router(proxy_router, prefix="/api/proxy")
    app.include_router(websocket_router, prefix="/api/proxy")  # Mount WebSocket router under /api/proxy
    app.include_router(ai_router, prefix="/api", tags=["ai"])

    @app.on_event("startup")
    async def startup_event():
        """Initialize resources on startup."""
        # Initialize proxy settings
        settings = Settings()
        try:
            # Docker container paths should be checked first
            docker_paths = [
                # Docker container path - this is where certificates are mounted in Docker
                (Path("/app/certs/ca.crt"), Path("/app/certs/ca.key")),
                # Alternative paths that might be used in Docker
                (Path("/certs/ca.crt"), Path("/certs/ca.key")),
            ]
            
            # Local development paths
            base_dir = Path(__file__).parent.parent
            local_paths = [
                (base_dir / "certs" / "ca.crt", base_dir / "certs" / "ca.key"),
                (base_dir / "ca.crt", base_dir / "ca.key"),
                (base_dir / "test_ca.crt", base_dir / "test_ca.key"),
            ]
            
            # Combine all possible paths, checking Docker paths first
            cert_locations = docker_paths + local_paths
            
            ca_cert_path = None
            ca_key_path = None
            
            # Find the first existing pair of certificate files
            for cert_path, key_path in cert_locations:
                if cert_path.exists() and key_path.exists():
                    ca_cert_path = cert_path
                    ca_key_path = key_path
                    logger.info(f"Using CA certificate files: {ca_cert_path}, {ca_key_path}")
                    break
            
            if ca_cert_path is None or ca_key_path is None:
                logger.warning("CA certificate files not found. HTTPS interception will be disabled.")
                # Check permissions on directories where we expected certificates
                for dir_path in [Path("/app/certs"), Path("/certs"), base_dir / "certs"]:
                    if dir_path.exists():
                        try:
                            logger.info(f"Directory {dir_path} exists, checking contents:")
                            for item in dir_path.iterdir():
                                logger.info(f"  - {item}")
                        except PermissionError:
                            logger.warning(f"Permission denied when checking directory: {dir_path}")
            
            # Configure proxy server with CA certificate paths if found
            from proxy.server import proxy_server
            proxy_config = {
                'host': settings.proxy_host,
                'port': settings.proxy_port,
                'max_connections': settings.proxy_max_connections,
                'max_keepalive_connections': settings.proxy_max_keepalive_connections,
                'keepalive_timeout': settings.proxy_keepalive_timeout,
            }
            
            # Add CA certificate paths if found
            if ca_cert_path and ca_key_path:
                proxy_config['ca_cert_path'] = ca_cert_path
                proxy_config['ca_key_path'] = ca_key_path
                logger.info("HTTPS interception will be enabled with the provided CA certificates")
            
            proxy_server.configure(proxy_config)
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
