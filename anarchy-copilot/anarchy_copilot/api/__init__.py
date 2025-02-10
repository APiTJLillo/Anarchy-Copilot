"""FastAPI application factory and configuration."""

from typing import Dict, Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import logging
import os

from . import health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Anarchy Copilot",
        description="AI-powered bug bounty suite",
        version="0.1.0",
        docs_url="/docs" if not config or config.get("DOCS_ENABLED", True) else None,
        redoc_url="/redoc" if not config or config.get("DOCS_ENABLED", True) else None,
    )

    # Load configuration
    app.state.config = {
        "DEBUG": os.getenv("DEBUG", "false").lower() == "true",
        "TESTING": os.getenv("TESTING", "false").lower() == "true",
        "API_KEY": os.getenv("API_KEY", ""),
        "ENVIRONMENT": os.getenv("ENVIRONMENT", "production"),
    }
    if config:
        app.state.config.update(config)

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if app.state.config["DEBUG"] else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Enable gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Register routers
    app.include_router(health.router, tags=["health"])

    # Register error handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle uncaught exceptions."""
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error",
                "code": "internal_error",
                "details": str(exc) if app.state.config["DEBUG"] else None
            }
        )

    # Register startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Initialize application resources."""
        logger.info("Starting Anarchy Copilot API")
        # Initialize managers and resources here

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup application resources."""
        logger.info("Shutting down Anarchy Copilot API")
        # Cleanup resources here

    # Add authentication middleware if not in test mode
    if not app.state.config["TESTING"]:
        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            """Verify API key authentication."""
            api_key = request.headers.get("X-API-Key")
            if not api_key or api_key != app.state.config["API_KEY"]:
                return JSONResponse(
                    status_code=401,
                    content={
                        "status": "error",
                        "message": "Invalid API key",
                        "code": "invalid_api_key"
                    }
                )
            return await call_next(request)

    # Add request logging in debug mode
    if app.state.config["DEBUG"]:
        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            """Log all incoming requests."""
            logger.debug(f"Request: {request.method} {request.url}")
            response = await call_next(request)
            logger.debug(f"Response: {response.status_code}")
            return response

    return app

def get_app() -> FastAPI:
    """Get the FastAPI application instance."""
    return create_app()
