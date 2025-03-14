"""Proxy server API module."""
import logging
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional
from pathlib import Path

from api.config import Settings
from database import get_db
from api.proxy import router as proxy_router
from api.proxy.websocket import router as websocket_router

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_proxy_app() -> FastAPI:
    """Create FastAPI application for proxy server."""
    settings = Settings()
    
    app = FastAPI(
        title="Anarchy Copilot Proxy API",
        description="API for controlling the proxy server",
        version="0.1.0"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add proxy-specific routes
    app.include_router(proxy_router, prefix="/api/proxy")
    app.include_router(websocket_router, prefix="/api/proxy/websocket")
    
    # Add health check endpoint
    @app.get("/api/health")
    async def health_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
        """Health check endpoint."""
        try:
            # Simple database check without creating new connections
            await db.execute("SELECT 1")
            
            # Simple file existence check without creating new instances
            ca_cert_exists = Path(settings.ca_cert_path).exists()
            ca_key_exists = Path(settings.ca_key_path).exists()
            
            return {
                "status": "healthy",
                "database": "connected",
                "ca_cert": "present" if ca_cert_exists else "missing",
                "ca_key": "present" if ca_key_exists else "missing"
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    return app

# Create the FastAPI application instance
app = create_proxy_app() 