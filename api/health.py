"""Health check and version information endpoints."""
from typing import Dict
from fastapi import APIRouter
from version import __version__

router = APIRouter()

@router.get("/health", tags=["system"])
async def health_check() -> Dict[str, str]:
    """Get system health status and version information."""
    return {
        "status": "healthy",
        "version": __version__,
        "api": "online"
    }

@router.get("/version", tags=["system"])
async def version_info() -> Dict[str, str]:
    """Get detailed version information."""
    return {
        "version": __version__,
        "name": "Anarchy Copilot",
        "api_compatibility": f"^{__version__.split('.')[0]}.0.0"
    }
