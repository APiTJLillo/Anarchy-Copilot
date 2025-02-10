"""Health check endpoints for API monitoring."""

from typing import Dict, Any
from fastapi import APIRouter, Response, status
import psutil
import os
from datetime import datetime, timezone

from ..vuln_module.vuln_manager import VulnManager
from ..version import __version__

router = APIRouter()

def get_system_status() -> Dict[str, Any]:
    """Get system resource usage information."""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent
        },
        "disk": {
            "total": disk.total,
            "free": disk.free,
            "percent": disk.percent
        }
    }

def get_application_status() -> Dict[str, Any]:
    """Get application-specific status information."""
    return {
        "version": __version__,
        "pid": os.getpid(),
        "process": {
            "memory": psutil.Process().memory_info().rss,
            "threads": psutil.Process().num_threads()
        },
        "start_time": datetime.fromtimestamp(
            psutil.Process().create_time(),
            tz=timezone.utc
        ).isoformat()
    }

def check_dependencies() -> Dict[str, bool]:
    """Check if required external tools are available."""
    dependencies = {
        "nuclei": False
    }
    
    try:
        # Check nuclei
        result = os.system("nuclei -version >/dev/null 2>&1")
        dependencies["nuclei"] = result == 0
    except Exception:
        pass
    
    return dependencies

@router.get("/health")
async def health_check(response: Response) -> Dict[str, Any]:
    """Basic health check endpoint."""
    dependencies = check_dependencies()
    all_dependencies_ok = all(dependencies.values())
    
    # Set appropriate status code
    response.status_code = (
        status.HTTP_200_OK if all_dependencies_ok
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    
    return {
        "status": "healthy" if all_dependencies_ok else "unhealthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": dependencies
    }

@router.get("/health/details")
async def detailed_health_check(response: Response) -> Dict[str, Any]:
    """Detailed health check with system and application metrics."""
    dependencies = check_dependencies()
    all_dependencies_ok = all(dependencies.values())
    
    # Set appropriate status code
    response.status_code = (
        status.HTTP_200_OK if all_dependencies_ok
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    
    return {
        "status": "healthy" if all_dependencies_ok else "unhealthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": get_system_status(),
        "application": get_application_status(),
        "dependencies": dependencies,
        "managers": {
            "vulnerability": VulnManager().get_status()
        }
    }

@router.get("/health/live")
async def liveness_probe() -> Dict[str, str]:
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}

@router.get("/health/ready")
async def readiness_probe(response: Response) -> Dict[str, Any]:
    """Kubernetes readiness probe endpoint."""
    dependencies = check_dependencies()
    all_dependencies_ok = all(dependencies.values())
    
    response.status_code = (
        status.HTTP_200_OK if all_dependencies_ok
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    
    return {
        "status": "ready" if all_dependencies_ok else "not_ready",
        "dependencies": dependencies
    }
