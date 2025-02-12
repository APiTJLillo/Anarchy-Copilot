"""
Health check endpoints for the API.
"""
import os
import shutil
import psutil
import asyncio
import subprocess
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Callable
from functools import partial

router = APIRouter(prefix="/health", tags=["health"])

TIMEOUT = 5  # seconds

def get_system_status() -> Dict[str, Any]:
    """Get system resource status."""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        
        return {
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "used_percent": disk.percent,
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"System status check failed: {str(e)}"
        )

def get_application_status() -> Dict[str, Any]:
    """Get application process status."""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "pid": process.pid,
            "memory": memory_info.rss,
            "cpu_usage": process.cpu_percent()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Application status check failed: {str(e)}"
        )

def check_dependencies() -> Dict[str, Any]:
    """Check external dependencies."""
    try:
        nuclei_path = shutil.which("nuclei")
        if not nuclei_path:
            return {
                "nuclei": {
                    "installed": False,
                    "version": None,
                    "path": None
                }
            }

        # Get nuclei version
        try:
            result = subprocess.run(
                ["nuclei", "-version"],
                capture_output=True,
                text=True,
                timeout=2
            )
            version = result.stdout.strip() if result.returncode == 0 else "unknown"
        except subprocess.TimeoutExpired:
            version = "unknown"

        return {
            "nuclei": {
                "installed": True,
                "version": version,
                "path": nuclei_path
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Dependency check failed: {str(e)}"
        )

async def run_with_timeout(func: Callable, timeout: float, *args, **kwargs):
    """Run a blocking function with timeout."""
    async def execute():
        loop = asyncio.get_running_loop()
        bound_func = partial(func, *args, **kwargs)
        result = await loop.run_in_executor(None, bound_func)
        return result
        
    try:
        return await asyncio.wait_for(execute(), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=500,
            detail=f"Operation timed out after {timeout} seconds"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("")
async def health_check():
    """Basic health check endpoint."""
    def check():
        psutil.virtual_memory()  # Check memory access
        psutil.disk_usage("/")   # Check disk access
        return {"status": "healthy"}

    return await run_with_timeout(check, TIMEOUT)

@router.get("/details")
async def detailed_health_check():
    """Get detailed system health information."""
    def get_details():
        return {
            "status": "healthy",
            "system": get_system_status(),
            "application": get_application_status(),
            "dependencies": check_dependencies()
        }

    return await run_with_timeout(get_details, TIMEOUT)

@router.get("/live")
async def liveness_probe():
    """Kubernetes liveness probe endpoint.
    
    This endpoint should always return 200 unless the process is completely dead.
    """
    return {"status": "alive"}

@router.get("/ready")
async def readiness_probe():
    """Kubernetes readiness probe endpoint.
    
    Verifies that key system resources are accessible.
    """
    def check():
        psutil.virtual_memory()  # Verify memory access
        psutil.disk_usage("/")   # Verify disk access
        return {"status": "ready"}

    return await run_with_timeout(check, TIMEOUT)
