"""Project management API module."""
from fastapi import APIRouter

router = APIRouter(prefix="/api/projects", tags=["projects"])

from . import endpoints

__all__ = ["router"]
