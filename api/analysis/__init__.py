"""
Initialize the analysis API module.

This module provides API endpoints for the Traffic Analysis Engine.
"""
from fastapi import APIRouter

from .endpoints import router as analysis_router

router = APIRouter()
router.include_router(analysis_router)
