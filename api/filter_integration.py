"""
Integration of the filter API with the main API router.

This module registers the filter API endpoints with the main API router.
"""

from fastapi import APIRouter

from api.filter import router as filter_router

def register_filter_api(api_router: APIRouter) -> None:
    """Register the filter API endpoints with the main API router."""
    api_router.include_router(filter_router)
