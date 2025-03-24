"""WebSocket routes for the proxy server."""
from fastapi import APIRouter
from .manager import WebSocketManager

def create_router(ws_manager: WebSocketManager) -> APIRouter:
    """Create a router with the given WebSocket manager instance."""
    router = APIRouter()
    return router
