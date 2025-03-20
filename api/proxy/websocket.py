"""WebSocket endpoints for proxy connection monitoring."""
from typing import List, Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect, APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from database.session import get_db
from .models import ConnectionInfo
from .endpoints import get_history
import asyncio
import logging
from .state import proxy_state
from .database_models import ProxySession, ProxyHistoryEntry
from .history import get_history_entries
from .connection import connection_manager
from sqlalchemy import select
from datetime import datetime

logger = logging.getLogger("proxy.core")

# Create router with prefix
router = APIRouter()

@router.websocket("/ws")
async def handle_proxy_connection_updates(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    """Handle WebSocket connections from UI clients."""
    try:
        await connection_manager.connect(websocket, connection_type="ui")
        logger.debug("New UI WebSocket connection established")
        
        # Get initial history
        history_entries = await get_history_entries(db)
        history_data = []
        for entry in history_entries:
            history_data.append({
                "id": entry.id,
                "timestamp": entry.timestamp.isoformat(),
                "method": entry.method,
                "url": entry.url,
                "host": entry.host,
                "path": entry.path,
                "status_code": entry.status_code,
                "response_status": entry.status_code,
                "duration": entry.duration,
                "is_intercepted": entry.is_intercepted,
                "is_encrypted": entry.is_encrypted,
                "tags": entry.tags or [],
                "notes": entry.notes,
                "request_headers": entry.request_headers,
                "request_body": entry.request_body,
                "response_headers": entry.response_headers,
                "response_body": entry.response_body,
                "raw_request": entry.request_body,
                "raw_response": entry.response_body,
                "decrypted_request": entry.decrypted_request,
                "decrypted_response": entry.decrypted_response,
                "applied_rules": None,
                "session_id": entry.session_id
            })
        
        # Send initial state
        active_session = await db.execute(
            select(ProxySession)
            .where(ProxySession.is_active == True)
            .order_by(ProxySession.start_time.desc())
        )
        active_session = active_session.scalar()
        
        initial_state = {
            "type": "initial_data",
            "data": {
                "wsConnected": True,
                "interceptRequests": active_session.settings.get("intercept_requests", True) if active_session else True,
                "interceptResponses": active_session.settings.get("intercept_responses", True) if active_session else True,
                "history": history_data,
                "status": {
                    "isRunning": active_session is not None and active_session.is_active,
                    "settings": active_session.settings if active_session else None
                },
                "version": proxy_state.version
            }
        }
        logger.debug(f"Sending initial state with {len(history_data)} history entries")
        await websocket.send_json(initial_state)
        
        while True:
            data = await websocket.receive_json()
            logger.debug(f"Received WebSocket message: {data}")
            
            if data.get("type") == "start_proxy":
                await proxy_state.start()
            elif data.get("type") == "stop_proxy":
                await proxy_state.stop()
                
    except WebSocketDisconnect:
        logger.info("UI WebSocket client disconnected")
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)

@router.websocket("/ws/internal")
async def handle_internal_connection(websocket: WebSocket):
    """Handle WebSocket connections from proxy container."""
    try:
        await connection_manager.connect(websocket, connection_type="internal")
        logger.info("Internal WebSocket connection established with proxy container")
        
        while True:
            data = await websocket.receive_json()
            logger.debug(f"Received internal WebSocket message: {data}")
            
            # Handle test connection message
            if data.get("type") == "test_connection":
                logger.debug("Received test connection message from proxy container")
                await websocket.send_json({"type": "test_connection_response", "status": "ok"})
                continue
            
            # Forward history updates to UI clients
            if data.get("type") == "proxy_history":
                logger.debug("Broadcasting history update to UI clients")
                await connection_manager.broadcast_history_update(data["data"])
                
    except WebSocketDisconnect:
        logger.info("Proxy container WebSocket disconnected")
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Internal WebSocket error: {e}")
        connection_manager.disconnect(websocket)

# Add the WebSocket endpoints to the router
@router.websocket("/proxy")
async def proxy_endpoint(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    """WebSocket endpoint for real-time connection updates."""
    await handle_proxy_connection_updates(websocket, db)

@router.websocket("/intercept")
async def intercept_endpoint(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    """WebSocket endpoint for real-time interception."""
    await handle_proxy_connection_updates(websocket, db)

@router.websocket("/ws/intercept")
async def ws_intercept_endpoint(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    """WebSocket endpoint for interception updates."""
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Handle intercept messages
            await connection_manager.broadcast_message(data)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connection_manager.disconnect(websocket)
