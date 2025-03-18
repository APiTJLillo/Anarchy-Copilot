"""WebSocket endpoints for proxy connection monitoring."""
from typing import List, Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect, APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from database.session import get_db
from .models import ConnectionInfo
from .endpoints import get_history
import asyncio
import logging
from proxy.server.state import proxy_state
from .database_models import ProxySession, ProxyHistoryEntry
from .history import get_history_entries
from .connection import connection_manager

logger = logging.getLogger("proxy.core")

# Create router with prefix
router = APIRouter()

async def handle_proxy_connection_updates(websocket: WebSocket, db: AsyncSession):
    """Handle WebSocket connections for proxy connection updates."""
    try:
        await connection_manager.connect(websocket)
        
        while websocket.client_state.value == 1:  # Only continue if connection is open
            try:
                # Keep connection alive and watch for state changes
                data = await websocket.receive_text()
                
                # Verify connection is still open before sending
                if websocket.client_state.value == 1:
                    # Send current state
                    connections = await proxy_state.get_all_connections()
                    history_entries = await get_history(limit=100, db=db)
                    
                    # Convert snake_case to camelCase for frontend
                    formatted_connections = {
                        conn_id: {
                            "id": conn_id,
                            "interceptRequests": conn_data.get("intercept_requests", True),
                            "interceptResponses": conn_data.get("intercept_responses", True),
                            "allowedHosts": conn_data.get("allowed_hosts", []),
                            "excludedHosts": conn_data.get("excluded_hosts", []),
                            **{k: v for k, v in conn_data.items() if k not in ["intercept_requests", "intercept_responses", "allowed_hosts", "excluded_hosts"]}
                        }
                        for conn_id, conn_data in connections.items()
                    }
                    
                    await websocket.send_json({
                        "type": "stateUpdate",
                        "data": {
                            "connections": formatted_connections,
                            "history": history_entries
                        }
                    })
                
            except WebSocketDisconnect:
                logger.debug("WebSocket client disconnected normally")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                break
                
    except Exception as e:
        logger.error(f"Error in WebSocket connection handler: {e}")
        
    finally:
        # Always ensure we clean up
        connection_manager.disconnect(websocket)
        try:
            if websocket.client_state.value == 1:
                await websocket.close()
        except:
            pass

# Add the WebSocket endpoints to the router
@router.websocket("/proxy")
async def proxy_endpoint(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    """WebSocket endpoint for real-time connection updates."""
    await handle_proxy_connection_updates(websocket, db)

@router.websocket("/intercept")
async def intercept_endpoint(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    """WebSocket endpoint for real-time interception."""
    await handle_proxy_connection_updates(websocket, db)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    """WebSocket endpoint for general proxy updates."""
    await connection_manager.connect(websocket)
    try:
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
        initial_state = {
            "type": "initial_data",
            "data": {
                "wsConnected": True,
                "interceptRequests": False,
                "interceptResponses": False,
                "history": history_data
            }
        }
        await websocket.send_json(initial_state)

        while True:
            try:
                # Wait for any client messages
                data = await websocket.receive_json()
                # Handle client messages if needed
            except Exception as e:
                print(f"Error in websocket: {e}")
                break
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

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
