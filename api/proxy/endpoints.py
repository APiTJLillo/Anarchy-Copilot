"""Proxy API endpoint handlers."""
import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, cast
from fastapi import HTTPException, Body, Depends, WebSocket, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from .utils import cleanup_port
from proxy.server.proxy_server import ProxyServer  # Import directly from module
from proxy.config import ProxyConfig
from database import get_async_session
from models.base import Project
from . import router, get_proxy_server
from .database_models import ProxySession, ProxyHistoryEntry
from .models import (
    CreateProxySession, 
    ProxySessionResponse, 
    ProxySettings, 
    InterceptedRequest,
    InterceptedResponse,
    HistoryEntryResponse
)

__all__ = [
    "get_proxy_status",
    "get_analysis_results",
    "clear_analysis_results",
    "test_endpoint",
    "stop_proxy",
    "start_proxy",
    "websocket_endpoint",
    "create_proxy_session",
    "get_project_sessions",
    "get_session",
    "stop_session",
    # Helper functions
    "get_project_by_id",
    "get_proxy_state",
    "assert_server_running",
    "require_proxy"
]

logger = logging.getLogger(__name__)

# Use get_proxy_server to access the global instance
proxy_server = get_proxy_server()

async def get_project_by_id(session: AsyncSession, project_id: int) -> Project:
    """Get project by ID or raise 404."""
    result = await session.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.post("/sessions", response_model=ProxySessionResponse)
async def create_proxy_session(
    data: CreateProxySession,
    db: AsyncSession = Depends(get_async_session)
) -> ProxySessionResponse:
    """Create a new proxy session."""
    # Verify project exists
    await get_project_by_id(db, data.project_id)
    
    # Create session
    session = ProxySession(
        name=data.name,
        description=data.description,
        project_id=data.project_id,
        settings=data.settings,
        created_by=data.created_by,
        is_active=True
    )
    
    db.add(session)
    await db.commit()
    await db.refresh(session)
    
    return ProxySessionResponse.from_db(session)

@router.get("/projects/{project_id}/sessions", response_model=List[ProxySessionResponse])
async def get_project_sessions(
    project_id: int,
    db: AsyncSession = Depends(get_async_session)
) -> List[ProxySessionResponse]:
    """Get all proxy sessions for a project."""
    # Verify project exists
    await get_project_by_id(db, project_id)
    
    # Get sessions
    result = await db.execute(
        select(ProxySession)
        .where(ProxySession.project_id == project_id)
        .order_by(ProxySession.start_time.desc())
    )
    sessions = result.scalars().all()
    
    return [ProxySessionResponse.from_db(session) for session in sessions]

@router.get("/sessions/{session_id}", response_model=ProxySessionResponse)
async def get_session(
    session_id: int,
    db: AsyncSession = Depends(get_async_session)
) -> ProxySessionResponse:
    """Get proxy session by ID."""
    result = await db.execute(
        select(ProxySession).where(ProxySession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return ProxySessionResponse.from_db(session)

@router.post("/sessions/{session_id}/stop")
async def stop_session(
    session_id: int,
    db: AsyncSession = Depends(get_async_session)
) -> Dict[str, str]:
    """Stop a proxy session."""
    # Get session
    result = await db.execute(
        select(ProxySession).where(ProxySession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    if not session.is_active:
        return {"message": "Session is already stopped"}
        
    # Stop session
    session.is_active = False
    session.end_time = datetime.utcnow()
    
    await db.commit()
    await db.refresh(session)
    
    return {"message": "Session stopped successfully"}

def get_proxy_state() -> Optional[ProxyServer]:
    """Get current proxy server instance."""
    global proxy_server
    if proxy_server and proxy_server.is_running:
        return proxy_server
    return None

def assert_server_running(server: Optional[ProxyServer]) -> ProxyServer:
    """Assert server is running and return it."""
    if not server or not server.is_running:
        raise HTTPException(
            status_code=400,
            detail="Proxy server is not running"
        )
    return cast(ProxyServer, server)  # Cast for type checker

def require_proxy() -> ProxyServer:
    """Assert that proxy server is running and return it."""
    return assert_server_running(get_proxy_state())

@router.get("/status", response_model=Dict[str, Any])
async def get_proxy_status(
    db: AsyncSession = Depends(get_async_session)
) -> Dict[str, Any]:
    """Get current proxy server status."""
    logger.debug("Proxy status endpoint called")
    
    server = get_proxy_state()
    is_running = bool(server and server.is_running)
    
    # Find active session if proxy is running
    active_session = None
    if is_running:
        result = await db.execute(
            select(ProxySession)
            .where(ProxySession.is_active == True)
            .order_by(ProxySession.start_time.desc())
            .limit(1)
        )
        active_session = result.scalar_one_or_none()
    
    status = {
        "isRunning": is_running,
        "settings": {
            "host": "127.0.0.1",
            "port": 8080,
            "interceptRequests": False,
            "interceptResponses": False,
            "allowedHosts": [],
            "excludedHosts": [],
            "maxConnections": 100,
            "maxKeepaliveConnections": 20,
            "keepaliveTimeout": 30
        },
        "session": None
    }
    
    if server and is_running:
        status["settings"].update({
            "interceptRequests": server.config.intercept_requests,
            "interceptResponses": server.config.intercept_responses,
            "allowedHosts": list(server.config.allowed_hosts),
            "excludedHosts": list(server.config.excluded_hosts)
        })
        
        if active_session:
            status["session"] = {
                "id": active_session.id,
                "name": active_session.name,
                "description": active_session.description,
                "project_id": active_session.project_id,
                "created_by": active_session.created_by,
                "start_time": active_session.start_time.isoformat() if active_session.start_time else None
            }
    
    logger.debug(f"Proxy status: {status}")
    return status

@router.get("/history", response_model=List[HistoryEntryResponse])
async def get_proxy_history(
    db: AsyncSession = Depends(get_async_session),
    limit: int = 100,
    offset: int = 0
) -> List[HistoryEntryResponse]:
    """Get proxy request/response history."""
    logger.debug("Proxy history endpoint called with limit=%d offset=%d", limit, offset)
    
    try:
        # Get history entries directly, with pagination
        query = (
            select(ProxyHistoryEntry)
            .order_by(ProxyHistoryEntry.timestamp.desc())
            .offset(offset)
            .limit(limit)
        )
        
        logger.debug("Executing query: %s", query)
        result = await db.execute(query)
        entries = result.scalars().all()
        logger.debug("Found %d history entries", len(entries))

        response_data = [HistoryEntryResponse.from_entry(entry) for entry in entries]
        logger.debug("Returning %d formatted entries", len(response_data))
        
        return response_data

    except Exception as e:
        logger.error("Error fetching proxy history: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch proxy history: {str(e)}"
        )

@router.get("/settings", response_model=Dict[str, Any])
async def get_proxy_settings(
    db: AsyncSession = Depends(get_async_session)
) -> Dict[str, Any]:
    """Get current proxy settings."""
    logger.debug("Proxy settings endpoint called")
    
    server = get_proxy_state()
    if not server or not server.is_running:
        raise HTTPException(status_code=400, detail="Proxy server is not running")
    
    settings = {
        "host": server.config.host,
        "port": server.config.port,
        "interceptRequests": server.config.intercept_requests,
        "interceptResponses": server.config.intercept_responses,
        "allowedHosts": list(server.config.allowed_hosts),
        "excludedHosts": list(server.config.excluded_hosts),
        "maxConnections": server.config.max_connections,
        "maxKeepaliveConnections": server.config.max_keepalive_connections,
        "keepaliveTimeout": server.config.keepalive_timeout
    }
    
    return settings

@router.get("/analysis/results", response_model=List[Dict[str, Any]])
async def get_analysis_results():
    """Get proxy analysis results."""
    logger.debug("Analysis results endpoint called")
    return []

@router.delete("/analysis/results")
async def clear_analysis_results():
    """Clear all analysis results."""
    logger.debug("Clear analysis results endpoint called")
    return {"message": "Analysis results cleared successfully"}

@router.get("/test")
async def test_endpoint() -> Dict[str, str]:
    """Test endpoint to verify routing."""
    logger.debug("Test endpoint called")
    return {"status": "ok", "message": "Test endpoint working"}

@router.post("/stop", status_code=201)
async def stop_proxy(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session)
) -> Dict[str, str]:
    """Stop proxy server and end active session."""
    server = get_proxy_state()
    
    if not server:
        return {"message": "Proxy server is already stopped"}

    # End active session if any
    result = await db.execute(
        select(ProxySession)
        .where(ProxySession.is_active == True)
        .order_by(ProxySession.start_time.desc())
        .limit(1)
    )
    active_session = result.scalar_one_or_none()
    
    if active_session:
        active_session.is_active = False
        active_session.end_time = datetime.utcnow()
        await db.commit()

    global proxy_server
    server_ref = server
    proxy_server = None

    async def do_stop(server: ProxyServer) -> None:
        try:
            await asyncio.wait_for(server.stop(), timeout=5.0)
            logger.info("Proxy server stopped successfully")
        except Exception as e:
            logger.warning(f"Non-critical error during stop: {e}")

    background_tasks.add_task(do_stop, server_ref)
    
    message = "Proxy server stopped successfully"
    if active_session:
        message += f" and session {active_session.id} ended"
    
    return {"message": message}

@router.post("/sessions/{session_id}/start", status_code=201)
async def start_proxy(
    session_id: int,
    settings: ProxySettings = Body(...),
    db: AsyncSession = Depends(get_async_session)
) -> Dict[str, str]:
    """Start proxy server for a session."""
    global proxy_server
    
    # Get session
    result = await db.execute(
        select(ProxySession).where(ProxySession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.is_active:
        raise HTTPException(status_code=400, detail="Cannot start proxy for inactive session")
    
    # Stop existing proxy if running
    if get_proxy_state():
        await stop_proxy(BackgroundTasks())
    
    # Ensure port is free
    max_cleanup_retries = 3
    for attempt in range(max_cleanup_retries):
        if await cleanup_port(settings.port, logger):
            break
        if attempt == max_cleanup_retries - 1:
            raise HTTPException(status_code=500, detail=f"Failed to free port {settings.port}")
        await asyncio.sleep(2)
    
    try:
        logger.debug(f"Starting proxy with settings: {settings}")
        ca_cert_path = os.getenv("CA_CERT_PATH")
        ca_key_path = os.getenv("CA_KEY_PATH")
        
        if not ca_cert_path or not ca_key_path:
            raise ValueError("CA_CERT_PATH and CA_KEY_PATH must be set in the environment variables.")
        
        config = ProxyConfig(
            host=settings.host,
            port=settings.port,
            intercept_requests=settings.interceptRequests,
            intercept_responses=settings.interceptResponses,
            excluded_hosts=set(settings.excludedHosts),
            max_connections=settings.maxConnections,
            max_keepalive_connections=settings.maxKeepaliveConnections,
            keepalive_timeout=settings.keepaliveTimeout,
            ca_cert_path=Path(os.getenv("CA_CERT_PATH")),
            ca_key_path=Path(os.getenv("CA_KEY_PATH"))
        )

        # Store settings in session
        session.settings = settings.dict()
        await db.commit()
        
        # Create and start the proxy server
        proxy_server = ProxyServer(config, add_default_interceptors=False)
        logger.debug("Proxy server instance created")
        
        # Start the proxy server with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Starting proxy server on {config.host}:{config.port} (attempt {attempt + 1})")
                await proxy_server.start()
                logger.info(f"Proxy server started successfully on {config.host}:{config.port}")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to start proxy server after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Failed to start proxy server (attempt {attempt + 1}): {e}")
                await asyncio.sleep(2)
        logger.info(f"Proxy server started successfully for session {session_id}")
        return {"message": "Proxy server started successfully", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"Failed to start proxy: {e}", exc_info=True)
        proxy_server = None
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/intercept")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time proxy interception."""
    await websocket.accept()
    try:
        await websocket.send_json({"type": "connected"})
        
        while True:
            data = await websocket.receive_json()
            if "type" not in data:
                await websocket.send_json({"type": "error", "message": "Missing type"})
                continue
                
            try:
                if data["type"] == "request":
                    await websocket.send_json({
                        "type": "request_processed",
                        "requestId": data.get("requestId"),
                        "success": True
                    })
                elif data["type"] == "response":
                    await websocket.send_json({
                        "type": "response_processed",
                        "requestId": data.get("requestId"),
                        "success": True
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown type: {data['type']}"
                    })
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if not websocket.client_state.DISCONNECTED:
            await websocket.close(code=1011, reason=str(e))
    finally:
        if not websocket.client_state.DISCONNECTED:
            await websocket.close()
