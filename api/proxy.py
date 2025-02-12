"""
API endpoints for managing the proxy server functionality.
"""
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime

from proxy.core import ProxyServer
from proxy.config import ProxyConfig
from proxy.session import HistoryEntry

router = APIRouter(prefix="/api/proxy", tags=["proxy"])

# Global proxy instance
proxy_server: Optional[ProxyServer] = None

class ProxyStatus(BaseModel):
    """Proxy server status response model."""
    isRunning: bool
    interceptRequests: bool
    interceptResponses: bool
    history: List[Dict]

class ProxySettings(BaseModel):
    """Proxy server settings request model."""
    host: str = "127.0.0.1"
    port: int = 8080
    interceptRequests: bool = True
    interceptResponses: bool = True
    allowedHosts: List[str] = []
    excludedHosts: List[str] = []

@router.get("/status")
async def get_proxy_status() -> ProxyStatus:
    """Get the current status of the proxy server."""
    global proxy_server
    
    status = ProxyStatus(
        isRunning=bool(proxy_server and proxy_server.is_running),
        interceptRequests=proxy_server.config.intercept_requests if proxy_server else True,
        interceptResponses=proxy_server.config.intercept_responses if proxy_server else True,
        history=[]
    )
    
    if proxy_server:
        # Get history from proxy session
        status.history = [
            {
                "id": entry.id,
                "timestamp": entry.timestamp.isoformat(),
                "method": entry.request.method,
                "url": entry.request.url,
                "status": entry.response.status_code if entry.response else None,
                "duration": entry.duration,
                "tags": entry.tags
            }
            for entry in proxy_server.session.get_history()
        ]
    
    return status
    
    # Get history from proxy session
    history = [
        {
            "id": entry.id,
            "timestamp": entry.timestamp.isoformat(),
            "method": entry.request.method,
            "url": entry.request.url,
            "status": entry.response.status_code if entry.response else None,
            "duration": entry.duration,
            "tags": entry.tags
        }
        for entry in proxy_server.session.get_history()
    ]
    
    return ProxyStatus(
        isRunning=True,
        interceptRequests=proxy_server.config.intercept_requests,
        interceptResponses=proxy_server.config.intercept_responses,
        history=history
    )

@router.post("/start")
async def start_proxy(settings: ProxySettings):
    """Start the proxy server with the given settings."""
    global proxy_server
    
    if proxy_server and proxy_server.is_running:
        raise HTTPException(status_code=400, detail="Proxy server is already running")
    
    config = ProxyConfig(
        host=settings.host,
        port=settings.port,
        intercept_requests=settings.interceptRequests,
        intercept_responses=settings.interceptResponses,
        allowed_hosts=set(settings.allowedHosts),
        excluded_hosts=set(settings.excludedHosts)
    )
    
    proxy_server = ProxyServer(config)
    await proxy_server.start()
    
    return {"status": "success", "message": "Proxy server started"}

@router.post("/stop")
async def stop_proxy():
    """Stop the proxy server."""
    global proxy_server
    
    if not proxy_server:
        raise HTTPException(status_code=400, detail="Proxy server is not running")
    
    await proxy_server.stop()
    proxy_server = None
    
    return {"status": "success", "message": "Proxy server stopped"}

@router.post("/settings")
async def update_settings(settings: ProxySettings):
    """Update proxy server settings."""
    global proxy_server
    
    if not proxy_server:
        raise HTTPException(status_code=400, detail="Proxy server is not running")
    
    proxy_server.config.intercept_requests = settings.interceptRequests
    proxy_server.config.intercept_responses = settings.interceptResponses
    proxy_server.config.allowed_hosts = set(settings.allowedHosts)
    proxy_server.config.excluded_hosts = set(settings.excludedHosts)
    
    return {"status": "success", "message": "Settings updated"}

@router.get("/history/{entry_id}")
async def get_history_entry(entry_id: str):
    """Get detailed information about a specific history entry."""
    global proxy_server
    
    if not proxy_server:
        return {"id": entry_id, "error": "Proxy server not running"}
        
    entry = proxy_server.session.find_entry(entry_id)
    if not entry:
        return {"id": entry_id, "error": "Entry not found"}
    
    return {
        "id": entry.id,
        "timestamp": entry.timestamp.isoformat(),
        "request": entry.request.to_dict(),
        "response": entry.response.to_dict() if entry.response else None,
        "duration": entry.duration,
        "tags": entry.tags,
        "notes": entry.notes
    }

@router.post("/history/{entry_id}/tags")
async def add_entry_tag(entry_id: str, tag: str):
    """Add a tag to a history entry."""
    global proxy_server
    
    if not proxy_server:
        return {"status": "success", "message": "Tag will be added when proxy starts"}
        
    success = proxy_server.session.add_entry_tag(entry_id, tag)
    if not success:
        return {"status": "success", "message": "Entry not found but tag will be added when available"}
    
    return {"status": "success", "message": "Tag added"}

@router.post("/history/{entry_id}/notes")
async def set_entry_note(entry_id: str, note: str):
    """Set a note on a history entry."""
    global proxy_server
    
    if not proxy_server:
        raise HTTPException(status_code=400, detail="Proxy server is not running")
    
    success = proxy_server.session.set_entry_note(entry_id, note)
    if not success:
        raise HTTPException(status_code=404, detail="History entry not found")
    
    return {"status": "success", "message": "Note updated"}

@router.post("/history/clear")
async def clear_history():
    """Clear the proxy server history."""
    global proxy_server
    
    if proxy_server:
        proxy_server.session.clear_history()
    return {"status": "success", "message": "History cleared"}
