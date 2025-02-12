"""Proxy management endpoints."""
from typing import Dict, List, Optional, Any, Awaitable
from fastapi import APIRouter, HTTPException, Body, Depends, Path, Request
from pydantic import BaseModel, Field, validator, ConfigDict

from proxy.core import ProxyServer
from proxy.config import ProxyConfig

router = APIRouter(prefix="/proxy", tags=["proxy"])
proxy_server: Optional[ProxyServer] = None

class ProxySettings(BaseModel):
    """Proxy server configuration settings."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    host: str = Field(
        ..., 
        description="Proxy host address",
        examples=["127.0.0.1"]
    )
    port: int = Field(
        ...,
        gt=0,
        lt=65536,
        description="Proxy port",
        examples=[8080]
    )
    interceptRequests: bool = True
    interceptResponses: bool = True
    allowedHosts: List[str] = []
    excludedHosts: List[str] = []

    @validator("host")
    def validate_host(cls, v: str) -> str:
        """Validate host address."""
        v = v.strip()
        if len(v) < 1:
            raise ValueError("min_length validation failed")
        return v

def require_proxy() -> ProxyServer:
    """Assert that proxy server is running and return it."""
    global proxy_server
    if not proxy_server or not proxy_server.is_running:
        raise HTTPException(
            status_code=400,
            detail="Proxy server is not running"
        )
    return proxy_server

class TagData(BaseModel):
    """Tag request data."""
    tag: str = Field(..., description="Tag to add")

class NoteData(BaseModel):
    """Note request data."""
    note: str = Field(..., description="Note to set")

@router.get("/status")
async def get_proxy_status() -> Dict[str, Any]:
    """Get current proxy server status."""
    global proxy_server
    status = {
        "isRunning": bool(proxy_server and proxy_server.is_running),
        "interceptRequests": False,
        "interceptResponses": False,
        "allowedHosts": [],
        "excludedHosts": [],
        "history": []
    }
    
    if proxy_server and proxy_server.is_running:
        status.update({
            "interceptRequests": proxy_server.config.intercept_requests,
            "interceptResponses": proxy_server.config.intercept_responses,
            "allowedHosts": list(proxy_server.config.allowed_hosts),
            "excludedHosts": list(proxy_server.config.excluded_hosts),
            "history": proxy_server.session.get_history()
        })
    return status

@router.post("/start", status_code=201)
async def start_proxy(settings: ProxySettings = Body(...)) -> Dict[str, str]:
    """Start proxy server with given settings."""
    global proxy_server

    # Stop existing server if running
    if proxy_server and proxy_server.is_running:
        await proxy_server.stop()
    
    # Create and configure new proxy server
    config = ProxyConfig(
        host=settings.host,
        port=settings.port,
        intercept_requests=settings.interceptRequests,
        intercept_responses=settings.interceptResponses,
        allowed_hosts=set(settings.allowedHosts),
        excluded_hosts=set(settings.excludedHosts)
    )

    proxy_server = ProxyServer(config)
    
    try:
        await proxy_server.start()
        return {"status": "success"}
    except Exception as e:
        proxy_server = None
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start proxy server: {str(e)}"
        )

@router.post("/stop")
async def stop_proxy() -> Dict[str, str]:
    """Stop proxy server."""
    server = require_proxy()
    try:
        await server.stop()
        global proxy_server
        proxy_server = None
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start proxy server: {str(e)}"
        )

@router.post("/settings")
async def update_settings(
    settings: ProxySettings = Body(...),
    server: ProxyServer = Depends(require_proxy)
) -> Dict[str, str]:
    """Update proxy server settings."""
    server.config.intercept_requests = settings.interceptRequests
    server.config.intercept_responses = settings.interceptResponses
    server.config.allowed_hosts = set(settings.allowedHosts)
    server.config.excluded_hosts = set(settings.excludedHosts)
    return {"status": "success"}

@router.get("/history/{entry_id}")
async def get_history_entry(
    entry_id: str = Path(...),
    server: ProxyServer = Depends(require_proxy)
) -> Dict[str, Any]:
    """Get details for a specific history entry."""
    entry = server.session.find_entry(entry_id)
    if not entry:
        raise HTTPException(
            status_code=404,
            detail=f"History entry {entry_id} not found"
        )
    
    return {
        "id": entry.id,
        "timestamp": entry.timestamp.isoformat(),
        "request": entry.request.to_dict(),
        "response": entry.response.to_dict(),
        "duration": entry.duration,
        "tags": entry.tags,
        "notes": entry.notes
    }

async def validate_tag_data(
    request: Request,
) -> Dict[str, str]:
    """Validate tag data body before other checks."""
    try:
        body = await request.json()
        return TagData.model_validate(body).model_dump()
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=[{
                "loc": ["body", "tag"],
                "msg": "field required",
                "type": "value_error.missing"
            }]
        )

async def validate_note_data(
    request: Request,
) -> Dict[str, str]:
    """Validate note data body before other checks."""
    try:
        body = await request.json()
        return NoteData.model_validate(body).model_dump()
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=[{
                "loc": ["body", "note"],
                "msg": "field required",
                "type": "value_error.missing"
            }]
        )

@router.post("/history/{entry_id}/tags")
async def add_entry_tag(
    data: Dict = Depends(validate_tag_data),
    entry_id: str = Path(...),
    server: ProxyServer = Depends(require_proxy),
) -> Dict[str, str]:
    """Add a tag to a history entry."""
    if not server.session.add_entry_tag(entry_id, data["tag"]):
        raise HTTPException(
            status_code=404,
            detail=f"History entry {entry_id} not found"
        )
    return {"status": "success"}

@router.post("/history/{entry_id}/notes")
async def set_entry_note(
    data: Dict = Depends(validate_note_data),
    entry_id: str = Path(...),
    server: ProxyServer = Depends(require_proxy),
) -> Dict[str, str]:
    """Set a note on a history entry."""
    if not server.session.set_entry_note(entry_id, data["note"]):
        raise HTTPException(
            status_code=404,
            detail=f"History entry {entry_id} not found"
        )
    return {"status": "success"}

@router.post("/history/clear")
async def clear_history(
    server: ProxyServer = Depends(require_proxy)
) -> Dict[str, str]:
    """Clear proxy history."""
    server.session.clear_history()
    return {"status": "success"}
