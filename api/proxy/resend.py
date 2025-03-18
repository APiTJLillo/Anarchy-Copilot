"""Endpoint for modifying and resending proxy requests."""
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
import httpx
from pydantic import BaseModel

from api.deps import get_db, get_current_user
from models.proxy import ModifiedRequest, ProxyHistoryEntry, ProxySession
from models.user import User

router = APIRouter()

class ResendRequestBody(BaseModel):
    original_request_id: int
    session_id: int
    method: str
    url: str
    headers: Optional[Dict[str, str]] = None
    body: Optional[str] = None

@router.post("/resend", response_model=Dict[str, Any])
async def resend_request(
    request_data: ResendRequestBody = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Modify and resend a proxy request."""
    # Get original request
    original_request = db.query(ProxyHistoryEntry).filter(
        ProxyHistoryEntry.id == request_data.original_request_id
    ).first()
    
    if not original_request:
        raise HTTPException(status_code=404, detail="Original request not found")
    
    # Determine which fields were modified
    modified_fields = []
    if request_data.method != original_request.method:
        modified_fields.append("method")
    if request_data.url != original_request.url:
        modified_fields.append("url")
    if request_data.headers != original_request.request_headers:
        modified_fields.append("headers")
    if request_data.body != original_request.request_body:
        modified_fields.append("body")
    
    # Create modified request record
    modified_request = ModifiedRequest(
        original_request_id=request_data.original_request_id,
        session_id=request_data.session_id,
        method=request_data.method,
        url=request_data.url,
        request_headers=request_data.headers,
        request_body=request_data.body,
        modified_fields=modified_fields,
        created_by=current_user.id
    )
    db.add(modified_request)
    
    try:
        # Send modified request
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request_data.method,
                url=request_data.url,
                headers=request_data.headers,
                content=request_data.body,
            )
            
            # Update response data
            modified_request.response_status = response.status_code
            modified_request.response_headers = dict(response.headers)
            modified_request.response_body = response.text
            modified_request.sent_at = datetime.now(timezone.utc)
            db.commit()
            
            return {
                "id": modified_request.id,
                "original_request_id": modified_request.original_request_id,
                "status": response.status_code,
                "headers": dict(response.headers),
                "body": response.text
            }
            
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to send request: {str(e)}")
