"""History management for proxy requests."""
import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from .database_models import ProxyHistoryEntry, ProxySession
from .connection import connection_manager

logger = logging.getLogger("proxy.core")

async def create_history_entry(
    db: AsyncSession,
    session_id: int,
    method: str,
    url: str,
    host: Optional[str] = None,
    path: Optional[str] = None,
    request_headers: Optional[Dict[str, Any]] = None,
    request_body: Optional[str] = None,
    response_headers: Optional[Dict[str, Any]] = None,
    response_body: Optional[str] = None,
    status_code: Optional[int] = None,
    duration: Optional[float] = None,
    is_intercepted: bool = True,
    is_encrypted: bool = False,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None
) -> ProxyHistoryEntry:
    """Create a new history entry and broadcast it to WebSocket clients."""
    try:
        # Create the history entry
        entry = ProxyHistoryEntry(
            session_id=session_id,
            method=method,
            url=url,
            host=host,
            path=path,
            request_headers=request_headers or {},
            request_body=request_body,
            response_headers=response_headers or {},
            response_body=response_body,
            status_code=status_code,
            duration=duration,
            is_intercepted=is_intercepted,
            is_encrypted=is_encrypted,
            tags=tags or [],
            notes=notes,
            timestamp=datetime.utcnow()
        )
        
        # Add to database
        db.add(entry)
        await db.commit()
        await db.refresh(entry)
        
        # Broadcast to WebSocket clients
        try:
            # Convert entry to dict for broadcasting
            entry_dict = {
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
                "tags": entry.tags,
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
            }
            await connection_manager.broadcast_history_update(entry_dict)
        except Exception as e:
            logger.error(f"Failed to broadcast history update: {e}")
        
        return entry
        
    except Exception as e:
        logger.error(f"Failed to create history entry: {e}")
        raise

async def get_history_entries(
    db: AsyncSession,
    session_id: Optional[int] = None,
    limit: int = 100
) -> List[ProxyHistoryEntry]:
    """Get history entries, optionally filtered by session."""
    try:
        # Build query
        stmt = (
            select(ProxyHistoryEntry)
            .order_by(ProxyHistoryEntry.timestamp.desc())
            .limit(limit)
        )
        if session_id:
            stmt = stmt.where(ProxyHistoryEntry.session_id == session_id)
            
        # Execute query
        result = await db.execute(stmt)
        entries = list(result.scalars().all())
        
        # Process entries
        for entry in entries:
            # Ensure headers are dictionaries
            if entry.request_headers is None:
                entry.request_headers = {}
            if entry.response_headers is None:
                entry.response_headers = {}
            
            # Ensure tags is a list
            if entry.tags is None:
                entry.tags = []
                
        return entries
        
    except Exception as e:
        logger.error(f"Failed to get history entries: {e}")
        raise

async def clear_history(db: AsyncSession, session_id: Optional[int] = None) -> None:
    """Clear history entries, optionally filtered by session."""
    try:
        if session_id:
            await db.execute(
                """DELETE FROM proxy_history WHERE session_id = :session_id""",
                {"session_id": session_id}
            )
        else:
            await db.execute("""DELETE FROM proxy_history""")
            
        await db.commit()
        
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        raise 