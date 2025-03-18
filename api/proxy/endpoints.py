"""FastAPI endpoints for proxy functionality."""

import logging
from . import router
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

__all__ = [
    "create_session",
    "start_proxy",
    "stop_proxy",
    "get_history",
    "clear_history",
    "get_analysis_results",
    "clear_analysis_results",
    "get_proxy_status",
    "create_rule",
    "update_rule",
    "delete_rule",
    "list_rules",
    "reorder_rules",
    "get_connections",
    "health_check"
]
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from database.session import get_db
from . import models
from .database_models import (
    ProxySession,
    ProxyHistoryEntry,
    ProxyAnalysisResult,
    InterceptionRule
)
from models.proxy import ProxyHistoryEntry

import os
from functools import lru_cache
from pathlib import Path
from ..config import Settings
from sqlalchemy.future import select
from proxy.server.certificates import CertificateAuthority
from proxy.config import ProxyConfig


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Session Management
@router.post("/sessions", response_model=models.ProxySession)
async def create_session(
    data: models.CreateProxySession,
    db: AsyncSession = Depends(get_db)
) -> ProxySession:
    """Create a new proxy session."""
    try:
        # Deactivate any currently active sessions
        await db.execute(
            update(ProxySession)
            .where(ProxySession.is_active == True)
            .values(is_active=False, end_time=datetime.utcnow())
        )

        # Create new session
        # Merge settings with defaults from config
        config = get_settings()
        default_settings = {
            "host": config.proxy_host,
            "port": config.proxy_port,
            "interceptRequests": config.proxy_intercept_requests,
            "interceptResponses": config.proxy_intercept_responses,
            "maxConnections": config.proxy_max_connections,
            "maxKeepaliveConnections": config.proxy_max_keepalive_connections,
            "keepaliveTimeout": config.proxy_keepalive_timeout,
            "ca_cert_path": config.ca_cert_path,
            "ca_key_path": config.ca_key_path
        }
        # Merge settings with user-provided settings taking precedence
        merged_settings = {**default_settings, **(data.settings or {})}

        session = ProxySession(
            name=data.name,
            project_id=data.project_id,
            created_by=data.user_id,
            settings=merged_settings or {},
            is_active=True,
            start_time=datetime.utcnow()
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        return session
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Proxy Control
@router.post("/sessions/{session_id}/start")
async def start_proxy(
    session_id: int,
    settings: models.ProxySettings,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Start the proxy with the specified settings."""
    try:
            config = get_settings()
            result = await db.execute(
                select(ProxySession).where(ProxySession.id == session_id)
            )
            session = result.scalar_one_or_none()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            # Verify CA certificate files exist
            if not os.path.exists(config.ca_cert_path):
                raise HTTPException(status_code=500, detail=f"CA certificate file not found: {config.ca_cert_path}")
            if not os.path.exists(config.ca_key_path):
                raise HTTPException(status_code=500, detail=f"CA key file not found: {config.ca_key_path}")

            # Initialize CA
            logger.info(f"Initializing CA with cert_path={config.ca_cert_path}, key_path={config.ca_key_path}")
            ca = CertificateAuthority(
                ca_cert_path=Path(config.ca_cert_path),
                ca_key_path=Path(config.ca_key_path)
            )

            # Start the proxy with the configured CA
            from proxy.server.proxy_server import ProxyServer
            proxy_server = ProxyServer(config=ProxyConfig(**session.settings), ca_instance=ca)

            try:
                await proxy_server.start()
            except Exception as e:
                logger.error(f"Failed to start proxy server: {e}")
                raise HTTPException(status_code=500, detail=str(e))

            # Update session settings and status
            merged_settings = {
                **settings.model_dump(),
                "ca_cert_path": config.ca_cert_path,
                "ca_key_path": config.ca_key_path
            }
            session.is_active = True
            session.settings.update(merged_settings)
            await db.commit()

            logger.info(f"Proxy started successfully with certificate paths: {config.ca_cert_path}, {config.ca_key_path}")
            return {"status": "started", "session_id": session_id}
    except Exception as e:
        logger.error(f"Failed to start proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_proxy(db: AsyncSession = Depends(get_db)) -> dict:
    """Stop the active proxy session."""
    try:
        await db.execute(
            update(ProxySession)
            .where(ProxySession.is_active == True)
            .values(is_active=False, end_time=datetime.utcnow())
        )
        await db.commit()
        return {"status": "stopped"}
    except Exception as e:
        logger.error(f"Failed to stop proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# History Management
@router.get("/history", response_model=List[models.ProxyHistory])
async def get_history(
    session_id: Optional[int] = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
) -> List[ProxyHistoryEntry]:
    """Get proxy history entries."""
    try:
        # Build query with proper type hints
        stmt = (
            select(ProxyHistoryEntry)
            .order_by(ProxyHistoryEntry.timestamp.desc())
            .limit(limit)
        )
        if session_id:
            stmt = stmt.where(ProxyHistoryEntry.session_id == session_id)

        # Execute and handle results properly
        result = await db.execute(stmt)
        entries = list(result.scalars().all())
        
        # Process each entry to ensure proper data format
        for entry in entries:
            # Ensure headers are always dictionaries
            if entry.request_headers is None:
                entry.request_headers = {}
            if entry.response_headers is None:
                entry.response_headers = {}
            
            # Ensure tags is always a list
            if entry.tags is None:
                entry.tags = []
            elif isinstance(entry.tags, str):
                try:
                    entry.tags = json.loads(entry.tags)
                except:
                    entry.tags = [entry.tags]

            # Include decrypted data if available
            if entry.decrypted_request is not None:
                entry.request_body = entry.decrypted_request
                if 'decrypted' not in entry.tags:
                    entry.tags.append('decrypted')

            if entry.decrypted_response is not None:
                entry.response_body = entry.decrypted_response
                if 'decrypted' not in entry.tags:
                    entry.tags.append('decrypted')
        
        return entries
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/history")
async def clear_history(
    session_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Clear proxy history."""
    try:
        if session_id:
            await db.execute(
                """DELETE FROM proxy_history WHERE session_id = :session_id""",
                {"session_id": session_id}
            )
        else:
            await db.execute("""DELETE FROM proxy_history""")
        await db.commit()
        return {"message": "History cleared"}
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analysis Management
@router.get("/analysis", response_model=List[models.ProxyAnalysis])
async def get_analysis_results(
    session_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
) -> List[ProxyAnalysisResult]:
    """Get analysis results."""
    try:
        # Build query with proper type hints
        stmt = (
            select(ProxyAnalysisResult)
            .order_by(ProxyAnalysisResult.timestamp.desc())
        )
        if session_id:
            stmt = stmt.where(ProxyAnalysisResult.session_id == session_id)
        
        # Execute and handle results properly
        result = await db.execute(stmt)
        entries = list(result.scalars().all())
        return entries
    except Exception as e:
        logger.error(f"Failed to get analysis results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/analysis")
async def clear_analysis_results(
    session_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Clear analysis results."""
    try:
        if session_id:
            await db.execute(
                """DELETE FROM proxy_analysis_results WHERE session_id = :session_id""",
                {"session_id": session_id}
            )
        else:
            await db.execute("""DELETE FROM proxy_analysis_results""")
        await db.commit()
        return {"message": "Analysis results cleared"}
    except Exception as e:
        logger.error(f"Failed to clear analysis results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connections", response_model=List[models.ConnectionInfo])
async def get_connections(db: AsyncSession = Depends(get_db)) -> List[models.ConnectionInfo]:
    """Get all active proxy connections."""
    try:
        # Get connection info from TunnelProtocol class
        from proxy.server.custom_protocol import TunnelProtocol
        active_connections = []

        for conn_id, conn_info in TunnelProtocol._active_connections.items():
            connection = models.ConnectionInfo(
                id=conn_id,
                host=conn_info.get("host", "unknown"),
                port=conn_info.get("port", 0),
                start_time=conn_info.get("created_at", datetime.utcnow()).timestamp(),
                status="active" if not conn_info.get("end_time") else "closed",
                events=conn_info.get("events", []),
                bytes_received=conn_info.get("bytes_received", 0),
                bytes_sent=conn_info.get("bytes_sent", 0),
                requests_processed=conn_info.get("requests_processed", 0),
                error=conn_info.get("error")
            )
            if conn_info.get("end_time"):
                connection.end_time = conn_info["end_time"].timestamp()

            active_connections.append(connection)

        return active_connections
    except Exception as e:
        logger.error(f"Failed to get connections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Status endpoint
@router.get("/status")
async def get_proxy_status(db: AsyncSession = Depends(get_db)) -> dict:
    """Get current proxy status."""
    try:
        # Get active session
        result = await db.execute(
            select(ProxySession)
            .where(ProxySession.is_active == True)
            .order_by(ProxySession.start_time.desc())
            .limit(1)
        )
        active_session = result.scalar_one_or_none()

        # Get recent history entries
        history_result = await db.execute(
            select(ProxyHistoryEntry.id)
            .order_by(ProxyHistoryEntry.timestamp.desc())
            .limit(100)
        )
        recent_history_ids = [row[0] for row in history_result]

        return {
            "isRunning": active_session is not None,
            "settings": active_session.settings if active_session else {},
            "interceptRequests": active_session.settings.get("interceptRequests", True) if active_session else False,
            "interceptResponses": active_session.settings.get("interceptResponses", True) if active_session else False,
            "allowedHosts": active_session.settings.get("allowedHosts", []) if active_session else [],
            "excludedHosts": active_session.settings.get("excludedHosts", []) if active_session else [],
            "history": recent_history_ids
        }
    except Exception as e:
        logger.error(f"Failed to get proxy status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/rules", response_model=models.InterceptionRule)
async def create_rule(
    rule: models.InterceptionRuleCreate,
    session_id: int,
    db: AsyncSession = Depends(get_db)
) -> InterceptionRule:
    """Create a new interception rule."""
    # Verify session exists and is active
    result = await db.execute(
        select(ProxySession)
        .where(ProxySession.id == session_id)
        .where(ProxySession.is_active == True)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or not active")

    # Get highest priority and increment
    result = await db.execute(
        select(func.max(InterceptionRule.priority))
        .where(InterceptionRule.session_id == session_id)
    )
    max_priority = result.scalar() or 0
    
    # Create new rule
    db_rule = InterceptionRule(
        name=rule.name,
        enabled=rule.enabled,
        session_id=session_id,
        conditions=rule.conditions,
        action=rule.action,
        modification=rule.modification,
        priority=max_priority + 1
    )
    
    db.add(db_rule)
    await db.commit()
    await db.refresh(db_rule)
    return db_rule

@router.put("/rules/{rule_id}", response_model=models.InterceptionRule)
async def update_rule(
    rule_id: int,
    rule_update: models.InterceptionRuleUpdate,
    db: AsyncSession = Depends(get_db)
) -> InterceptionRule:
    """Update an existing interception rule."""
    result = await db.execute(
        select(InterceptionRule).where(InterceptionRule.id == rule_id)
    )
    db_rule = result.scalar_one_or_none()
    if not db_rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    if rule_update.name is not None:
        db_rule.name = rule_update.name
    if rule_update.enabled is not None:
        db_rule.enabled = rule_update.enabled
    if rule_update.conditions is not None:
        db_rule.conditions = rule_update.conditions
    if rule_update.action is not None:
        db_rule.action = rule_update.action
    if rule_update.modification is not None:
        db_rule.modification = rule_update.modification
    
    db_rule.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(db_rule)
    return db_rule

@router.delete("/rules/{rule_id}")
async def delete_rule(
    rule_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete an interception rule."""
    result = await db.execute(
        select(InterceptionRule).where(InterceptionRule.id == rule_id)
    )
    db_rule = result.scalar_one_or_none()
    if not db_rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    await db.delete(db_rule)
    await db.commit()
    return {"message": "Rule deleted"}

@router.get("/sessions/{session_id}/rules", response_model=List[models.InterceptionRule])
async def list_rules(
    session_id: int,
    enabled_only: bool = False,
    db: AsyncSession = Depends(get_db)
) -> List[InterceptionRule]:
    """List interception rules, optionally filtered by session."""
    try:
        # Build query with proper type hints
        stmt = select(InterceptionRule).order_by(InterceptionRule.priority)
        if session_id:
            stmt = stmt.where(InterceptionRule.session_id == session_id)
        if enabled_only:
            stmt = stmt.where(InterceptionRule.enabled == True)
        
        # Execute and handle results properly
        result = await db.execute(stmt)
        rules = list(result.scalars().all())
        return rules
    except Exception as e:
        logger.error(f"Failed to list rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/rules/reorder")
async def reorder_rules(
    rule_ids: List[int],
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Reorder rules by assigning new priorities."""
    # Verify all rules exist
    result = await db.execute(
        select(InterceptionRule).where(InterceptionRule.id.in_(rule_ids))
    )
    existing_rules = {rule.id: rule for rule in result.scalars().all()}
    if len(existing_rules) != len(rule_ids):
        raise HTTPException(status_code=400, detail="Some rules not found")

    # Update priorities
    for priority, rule_id in enumerate(rule_ids, start=1):
        existing_rules[rule_id].priority = priority
    
    await db.commit()
    return {"message": "Rules reordered successfully"}

@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
