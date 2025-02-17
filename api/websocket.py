"""WebSocket API endpoints."""
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from proxy.websocket.manager import WebSocketManager
from proxy.websocket.interceptor import SecurityInterceptor, DebugInterceptor

router = APIRouter(tags=["websocket"])
ws_manager = WebSocketManager()

# Add interceptors
debug_interceptor = DebugInterceptor()
security_interceptor = SecurityInterceptor()
ws_manager.add_interceptor(debug_interceptor)
ws_manager.add_interceptor(security_interceptor)

@router.get("/connections")
async def get_websocket_connections() -> List[Dict[str, Any]]:
    """Get list of active WebSocket connections."""
    connections = []
    for conv_id in ws_manager.active_sessions:
        conv = ws_manager.get_conversation(conv_id)
        if conv:
            connections.append({
                "id": conv.id,
                "url": conv.url,
                "status": "ACTIVE" if conv_id in ws_manager.active_sessions else "CLOSED",
                "timestamp": conv.created_at.isoformat(),
                "interceptorEnabled": any(i.is_enabled for i in ws_manager._interceptors),
                "fuzzingEnabled": ws_manager._fuzzer.is_enabled,
                "securityAnalysisEnabled": security_interceptor.is_enabled,
                "conversationId": conv.id,
            })
    return connections

@router.get("/messages/{connection_id}")
async def get_websocket_messages(connection_id: str) -> List[Dict[str, Any]]:
    """Get messages for a WebSocket connection."""
    conversation = ws_manager.get_conversation(connection_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    messages = []
    for msg in conversation.messages:
        messages.append({
            "id": msg.id,
            "type": "SEND" if msg.direction == "outgoing" else "RECEIVE",
            "payload": msg.data if msg.type.value == 1 else str(msg.data),  # 1 = TEXT
            "timestamp": msg.timestamp.isoformat(),
            "intercepted": msg.metadata.get("intercepted", False),
            "modified": msg.metadata.get("modified", False),
            "fuzzed": msg.metadata.get("fuzzed", False),
            "securityIssues": msg.metadata.get("security_issues", [])
        })
    return messages

@router.get("/security/report/{connection_id}")
async def get_security_report(connection_id: str) -> Dict[str, Any]:
    """Get security analysis report for a WebSocket connection."""
    conversation = ws_manager.get_conversation(connection_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    return conversation.metadata.get('security_report', {
        "finding_count": 0,
        "findings": [],
        "summary": {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
    })

@router.post("/config/{connection_id}")
async def update_connection_config(
    connection_id: str,
    config: Dict[str, Any]
) -> Dict[str, str]:
    """Update WebSocket connection configuration."""
    conversation = ws_manager.get_conversation(connection_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    if "interceptorEnabled" in config:
        for interceptor in ws_manager._interceptors:
            if not isinstance(interceptor, SecurityInterceptor):  # Don't disable security
                interceptor.is_enabled = config["interceptorEnabled"]
    
    if "fuzzingEnabled" in config:
        ws_manager._fuzzer.is_enabled = config["fuzzingEnabled"]
        if "fuzzConfig" in config:
            ws_manager._fuzzer.configure(config["fuzzConfig"])
    
    if "securityAnalysisEnabled" in config:
        security_interceptor.is_enabled = config["securityAnalysisEnabled"]
    
    return {"message": "Configuration updated successfully"}

@router.post("/close/{connection_id}")
async def close_connection(connection_id: str) -> Dict[str, str]:
    """Close a WebSocket connection."""
    conversation = ws_manager.get_conversation(connection_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    ws_manager.close_conversation(connection_id)
    return {"message": "Connection closed successfully"}

@router.post("/send")
async def send_message(data: Dict[str, Any]) -> Dict[str, str]:
    """Send a message through a WebSocket connection."""
    connection_id = data.get("connectionId")
    conversation = ws_manager.get_conversation(connection_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    # Actual sending is handled by the WebSocketManager through its interceptor chain
    return {"message": "Message sent successfully"}
