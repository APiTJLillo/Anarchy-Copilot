"""API endpoints for WebSocket parameter detection and fuzzing."""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
import logging
import asyncio
import json

from ..websocket.manager import WebSocketManager
from ..websocket.parameter_detector import ParameterDetector
from ..websocket.fuzzing import WSFuzzer

router = APIRouter(prefix="/api/websocket/parameters", tags=["websocket-parameters"])
logger = logging.getLogger(__name__)

@router.post("/detect")
async def detect_parameters(
    connection_id: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Detect parameters in WebSocket messages.
    
    Args:
        connection_id: WebSocket connection ID
        config: Optional configuration for parameter detection
        
    Returns:
        Dictionary with detected parameters
    """
    try:
        # Get WebSocket manager instance
        ws_manager = WebSocketManager.get_instance()
        
        # Get conversation for the connection
        conversation = ws_manager.get_conversation(connection_id)
        if not conversation:
            raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")
            
        # Create parameter detector with config
        confidence_threshold = 0.7
        if config and "confidence_threshold" in config:
            confidence_threshold = float(config["confidence_threshold"])
            
        detector = ParameterDetector(confidence_threshold=confidence_threshold)
        
        # Create fuzzer instance
        fuzzer = WSFuzzer(is_enabled=True, parameter_detector=detector)
        
        # Get detected parameters
        parameters = fuzzer.get_detected_parameters(conversation)
        
        return {
            "connection_id": connection_id,
            "parameters": parameters,
            "count": len(parameters)
        }
    except Exception as e:
        logger.error(f"Error detecting parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fuzz")
async def fuzz_parameters(
    connection_id: str,
    parameters: List[str],
    list_id: Optional[str] = None,
    auto_detect: bool = True
) -> Dict[str, Any]:
    """Fuzz detected parameters.
    
    Args:
        connection_id: WebSocket connection ID
        parameters: List of parameter names to fuzz
        list_id: Optional fuzzing list ID to use
        auto_detect: Whether to auto-detect best injection points
        
    Returns:
        Dictionary with fuzzed messages
    """
    try:
        # Get WebSocket manager instance
        ws_manager = WebSocketManager.get_instance()
        
        # Get conversation for the connection
        conversation = ws_manager.get_conversation(connection_id)
        if not conversation:
            raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")
            
        # Create fuzzer instance
        fuzzer = WSFuzzer(is_enabled=True)
        
        # Fuzz parameters
        fuzzed_messages = await fuzzer.detect_and_fuzz_parameters(
            conversation=conversation,
            list_id=list_id
        )
        
        # Filter by selected parameters if specified
        if parameters:
            fuzzed_messages = [
                msg for msg in fuzzed_messages
                if msg.metadata.get("param_name") in parameters
            ]
        
        # Convert to serializable format
        serialized_messages = []
        for msg in fuzzed_messages:
            serialized_messages.append({
                "id": str(msg.id),
                "type": msg.type.value if hasattr(msg.type, "value") else str(msg.type),
                "data": msg.data,
                "direction": msg.direction.value if hasattr(msg.direction, "value") else str(msg.direction),
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            })
        
        return {
            "connection_id": connection_id,
            "fuzzed_messages": serialized_messages,
            "count": len(serialized_messages)
        }
    except Exception as e:
        logger.error(f"Error fuzzing parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))
