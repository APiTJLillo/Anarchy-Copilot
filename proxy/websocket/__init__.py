"""
WebSocket interception and analysis module for Anarchy Copilot.

This package provides comprehensive WebSocket support including:
- Message interception and modification
- Conversation tracking and analysis
- Security testing and fuzzing
- Pattern matching and validation
"""

from .types import MessageType, MessageDirection, WSMessage
from .conversation import WSConversation
from .interceptor import WebSocketInterceptor, DebugInterceptor, SecurityInterceptor, RateLimitInterceptor
from .manager import WebSocketManager
from .fuzzer import WSFuzzer
from .routes import create_router

# Create global WebSocket manager instance
ws_manager = WebSocketManager()

__all__ = [
    'MessageType',
    'MessageDirection',
    'WSMessage',
    'WSConversation',
    'WebSocketInterceptor',
    'DebugInterceptor',
    'SecurityInterceptor',
    'RateLimitInterceptor',
    'WebSocketManager',
    'WSFuzzer',
    'create_router',
    'ws_manager'
]
