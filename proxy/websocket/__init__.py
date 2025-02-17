"""
WebSocket interception and analysis module for Anarchy Copilot.

This package provides comprehensive WebSocket support including:
- Message interception and modification
- Conversation tracking and analysis
- Security testing and fuzzing
- Pattern matching and validation
"""

from .types import WSMessageType, WSMessage
from .conversation import WSConversation
from .interceptor import WebSocketInterceptor, DebugInterceptor, SecurityInterceptor, RateLimitInterceptor
from .manager import WebSocketManager
from .fuzzing import WSFuzzer

__all__ = [
    'WSMessageType',
    'WSMessage',
    'WSConversation',
    'WebSocketInterceptor',
    'DebugInterceptor',
    'SecurityInterceptor',
    'RateLimitInterceptor',
    'WebSocketManager',
    'WSFuzzer'
]
