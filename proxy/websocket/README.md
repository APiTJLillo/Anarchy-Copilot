# WebSocket Module

This module provides WebSocket support for the Anarchy Copilot proxy, including message interception, fuzzing, and traffic analysis.

## Components

### WSMessage and WSMessageType
Basic types representing WebSocket messages and their types (TEXT, BINARY, etc).

### WSConversation
Tracks and manages WebSocket message history for a connection, enabling:
- Message recording and playback
- Security analysis
- Pattern matching

### WSFuzzer
Implements WebSocket fuzzing capabilities:
- Protocol-level fuzzing (frame flags, message types)
- Content-based fuzzing (SQL injection, XSS)
- JSON structure mutations

### SecurityAnalyzer
Analyzes WebSocket traffic for security issues:
- Sensitive data detection (JWT tokens, API keys)
- Attack pattern detection
- Message validation

### WebSocket Interceptors
Pipeline for intercepting and modifying WebSocket traffic:
- DebugInterceptor: Logs connection details and messages
- SecurityInterceptor: Validates handshakes and checks messages
- RateLimitInterceptor: Implements rate limiting

### WebSocketManager
Central component that coordinates WebSocket handling:
- Connection management
- Message routing
- Interceptor pipeline execution
- Fuzzing integration

## Usage Example

```python
from proxy.websocket import (
    WebSocketManager, 
    DebugInterceptor,
    SecurityInterceptor
)

# Create manager instance
ws_manager = WebSocketManager()

# Add interceptors
ws_manager.add_interceptor(DebugInterceptor())
ws_manager.add_interceptor(SecurityInterceptor())

# Handle WebSocket upgrade
async def handle_upgrade(request):
    target_url = "ws://target-server/ws"
    return await ws_manager.handle_websocket((request, target_url, {}))
```

## Message Flow

1. Client initiates WebSocket upgrade
2. Manager creates new conversation
3. Interceptors validate handshake
4. Bidirectional communication starts:
   - Messages pass through interceptors
   - Security analysis is performed
   - Messages are recorded in conversation
   - Optional fuzzing can be applied
5. Connection closes, conversation is archived

## Security Features

- Handshake validation
- Message rate limiting
- Security pattern detection
- Fuzzing capabilities
- Traffic analysis
- Connection monitoring

## Adding Custom Interceptors

Create a new interceptor by inheriting from WebSocketInterceptor:

```python
from proxy.websocket import WebSocketInterceptor, WSMessage, WSConversation

class CustomInterceptor(WebSocketInterceptor):
    async def on_connect(self, request) -> bool:
        # Validate connection
        return True
        
    async def on_message(self, message: WSMessage, conversation: WSConversation) -> WSMessage:
        # Process message
        return message
        
    async def on_close(self, conversation: WSConversation) -> None:
        # Clean up
        pass
