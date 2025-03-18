"""WebSocket conversation model."""
from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any, Dict, List, Optional, Protocol,
    TYPE_CHECKING, Union, runtime_checkable
)
import uuid

from .types import (
    WSMessage, MessageType, MessageDirection,
    MessageContainer as BaseMessageContainer
)

# Re-export these types for convenience
__all__ = ['WSConversation', 'MessageID', 'MessagePattern']

if TYPE_CHECKING:
    from aiohttp import ClientWebSocketResponse
    from aiohttp.web import WebSocketResponse

# Type aliases for clarity
MessageID = Union[str, uuid.UUID]
MessagePattern = Union[str, List[str]]

@runtime_checkable
class MessageContainer(Protocol):
    """Protocol for objects containing messages."""
    
    def __len__(self) -> int:
        """Get number of messages."""
        ...
    
    def __iter__(self) -> Iterator[WSMessage]:
        """Iterate over messages."""
        ...

    def __bool__(self) -> bool:
        """Check if container has messages."""
        ...

    def get_message_stats(self) -> Dict[str, int]:
        """Get statistics about messages."""
        ...

@dataclass
class WSConversation(MessageContainer):
    """Represents a WebSocket conversation session."""
    
    id: str
    url: str
    created_at: datetime = field(default_factory=datetime.now)
    messages: List[WSMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    closed_at: Optional[datetime] = None

    def __len__(self) -> int:
        """Get number of messages in conversation."""
        return len(self.messages)
    
    def __iter__(self) -> 'Iterator[WSMessage]':
        """Iterate over messages in conversation."""
        return iter(self.messages)
    
    def __bool__(self) -> bool:
        """Check if conversation has any messages."""
        return bool(self.messages)

    def add_message(self, message: WSMessage) -> None:
        """Add a message to the conversation history."""
        self.messages.append(message)
        
    def message_count(self) -> int:
        """Get the number of messages in the conversation."""
        return len(self)

    def get_message_stats(self) -> Dict[str, int]:
        """Get statistics about messages in the conversation."""
        return {
            "total": len(self.messages),
            "outgoing": sum(1 for m in self.messages if m.direction == MessageDirection.OUTGOING),
            "incoming": sum(1 for m in self.messages if m.direction == MessageDirection.INCOMING),
            "text": sum(1 for m in self.messages if m.type == MessageType.TEXT),
            "binary": sum(1 for m in self.messages if m.type == MessageType.BINARY)
        }

    def find_patterns(self, patterns: MessagePattern) -> List[WSMessage]:
        """Find messages matching specific patterns.
        
        Args:
            patterns: Pattern or list of patterns to search for
            
        Returns:
            List of messages matching any of the patterns
        """
        # Convert single pattern to list
        if isinstance(patterns, str):
            patterns = [patterns]
            
        matching_messages = []
        for msg in self.messages:
            if msg.type == MessageType.TEXT:
                data = str(msg.data)
                if any(pattern in data for pattern in patterns):
                    matching_messages.append(msg)
        return matching_messages

    def find_message_by_id(self, message_id: MessageID) -> Optional[WSMessage]:
        """Find a message by its ID.
        
        Args:
            message_id: ID of message to find
            
        Returns:
            Matching message or None if not found
        """
        message_id = str(message_id)
        for msg in self.messages:
            if str(msg.id) == message_id:
                return msg
        return None

    def get_security_analysis(self) -> Dict[str, Any]:
        """Get security analysis of the conversation.
        
        Returns:
            Dictionary containing security analysis results
        """
        suspicious_patterns = [
            "SELECT", "INSERT", "UPDATE", "DELETE",  # SQL
            "<script>", "javascript:",  # XSS
            "../", "..\\", "etc/passwd",  # Path traversal
            "eval(", "setTimeout(", "setInterval("  # Code injection
        ]

        return {
            "suspicious_messages": len(self.find_patterns(suspicious_patterns)),
            "binary_messages": sum(1 for m in self.messages if m.type == MessageType.BINARY),
            "message_sizes": [len(str(m.data)) for m in self.messages if m.type == MessageType.TEXT],
            "metadata": self.metadata
        }

    async def replay_messages(
        self, 
        client_ws: 'WebSocketResponse',
        server_ws: 'ClientWebSocketResponse'
    ) -> None:
        """Replay conversation messages in order.
        
        Args:
            client_ws: Client WebSocket connection
            server_ws: Server WebSocket connection
        """
        for msg in self.messages:
            target = server_ws if msg.direction == MessageDirection.OUTGOING else client_ws
            if msg.type == MessageType.TEXT:
                await target.send_str(msg.data)
            elif msg.type == MessageType.BINARY:
                await target.send_bytes(msg.data)
            elif msg.type == MessageType.PING:
                await target.ping(msg.data)
            elif msg.type == MessageType.PONG:
                await target.pong(msg.data)
            elif msg.type == MessageType.CLOSE:
                await target.close()
