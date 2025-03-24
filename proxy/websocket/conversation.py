"""WebSocket conversation model."""
from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Protocol,
    TYPE_CHECKING, Union, runtime_checkable,
    TypedDict
)
import uuid
import logging

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

logger = logging.getLogger(__name__)

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

    def get_message_stats(self) -> 'ConversationStats':
        """Get statistics about messages."""
        ...

class ConversationState(Enum):
    """WebSocket conversation states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"

# Base required stats
class ConversationStatsRequired(TypedDict):
    """Required conversation statistics."""
    total_messages: int
    text_messages: int
    binary_messages: int
    state: str
    errors: int

# Optional stats
class ConversationStats(ConversationStatsRequired, total=False):
    """Complete conversation statistics."""
    outgoing_messages: int
    incoming_messages: int
    duration: Optional[float]
    ping_messages: int
    pong_messages: int
    last_activity: Optional[datetime]

@dataclass
class WSConversation(MessageContainer):
    """Represents a WebSocket conversation session."""
    
    id: str
    url: str
    created_at: datetime = field(default_factory=datetime.now)
    messages: List[WSMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    closed_at: Optional[datetime] = None
    state: ConversationState = field(default=ConversationState.CONNECTING)
    error_count: int = field(default=0)
    last_error: Optional[str] = field(default=None)

    def set_state(self, state: ConversationState, error: Optional[str] = None) -> None:
        """Update conversation state."""
        self.state = state
        if state == ConversationState.ERROR and error:
            self.error_count += 1
            self.last_error = error
            logger.error(f"Conversation {self.id} error: {error}")

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

    def get_message_stats(self) -> ConversationStats:
        """Get statistics about messages in the conversation."""
        stats: ConversationStats = {
            "total_messages": len(self.messages),
            "outgoing_messages": sum(1 for m in self.messages if m.direction == MessageDirection.OUTGOING),
            "incoming_messages": sum(1 for m in self.messages if m.direction == MessageDirection.INCOMING),
            "text_messages": sum(1 for m in self.messages if m.type == MessageType.TEXT),
            "binary_messages": sum(1 for m in self.messages if m.type == MessageType.BINARY),
            "state": self.state.value,
            "duration": self.duration,
            "errors": self.error_count,
            "ping_messages": sum(1 for m in self.messages if m.type == MessageType.PING),
            "pong_messages": sum(1 for m in self.messages if m.type == MessageType.PONG)
        }

        # Add last activity if we have messages
        if self.messages:
            stats["last_activity"] = self.messages[-1].timestamp

        return stats

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
                self.set_state(ConversationState.CLOSING)
                try:
                    await target.close()
                    self.set_state(ConversationState.CLOSED)
                except Exception as e:
                    self.set_state(ConversationState.ERROR, str(e))
                break

    @property
    def duration(self) -> Optional[float]:
        """Get conversation duration in seconds."""
        if self.closed_at:
            return (self.closed_at - self.created_at).total_seconds()
        if self.state in (ConversationState.CONNECTED, ConversationState.CLOSING):
            return (datetime.now() - self.created_at).total_seconds()
        return None

    @property
    def is_active(self) -> bool:
        """Check if conversation is active."""
        return self.state in (ConversationState.CONNECTING, ConversationState.CONNECTED)

    @property
    def had_errors(self) -> bool:
        """Check if conversation had any errors."""
        return self.error_count > 0

    def __str__(self) -> str:
        """String representation of conversation."""
        return (f"WSConversation(id={self.id}, url={self.url}, "
                f"state={self.state.value}, messages={len(self)}, "
                f"errors={self.error_count})")
