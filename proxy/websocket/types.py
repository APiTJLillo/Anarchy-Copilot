"""WebSocket type definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any, Dict, List, Protocol, Optional,
    Union, TYPE_CHECKING, TypeVar, runtime_checkable,
    Callable, Awaitable, Literal, Type, Sequence,
    Generic, ClassVar, TypedDict
)
from collections.abc import Iterator, Sized
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from .fuzzing import WSFuzzer

# Type variables and aliases
T = TypeVar('T')

class ConnectionState(Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected" 
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"

class MessageType(Enum):
    """WebSocket message types."""
    TEXT = "text"
    BINARY = "binary" 
    PING = "ping"
    PONG = "pong"
    CLOSE = "close"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_RESPONSE = "heartbeat_response"
    CONNECTION_STATUS = "connection_status"
    STATE_CHANGE = "state_change"

ConversationType = Union[str, MessageType]

# Legacy type alias 
WSMessageType = MessageType

# Type aliases
MessageID = Union[str, UUID]

class MessageDirection(Enum):
    """Message direction."""
    INCOMING = "incoming"
    OUTGOING = "outgoing"
    INTERNAL = "internal"  # For internal control messages

class ConnectionType(Enum):
    """WebSocket connection types."""
    UI = "ui"
    INTERNAL = "internal"
    PROXY = "proxy"

@runtime_checkable
class MessageBase(Protocol):
    """Base protocol for message objects."""
    
    def __len__(self) -> int:
        """Get count of items."""
        ...

    def __iter__(self) -> Iterator[T]:
        """Iterate over items."""
        ...

    def __bool__(self) -> bool:
        """Check if container has items."""
        ...

class ConnectionContext(TypedDict):
    """Connection context for status updates."""
    id: str
    state: ConnectionState
    type: ConnectionType
    created_at: datetime
    last_activity: datetime
    healthy: bool
    error_count: int
    last_error: Optional[str]
    metadata: Dict[str, Any]

@dataclass 
class WSMessage:
    """WebSocket message."""
    id: MessageID
    type: MessageType
    data: Any
    direction: MessageDirection
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Convert string ID to UUID if needed."""
        if isinstance(self.id, str):
            try:
                self.id = UUID(self.id)
            except ValueError:
                pass  # Leave as string if not valid UUID

@runtime_checkable
class MessageContainer(Protocol):
    """Protocol for WebSocket message containers."""
    
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
        """Get message statistics."""
        ...

class ConversationContext(TypedDict):
    """Context for WebSocket conversations."""
    type: ConversationType
    messages: Sequence[WSMessage]
    state: ConnectionState

@runtime_checkable
class WSConversation(Protocol):
    """Protocol for WebSocket conversations."""
    
    conversation_type: ClassVar[ConversationType]
    _type: ConversationType
    _messages: Sequence[WSMessage]
    _state: ConnectionState

    @property
    def type(self) -> ConversationType:
        """Get conversation type."""
        return self._type

    @type.setter 
    def type(self, value: ConversationType) -> None:
        """Set conversation type."""
        self._type = value

    @property
    def state(self) -> ConnectionState:
        """Get conversation state."""
        return self._state

    @state.setter
    def state(self, value: ConnectionState) -> None:
        """Set conversation state."""
        self._state = value

    @property
    def messages(self) -> Sequence[WSMessage]:
        """Get conversation messages."""
        return self._messages

    def from_conversation(self, other: WSConversation) -> None:
        """Initialize from another conversation."""
        ...

    def validate(self) -> bool:
        """Validate conversation."""
        ...

    def get_context(self) -> ConversationContext:
        """Get conversation context."""
        return {
            "type": self.type,
            "messages": self.messages,
            "state": self.state
        }

@runtime_checkable
class Fuzzable(Protocol):
    """Protocol for fuzzable items."""

    conversation_type: ClassVar[ConversationType]
    _type: ConversationType
    _messages: Sequence[WSMessage]
    _state: ConnectionState

    @property
    def type(self) -> ConversationType:
        """Get conversation type."""
        return self._type

    @type.setter
    def type(self, value: ConversationType) -> None:
        """Set conversation type."""
        self._type = value

    @property
    def state(self) -> ConnectionState:
        """Get conversation state."""
        return self._state

    @state.setter
    def state(self, value: ConnectionState) -> None:
        """Set conversation state."""
        self._state = value

    @property
    def messages(self) -> Sequence[WSMessage]:
        """Get messages."""
        return self._messages

    def from_conversation(self, conv: WSConversation) -> None:
        """Initialize from conversation."""
        ...

    def validate(self) -> bool:
        """Validate item."""
        ...

    def get_context(self) -> ConversationContext:
        """Get conversation context."""
        return {
            "type": self.type,
            "messages": self.messages,
            "state": self.state
        }

TestContainer = Optional[MessageContainer]

# Event handler types  
EventHandler = Callable[..., Union[None, Awaitable[None]]]
ConnectionEventType = Literal["connect", "disconnect", "error", "state_change"]

@dataclass
class ConnectionStatus:
    """WebSocket connection status."""
    connected: bool
    connected_at: datetime
    last_activity: datetime
    type: ConnectionType
    state: ConnectionState
    duration: float
    message_count: int
    healthy: bool
    last_error: Optional[str]
    reconnect_count: int
    last_reconnect: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        connection_type: Union[str, ConnectionType],
        connected: bool = True,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConnectionStatus:
        """Create new connection status."""
        now = timestamp or datetime.utcnow()
        conn_type = ConnectionType(connection_type) if isinstance(connection_type, str) else connection_type
        return cls(
            connected=connected,
            connected_at=now,
            last_activity=now,
            type=conn_type,
            state=ConnectionState.CONNECTED if connected else ConnectionState.CLOSED,
            duration=0.0,
            message_count=0,
            healthy=True,
            last_error=None,
            reconnect_count=0,
            last_reconnect=None,
            metadata=metadata or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "connected": self.connected,
            "connected_at": self.connected_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "type": self.type.value,
            "state": self.state.value,
            "duration": self.duration,
            "message_count": self.message_count,
            "healthy": self.healthy,
            "last_error": self.last_error,
            "reconnect_count": self.reconnect_count,
            "last_reconnect": self.last_reconnect.isoformat() if self.last_reconnect else None,
            "metadata": self.metadata
        }

    def update_state(self, state: ConnectionState, error: Optional[str] = None) -> None:
        """Update connection state."""
        self.state = state
        if state == ConnectionState.ERROR and error:
            self.healthy = False
            self.last_error = error
        elif state == ConnectionState.CONNECTED:
            self.connected = True
            self.healthy = True
            self.last_error = None
        elif state == ConnectionState.CLOSED:
            self.connected = False

# Message creation functions
def create_message(
    data: Any,
    type: MessageType,
    direction: MessageDirection,
    id: Optional[Union[str, UUID]] = None,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> WSMessage:
    """Create a new WebSocket message."""
    return WSMessage(
        id=id or uuid4(),
        type=type,
        data=data,
        direction=direction,
        timestamp=timestamp or datetime.now(),
        metadata=metadata or {}
    )

def create_heartbeat_message(
    data: Optional[Dict[str, Any]] = None,
    direction: MessageDirection = MessageDirection.OUTGOING,
    response: bool = False
) -> WSMessage:
    """Create a heartbeat message."""
    msg_type = MessageType.HEARTBEAT_RESPONSE if response else MessageType.HEARTBEAT
    msg_data = {
        "timestamp": datetime.utcnow().isoformat(),
        **(data or {})
    }
    return create_message(msg_data, msg_type, direction)

def create_text_message(
    data: str,
    direction: MessageDirection,
    id: Optional[MessageID] = None,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> WSMessage:
    """Create a text message."""
    return create_message(data, MessageType.TEXT, direction, id, timestamp, metadata)

def create_binary_message(
    data: bytes,
    direction: MessageDirection,
    id: Optional[MessageID] = None,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> WSMessage:
    """Create a binary message."""
    return create_message(data, MessageType.BINARY, direction, id, timestamp, metadata)

def create_status_message(
    connection_id: str,
    status: ConnectionStatus,
    event_type: ConnectionEventType
) -> WSMessage:
    """Create a connection status message."""
    data = {
        "connection_id": connection_id,
        "status": status.to_dict(),
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat()
    }
    return create_message(
        data,
        MessageType.CONNECTION_STATUS,
        MessageDirection.INTERNAL
    )

def create_state_message(
    connection_id: str,
    old_state: ConnectionState,
    new_state: ConnectionState,
    metadata: Optional[Dict[str, Any]] = None
) -> WSMessage:
    """Create a state change message."""
    data = {
        "connection_id": connection_id,
        "old_state": old_state.value,
        "new_state": new_state.value,
        "timestamp": datetime.utcnow().isoformat(),
        **(metadata or {})
    }
    return create_message(
        data,
        MessageType.STATE_CHANGE,
        MessageDirection.INTERNAL
    )

class TestMessage:
    """Helper class for creating test messages."""
    
    @staticmethod
    def text(
        data: str = "test",
        direction: MessageDirection = MessageDirection.OUTGOING,
        id: Optional[MessageID] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WSMessage:
        """Create a text message for testing."""
        return create_message(data, MessageType.TEXT, direction, id, timestamp, metadata)

    @staticmethod
    def binary(
        data: bytes = b"test",
        direction: MessageDirection = MessageDirection.OUTGOING,
        id: Optional[MessageID] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WSMessage:
        """Create a binary message for testing."""
        return create_message(data, MessageType.BINARY, direction, id, timestamp, metadata)

    @staticmethod
    def heartbeat(
        response: bool = False,
        direction: MessageDirection = MessageDirection.OUTGOING
    ) -> WSMessage:
        """Create a heartbeat message for testing."""
        return create_heartbeat_message(direction=direction, response=response)

    @staticmethod
    def ping() -> WSMessage:
        """Create ping message for testing."""
        return create_message(b"", MessageType.PING, MessageDirection.OUTGOING)

    @staticmethod
    def pong() -> WSMessage:
        """Create pong message for testing."""
        return create_message(b"", MessageType.PONG, MessageDirection.OUTGOING)

    @staticmethod
    def close() -> WSMessage:
        """Create close message for testing."""
        return create_message(None, MessageType.CLOSE, MessageDirection.OUTGOING)

    @staticmethod
    def connection_status(
        connection_id: str = "test",
        status: Optional[ConnectionStatus] = None,
        event_type: ConnectionEventType = "connect"
    ) -> WSMessage:
        """Create a connection status message for testing."""
        if status is None:
            status = ConnectionStatus.create("ui")
        return create_status_message(connection_id, status, event_type)

    @staticmethod
    def state_change(
        connection_id: str = "test",
        old_state: ConnectionState = ConnectionState.CONNECTING,
        new_state: ConnectionState = ConnectionState.CONNECTED
    ) -> WSMessage:
        """Create a state change message for testing."""
        return create_state_message(connection_id, old_state, new_state)

    @staticmethod
    def sql_injection() -> WSMessage:
        """Create message with SQL injection for testing."""
        return create_text_message("SELECT * FROM users", MessageDirection.OUTGOING)

    @staticmethod
    def xss() -> WSMessage:
        """Create message with XSS for testing."""
        return create_text_message("<script>alert(1)</script>", MessageDirection.OUTGOING)

    @classmethod
    def create_all(cls) -> List[WSMessage]:
        """Create one of each message type."""
        return [
            cls.text(),
            cls.binary(),
            cls.heartbeat(),
            cls.ping(),
            cls.pong(),
            cls.close(),
            cls.sql_injection(),
            cls.xss(),
            cls.heartbeat(response=True),
            cls.connection_status(),
            cls.state_change()
        ]

    @staticmethod
    def now() -> datetime:
        """Get current timestamp for testing."""
        return datetime.now()

    @staticmethod
    def id() -> UUID:
        """Generate new UUID for testing."""
        return uuid4()

    @classmethod
    def null_container(cls) -> Optional[MessageContainer]:
        """Return None for testing null checks."""
        return None

    @classmethod
    def empty_container(cls) -> MessageContainer:
        """Return empty container for testing."""
        class EmptyContainer:
            def __len__(self) -> int:
                return 0

            def __iter__(self) -> Iterator[WSMessage]:
                return iter([])

            def __bool__(self) -> bool:
                return False

            def get_message_stats(self) -> Dict[str, int]:
                return {"total": 0, "text": 0, "binary": 0}
        return EmptyContainer()

    @classmethod
    def fuzzer(cls) -> "WSFuzzer":
        """Get fuzzer instance for testing."""
        from .fuzzing import WSFuzzer
        return WSFuzzer()

# Re-export types
__all__ = [
    'ConnectionContext',
    'ConnectionEventType',
    'ConnectionState',
    'ConnectionStatus',
    'ConnectionType',
    'ConversationContext',
    'ConversationType',
    'EventHandler',
    'Fuzzable',
    'MessageBase',
    'MessageContainer',
    'MessageDirection',
    'MessageID',
    'MessageType',
    'TestContainer',
    'TestMessage',
    'WSConversation',
    'WSMessage',
    'WSMessageType',  # Legacy alias re-export
    'create_binary_message',
    'create_heartbeat_message',
    'create_message',
    'create_status_message',
    'create_state_message',
    'create_text_message'
]
