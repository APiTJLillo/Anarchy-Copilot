"""WebSocket type definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any, Dict, List, Protocol, Optional, 
    Union, TYPE_CHECKING, TypeVar, runtime_checkable
)
from collections.abc import Iterator, Sized
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from .fuzzing import WSFuzzer

# Type variables and aliases
T = TypeVar('T')

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

# Type aliases
TestContainer = Optional[MessageContainer]

class MessageType(Enum):
    """WebSocket message types."""
    TEXT = auto()
    BINARY = auto()
    PING = auto()
    PONG = auto()
    CLOSE = auto()

# Legacy alias for backward compatibility 
WSMessageType = MessageType

class MessageDirection(Enum):
    """Message direction relative to proxy."""
    INCOMING = auto()
    OUTGOING = auto()

MessageID = Union[str, UUID]

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

class SecurityAnalyzer:
    """Security analysis for WebSocket messages."""
    
    @staticmethod
    def scan_message(message: WSMessage) -> Dict[str, Any]:
        """Scan a message for security issues.
        
        Args:
            message: Message to analyze
            
        Returns:
            Dict containing scan results
        """
        issues = []
        
        if message.type == MessageType.TEXT:
            content = str(message.data)
            
            # SQL Injection
            if any(x in content.upper() for x in ["SELECT", "INSERT", "UPDATE", "DELETE"]):
                issues.append("Potential SQL injection")
                
            # XSS
            if any(x in content.lower() for x in ["<script>", "javascript:", "onerror="]):
                issues.append("Potential XSS")
                
            # Path traversal
            if any(x in content for x in ["../", "..\\"]):
                issues.append("Potential path traversal")
                
            # Code injection
            if any(x in content for x in ["eval(", "setTimeout(", "setInterval("]):
                issues.append("Potential code injection")
                
        return {
            "message_id": str(message.id),
            "issues": issues,
            "severity": "high" if issues else "low",
            "timestamp": datetime.now().isoformat()
        }

def create_message(
    data: Any,
    type: MessageType,
    direction: MessageDirection,
    id: Optional[Union[str, UUID]] = None,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> WSMessage:
    """Create a new WebSocket message.
    
    Args:
        data: Message payload
        type: Message type
        direction: Message direction
        id: Optional message ID (default: new UUID)
        timestamp: Optional timestamp (default: now)
        metadata: Optional metadata dict

    Returns:
        New WSMessage instance
    """
    return WSMessage(
        id=id or uuid4(),
        type=type,
        data=data,
        direction=direction,
        timestamp=timestamp or datetime.now(),
        metadata=metadata or {}
    )

# Test helpers
def create_test_message(
    data: Any = "test",
    type: MessageType = MessageType.TEXT,
    direction: MessageDirection = MessageDirection.OUTGOING,
    id: Optional[MessageID] = None,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> WSMessage:
    """Create a message for testing with sensible defaults."""
    return create_message(data, type, direction, id, timestamp, metadata)

def create_text_message(
    data: str,
    direction: MessageDirection,
    id: Optional[MessageID] = None,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> WSMessage:
    """Create a text message with defaults."""
    return create_message(data, MessageType.TEXT, direction, id, timestamp, metadata)

def create_binary_message(
    data: bytes,
    direction: MessageDirection,
    id: Optional[MessageID] = None,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> WSMessage:
    """Create a binary message with defaults."""
    return create_message(data, MessageType.BINARY, direction, id, timestamp, metadata)

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
        """Create a text message for testing with defaults."""
        return create_text_message(data, direction, id, timestamp, metadata)

    @staticmethod
    def binary(
        data: bytes = b"test",
        direction: MessageDirection = MessageDirection.OUTGOING,
        id: Optional[MessageID] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WSMessage:
        """Create a binary message for testing with defaults."""
        return create_binary_message(data, direction, id, timestamp, metadata)

    @staticmethod
    def create(
        data: Any = "test",
        type: MessageType = MessageType.TEXT,
        direction: MessageDirection = MessageDirection.OUTGOING,
        id: Optional[MessageID] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WSMessage:
        """Create a test message with custom type and defaults."""
        return create_test_message(data, type, direction, id, timestamp, metadata)

    @staticmethod
    def now() -> datetime:
        """Get current timestamp for testing."""
        return datetime.now()

    @staticmethod
    def id() -> UUID:
        """Generate new UUID for testing."""
        return uuid4()

    # Factory methods for common test cases
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

    @classmethod
    def analyzer(cls) -> SecurityAnalyzer:
        """Get analyzer instance for testing."""
        return SecurityAnalyzer()

    @classmethod 
    def create_all(cls) -> List[WSMessage]:
        """Create one of each message type."""
        return [
            cls.text(),
            cls.binary(),
            cls.ping(),
            cls.pong(),
            cls.close(),
            cls.sql_injection(),
            cls.xss()
        ]

    # Message type factories
    @classmethod
    def ping(cls) -> WSMessage:
        """Create ping message."""
        return cls.create(b"", MessageType.PING)

    @classmethod
    def pong(cls) -> WSMessage:
        """Create pong message."""
        return cls.create(b"", MessageType.PONG)

    @classmethod
    def close(cls) -> WSMessage:
        """Create close message."""
        return cls.create(None, MessageType.CLOSE)

    @classmethod
    def sql_injection(cls) -> WSMessage:
        """Create message with SQL injection for testing."""
        return cls.text("SELECT * FROM users")

    @classmethod
    def xss(cls) -> WSMessage:
        """Create message with XSS for testing."""
        return cls.text("<script>alert(1)</script>")

# Re-export types
__all__ = [
    'MessageType',
    'MessageDirection', 
    'WSMessage',
    'MessageContainer',
    'WSMessageType',  # Legacy alias
    'MessageID',  # Type alias
    'create_message',  # Basic message creation
    'create_text_message',  # Text message helper
    'create_binary_message',  # Binary message helper 
    'create_test_message',  # Test message creation
    'SecurityAnalyzer',  # Security analysis
    'TestMessage'  # Test message factory
]
