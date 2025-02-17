"""
Base types for WebSocket message handling.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Union

class WSMessageType(Enum):
    """WebSocket message types."""
    TEXT = "text"
    BINARY = "binary"
    PING = "ping"
    PONG = "pong"
    CLOSE = "close"

@dataclass
class WSMessage:
    """Represents a WebSocket message."""
    type: WSMessageType
    data: Union[str, bytes]
    timestamp: datetime = field(default_factory=datetime.now)
    is_fuzzing: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert message to a dictionary format.
        
        Returns:
            Dictionary representation of the message
        """
        return {
            'type': self.type.value,
            'data': self.data.decode('utf-8', errors='ignore') if isinstance(self.data, bytes) else self.data,
            'timestamp': self.timestamp.isoformat(),
            'is_fuzzing': self.is_fuzzing,
            'metadata': self.metadata
        }
    
    def is_text(self) -> bool:
        """Check if this is a text message.
        
        Returns:
            True if this is a text message, False otherwise
        """
        return self.type == WSMessageType.TEXT
    
    def is_binary(self) -> bool:
        """Check if this is a binary message.
        
        Returns:
            True if this is a binary message, False otherwise
        """
        return self.type == WSMessageType.BINARY
    
    def is_control(self) -> bool:
        """Check if this is a control message (PING/PONG/CLOSE).
        
        Returns:
            True if this is a control message, False otherwise
        """
        return self.type in [WSMessageType.PING, WSMessageType.PONG, WSMessageType.CLOSE]

    def get_size(self) -> int:
        """Get the size of the message data in bytes.
        
        Returns:
            Size of the message data in bytes
        """
        if isinstance(self.data, str):
            return len(self.data.encode('utf-8'))
        return len(self.data)
