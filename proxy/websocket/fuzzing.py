"""WebSocket fuzzing implementation."""
from dataclasses import dataclass, field
from typing import (
    Dict, Any, List, Optional, Awaitable,
    cast, Union, Protocol, runtime_checkable,
    TypeVar, Type, get_args, Iterator
)
from enum import Enum, auto
import asyncio
import json
import logging
import random
import uuid
from datetime import datetime

from .types import WSMessage, MessageType, MessageDirection
from .conversation import WSConversation, ConversationState

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='Fuzzable')

@runtime_checkable
class FuzzableMessage(Protocol):
    """Protocol for messages that can be fuzzed."""
    id: uuid.UUID
    type: MessageType 
    data: Union[str, bytes]
    direction: MessageDirection
    timestamp: datetime
    metadata: Dict[str, Any]

@runtime_checkable
class Fuzzable(Protocol):
    """Protocol for objects that can be fuzzed."""
    messages: List[FuzzableMessage]
    id: Union[str, uuid.UUID]
    type: Union[str, MessageType]

    @classmethod
    def from_conversation(cls: Type[T], conv: Optional['WSConversation']) -> T:
        """Create a fuzzable from a conversation."""
        if not conv:
            raise ValueError("Cannot create Fuzzable from None")
        if not isinstance(conv.type, (str, MessageType)):
            raise TypeError(f"Invalid conversation type: {type(conv.type)}")
        return cast(T, conv)

    def validate(self) -> bool:
        """Validate that the fuzzable object is valid."""
        if not self.messages:
            return False
        return all(isinstance(msg, FuzzableMessage) for msg in self.messages)

class FuzzingType(Enum):
    """Types of fuzzing operations."""
    SQL_INJECTION = auto()
    XSS = auto()
    PROTOCOL = auto()
    JSON_STRUCTURE = auto()
    
    def __str__(self) -> str:
        return self.name.lower()

@dataclass
class WSFuzzer:
    """WebSocket fuzzer for testing message variations."""
    
    is_enabled: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    _sql_patterns: List[str] = field(default_factory=lambda: [
        "' OR '1'='1",
        "; DROP TABLE",
        "UNION SELECT",
        "--",
        "/**/",
    ])
    _xss_patterns: List[str] = field(default_factory=lambda: [
        "<script>alert(1)</script>",
        "javascript:alert(1)",
        "<img src=x onerror=alert(1)>",
        "'\"><script>alert(1)</script>",
    ])

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure fuzzer settings."""
        try:
            self.config.update(config or {})
            self.is_enabled = config.get('enabled', False)
            logger.info(f"Fuzzer configured - enabled: {self.is_enabled}")
        except Exception as e:
            logger.error(f"Error configuring fuzzer: {e}")
            self.is_enabled = False

    def validate_message(self, msg: FuzzableMessage) -> bool:
        """Validate a message for fuzzing."""
        return all([
            msg.id is not None,
            msg.direction is not None,
            msg.timestamp is not None,
            msg.type is not None,
            msg.data is not None
        ])

    async def fuzz_conversation(self, conversation: Union[Fuzzable, WSConversation]) -> List[WSMessage]:
        """Fuzz a conversation's messages.
        
        Args:
            conversation: WebSocket conversation to fuzz
            
        Returns:
            List of fuzzed messages
        """
        if not self.is_enabled:
            return []

        try:
            conv = conversation if isinstance(conversation, Fuzzable) else Fuzzable.from_conversation(conversation)
            if not conv.validate():
                logger.error("Failed to validate conversation for fuzzing")
                return []

            fuzzed_messages: List[WSMessage] = []
            for msg in conv.messages:
                try:
                    fuzzed_msg = WSMessage(
                        id=uuid.uuid4(),
                        type=msg.type,
                        data=self._fuzz_data(msg.data),
                        direction=msg.direction,
                        timestamp=datetime.now(),
                        metadata={"fuzzed": True, **msg.metadata}
                    )
                    fuzzed_messages.append(fuzzed_msg)
                except Exception as e:
                    logger.error(f"Error fuzzing message {msg.id}: {e}")
                    continue
                    
            logger.info(f"Generated {len(fuzzed_messages)} fuzzed messages")
            return fuzzed_messages
            
        except Exception as e:
            logger.error(f"Error in fuzz_conversation: {e}")
            return []

    async def fuzz_sql_injection(self, conversation: Union[Fuzzable, WSConversation]) -> List[WSMessage]:
        """Perform SQL injection fuzzing."""
        if not self.is_enabled:
            return []
            
        conv = conversation if isinstance(conversation, Fuzzable) else Fuzzable.from_conversation(conversation)
        if not conv.validate():
            logger.error("Failed to validate conversation for SQL injection fuzzing")
            return []

        fuzzed_messages: List[WSMessage] = []
        for msg in conv.messages:
            if msg.type != MessageType.TEXT:
                continue
                
            for pattern in self._sql_patterns:
                fuzzed_msg = WSMessage(
                    id=uuid.uuid4(),
                    type=msg.type,
                    data=self._inject_pattern(str(msg.data), pattern),
                    direction=msg.direction,
                    timestamp=datetime.now(),
                    metadata={"fuzzed": True, "fuzz_type": "sql_injection", **msg.metadata}
                )
                fuzzed_messages.append(fuzzed_msg)
        return fuzzed_messages

    async def fuzz_xss(self, conversation: Union[Fuzzable, WSConversation]) -> List[WSMessage]:
        """Perform XSS fuzzing."""
        if not self.is_enabled:
            return []
            
        conv = conversation if isinstance(conversation, Fuzzable) else Fuzzable.from_conversation(conversation)
        if not conv.validate():
            logger.error("Failed to validate conversation for XSS fuzzing")
            return []
            
        fuzzed_messages: List[WSMessage] = []
        for msg in conv.messages:
            if msg.type != MessageType.TEXT:
                continue
                
            for pattern in self._xss_patterns:
                fuzzed_msg = WSMessage(
                    id=uuid.uuid4(),
                    type=msg.type,
                    data=self._inject_pattern(str(msg.data), pattern),
                    direction=msg.direction,
                    timestamp=datetime.now(),
                    metadata={"fuzzed": True, "fuzz_type": "xss", **msg.metadata}
                )
                fuzzed_messages.append(fuzzed_msg)
        return fuzzed_messages

    async def fuzz_protocol(self, conversation: Union[Fuzzable, WSConversation]) -> List[WSMessage]:
        """Perform WebSocket protocol fuzzing."""
        if not self.is_enabled:
            return []
            
        conv = conversation if isinstance(conversation, Fuzzable) else Fuzzable.from_conversation(conversation)
        if not conv.validate():
            logger.error("Failed to validate conversation for protocol fuzzing")
            return []
            
        fuzzed_messages: List[WSMessage] = []
        mutations = [
            (MessageType.TEXT, "Invalid UTF-8: "),
            (MessageType.BINARY, b"" * 1000),
            (MessageType.PING, b"large ping" * 1000),
            (MessageType.CLOSE, b"invalid close reason")
        ]
        
        for msg in conv.messages:
            for mut_type, mut_data in mutations:
                fuzzed_msg = WSMessage(
                    id=uuid.uuid4(),
                    type=mut_type,
                    data=mut_data,
                    direction=msg.direction,
                    timestamp=datetime.now(),
                    metadata={"fuzzed": True, "fuzz_type": "protocol", **msg.metadata}
                )
                fuzzed_messages.append(fuzzed_msg)
        return fuzzed_messages

    async def fuzz_json_structure(self, conversation: Union[Fuzzable, WSConversation]) -> List[WSMessage]:
        """Perform JSON structure fuzzing."""
        if not self.is_enabled:
            return []
            
        conv = conversation if isinstance(conversation, Fuzzable) else Fuzzable.from_conversation(conversation)
        if not conv.validate():
            logger.error("Failed to validate conversation for JSON structure fuzzing")
            return []
            
        fuzzed_messages: List[WSMessage] = []
        for msg in conv.messages:
            if msg.type != MessageType.TEXT:
                continue
                
            try:
                if not self.validate_message(msg):
                    logger.warning(f"Invalid message format: {msg}")
                    continue
                    
                data = json.loads(str(msg.data))
                fuzzed_data = self._fuzz_json(data)
                fuzzed_msg = WSMessage(
                    id=uuid.uuid4(),
                    type=msg.type,
                    data=json.dumps(fuzzed_data),
                    direction=msg.direction,
                    timestamp=datetime.now(),
                    metadata={"fuzzed": True, "fuzz_type": "json_structure", **msg.metadata}
                )
                fuzzed_messages.append(fuzzed_msg)
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse JSON from message {msg.id}")
                continue
                
        return fuzzed_messages

    def _fuzz_data(self, data: Union[str, bytes]) -> Union[str, bytes]:
        """Apply fuzzing mutations to message data.
        
        Args:
            data: String or bytes data to fuzz
            
        Returns:
            Fuzzed data of the same type as input
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        try:
            if isinstance(data, str):
                try:
                    # Try to parse as JSON
                    json_data = json.loads(data)
                    fuzzed_json = self._fuzz_json(json_data)
                    return json.dumps(fuzzed_json)
                except json.JSONDecodeError:
                    # Not JSON, fuzz as string
                    return self._fuzz_string(data)
            elif isinstance(data, bytes):
                return self._fuzz_bytes(data)
            return data
        except Exception as e:
            logger.error(f"Error fuzzing data: {e}")
            return data

    def _inject_pattern(self, data: str, pattern: str) -> str:
        """Inject a pattern into string data."""
        # Try to inject at different positions
        positions = [
            lambda s, p: p + s,  # Prefix
            lambda s, p: s + p,  # Suffix
            lambda s, p: s.replace("=", f"={p}"),  # After equals
            lambda s, p: s.replace(" ", f" {p} "),  # Between words
        ]
        inject = random.choice(positions)
        return inject(data, pattern)

    def _fuzz_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fuzz JSON data by modifying values and structure."""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self._fuzz_string(value)
            elif isinstance(value, (int, float)):
                result[key] = value + random.randint(-10, 10)
            elif isinstance(value, bool):
                result[key] = not value
            elif isinstance(value, dict):
                result[key] = self._fuzz_json(value)
            else:
                result[key] = value
        return result

    def _fuzz_string(self, data: str) -> str:
        """Apply string fuzzing mutations."""
        mutations = [
            lambda s: s.upper(),
            lambda s: s.lower(),
            lambda s: s + "'",
            lambda s: s + "\\",
            lambda s: s + "",  # Null byte
            lambda s: s + "",  # Zero-width space
            lambda s: s[::-1],
            lambda s: s.replace(" ", "\t"),
            lambda s: "".join(c * 2 for c in s),  # Double characters
            lambda s: s + random.choice(self._xss_patterns),
        ]
        mutation = random.choice(mutations)
        return mutation(data)

    def _fuzz_bytes(self, data: bytes) -> bytes:
        """Apply byte fuzzing mutations."""
        mutations = [
            lambda b: b + b"",
            lambda b: b + b"",
            lambda b: b[::-1],
            lambda b: bytes([x ^ 0xff for x in b])
        ]
        mutation = random.choice(mutations)
        return mutation(data)
