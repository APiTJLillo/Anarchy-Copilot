"""WebSocket fuzzing implementation with list loading capabilities."""
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
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from .types import WSMessage, MessageType, MessageDirection
from .conversation import WSConversation, ConversationState
from .parameter_detector import ParameterDetector, DetectedParameter

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
    CUSTOM = auto()
    AUTO_DETECT = auto()  # New fuzzing type for auto-detected parameters
    
    def __str__(self) -> str:
        return self.name.lower()

@dataclass
class FuzzingList:
    """Represents a list of fuzzing payloads."""
    id: str
    name: str
    description: str
    category: str
    payloads: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_file(cls, file_path: str, name: Optional[str] = None, category: Optional[str] = None) -> 'FuzzingList':
        """Create a fuzzing list from a file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            payloads = [line.strip() for line in f if line.strip()]
            
        return cls(
            id=str(uuid.uuid4()),
            name=name or path.stem,
            description=f"Imported from {path.name}",
            category=category or "imported",
            payloads=payloads
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "payload_count": len(self.payloads),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class FuzzingListManager:
    """Manages fuzzing lists storage and retrieval."""
    
    def __init__(self, db_path: str = "fuzzing_lists.db"):
        """Initialize the fuzzing list manager."""
        self.db_path = db_path
        self._initialize_db()
        
    def _initialize_db(self) -> None:
        """Initialize the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fuzzing_lists (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            category TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fuzzing_payloads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            list_id TEXT,
            payload TEXT,
            FOREIGN KEY (list_id) REFERENCES fuzzing_lists (id) ON DELETE CASCADE
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def save_list(self, fuzzing_list: FuzzingList) -> bool:
        """Save a fuzzing list to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update the list's updated_at timestamp
            fuzzing_list.updated_at = datetime.now()
            
            # Check if list exists
            cursor.execute("SELECT id FROM fuzzing_lists WHERE id = ?", (fuzzing_list.id,))
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update existing list
                cursor.execute('''
                UPDATE fuzzing_lists 
                SET name = ?, description = ?, category = ?, updated_at = ?
                WHERE id = ?
                ''', (
                    fuzzing_list.name,
                    fuzzing_list.description,
                    fuzzing_list.category,
                    fuzzing_list.updated_at.isoformat(),
                    fuzzing_list.id
                ))
                
                # Delete existing payloads
                cursor.execute("DELETE FROM fuzzing_payloads WHERE list_id = ?", (fuzzing_list.id,))
            else:
                # Insert new list
                cursor.execute('''
                INSERT INTO fuzzing_lists (id, name, description, category, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    fuzzing_list.id,
                    fuzzing_list.name,
                    fuzzing_list.description,
                    fuzzing_list.category,
                    fuzzing_list.created_at.isoformat(),
                    fuzzing_list.updated_at.isoformat()
                ))
            
            # Insert payloads
            for payload in fuzzing_list.payloads:
                cursor.execute('''
                INSERT INTO fuzzing_payloads (list_id, payload)
                VALUES (?, ?)
                ''', (fuzzing_list.id, payload))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving fuzzing list: {e}")
            return False
    
    def get_list(self, list_id: str) -> Optional[FuzzingList]:
        """Get a fuzzing list by ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get list metadata
            cursor.execute('''
            SELECT id, name, description, category, created_at, updated_at
            FROM fuzzing_lists
            WHERE id = ?
            ''', (list_id,))
            
            row = cursor.fetchone()
            if not row:
                conn.close()
                return None
                
            list_id, name, description, category, created_at, updated_at = row
            
            # Get payloads
            cursor.execute('''
            SELECT payload FROM fuzzing_payloads
            WHERE list_id = ?
            ''', (list_id,))
            
            payloads = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            return FuzzingList(
                id=list_id,
                name=name,
                description=description,
                category=category,
                payloads=payloads,
                created_at=datetime.fromisoformat(created_at),
                updated_at=datetime.fromisoformat(updated_at)
            )
        except Exception as e:
            logger.error(f"Error getting fuzzing list: {e}")
            return None
    
    def get_all_lists(self) -> List[Dict[str, Any]]:
        """Get all fuzzing lists (metadata only)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT l.id, l.name, l.description, l.category, l.created_at, l.updated_at, COUNT(p.id) as payload_count
            FROM fuzzing_lists l
            LEFT JOIN fuzzing_payloads p ON l.id = p.list_id
            GROUP BY l.id
            ''')
            
            lists = []
            for row in cursor.fetchall():
                list_id, name, description, category, created_at, updated_at, payload_count = row
                lists.append({
                    "id": list_id,
                    "name": name,
                    "description": description,
                    "category": category,
                    "payload_count": payload_count,
                    "created_at": created_at,
                    "updated_at": updated_at
                })
            
            conn.close()
            return lists
        except Exception as e:
            logger.error(f"Error getting all fuzzing lists: {e}")
            return []
    
    def delete_list(self, list_id: str) -> bool:
        """Delete a fuzzing list."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM fuzzing_lists WHERE id = ?", (list_id,))
            # Payloads will be deleted automatically due to ON DELETE CASCADE
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error deleting fuzzing list: {e}")
            return False
    
    def import_from_file(self, file_path: str, name: Optional[str] = None, category: Optional[str] = None) -> Optional[FuzzingList]:
        """Import a fuzzing list from a file."""
        try:
            fuzzing_list = FuzzingList.from_file(file_path, name, category)
            if self.save_list(fuzzing_list):
                return fuzzing_list
            return None
        except Exception as e:
            logger.error(f"Error importing fuzzing list from file: {e}")
            return None

@dataclass
class WSFuzzer:
    """WebSocket fuzzer for testing message variations."""
    
    is_enabled: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    list_manager: FuzzingListManager = field(default_factory=lambda: FuzzingListManager())
    parameter_detector: ParameterDetector = field(default_factory=lambda: ParameterDetector())
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
            
            # Configure parameter detector if settings provided
            if 'parameter_detector' in config:
                detector_config = config['parameter_detector']
                if 'confidence_threshold' in detector_config:
                    self.parameter_detector = ParameterDetector(
                        confidence_threshold=float(detector_config['confidence_threshold'])
                    )
                    
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

    async def fuzz_with_list(self, conversation: Union[Fuzzable, WSConversation], list_id: str) -> List[WSMessage]:
        """Fuzz a conversation using a specific fuzzing list.
        
        Args:
            conversation: WebSocket conversation to fuzz
            list_id: ID of the fuzzing list to use
            
        Returns:
            List of fuzzed messages
        """
        if not self.is_enabled:
            return []
            
        fuzzing_list = self.list_manager.get_list(list_id)
        if not fuzzing_list:
            logger.error(f"Fuzzing list not found: {list_id}")
            return []
            
        try:
            conv = conversation if isinstance(conversation, Fuzzable) else Fuzzable.from_conversation(conversation)
            if not conv.validate():
                logger.error("Failed to validate conversation for custom list fuzzing")
                return []

            fuzzed_messages: List[WSMessage] = []
            for msg in conv.messages:
                if msg.type != MessageType.TEXT:
                    continue
                    
                for payload in fuzzing_list.payloads:
                    fuzzed_msg = WSMessage(
                        id=uuid.uuid4(),
                        type=msg.type,
                        data=self._inject_pattern(str(msg.data), payload),
                        direction=msg.direction,
                        timestamp=datetime.now(),
                        metadata={
                            "fuzzed": True, 
                            "fuzz_type": "custom", 
                            "list_id": list_id,
                            "list_name": fuzzing_list.name,
                            **msg.metadata
                        }
                    )
                    fuzzed_messages.append(fuzzed_msg)
            
            logger.info(f"Generated {len(fuzzed_messages)} fuzzed messages using list '{fuzzing_list.name}'")
            return fuzzed_messages
            
        except Exception as e:
            logger.error(f"Error in fuzz_with_list: {e}")
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

    async def detect_and_fuzz_parameters(self, conversation: Union[Fuzzable, WSConversation], 
                                        list_id: Optional[str] = None) -> List[WSMessage]:
        """Auto-detect parameters and fuzz them.
        
        Args:
            conversation: WebSocket conversation to analyze and fuzz
            list_id: Optional ID of fuzzing list to use for payloads
            
        Returns:
            List of fuzzed messages targeting detected parameters
        """
        if not self.is_enabled:
            return []
            
        conv = conversation if isinstance(conversation, Fuzzable) else Fuzzable.from_conversation(conversation)
        if not conv.validate():
            logger.error("Failed to validate conversation for auto-detect parameter fuzzing")
            return []
            
        fuzzed_messages: List[WSMessage] = []
        
        # Get payloads to use
        payloads = []
        if list_id:
            fuzzing_list = self.list_manager.get_list(list_id)
            if fuzzing_list:
                payloads = fuzzing_list.payloads
            else:
                logger.warning(f"Fuzzing list not found: {list_id}, using default payloads")
                payloads = self._get_default_payloads()
        else:
            payloads = self._get_default_payloads()
            
        # Process each message
        for msg in conv.messages:
            if msg.type != MessageType.TEXT:
                continue
                
            # Detect parameters
            detected_params = self.parameter_detector.detect_parameters(msg)
            if not detected_params:
                logger.info(f"No parameters detected in message {msg.id}")
                continue
                
            logger.info(f"Detected {len(detected_params)} parameters in message {msg.id}")
            
            # Fuzz each parameter with each payload
            for param in detected_params:
                for payload in payloads:
                    try:
                        # Create fuzzed message with the parameter replaced
                        fuzzed_data = self._replace_parameter_with_payload(
                            str(msg.data), param, payload
                        )
                        
                        if fuzzed_data:
                            fuzzed_msg = WSMessage(
                                id=uuid.uuid4(),
                                type=msg.type,
                                data=fuzzed_data,
                                direction=msg.direction,
                                timestamp=datetime.now(),
                                metadata={
                                    "fuzzed": True,
                                    "fuzz_type": "auto_detect",
                                    "param_name": param.name,
                                    "param_type": str(param.param_type),
                                    "param_path": param.path,
                                    "original_value": param.value,
                                    "payload": payload,
                                    **msg.metadata
                                }
                            )
                            fuzzed_messages.append(fuzzed_msg)
                    except Exception as e:
                        logger.error(f"Error fuzzing parameter {param.name}: {e}")
                        continue
        
        logger.info(f"Generated {len(fuzzed_messages)} fuzzed messages using auto-detected parameters")
        return fuzzed_messages
    
    def get_detected_parameters(self, conversation: Union[Fuzzable, WSConversation]) -> List[Dict[str, Any]]:
        """Get all detected parameters from a conversation.
        
        Args:
            conversation: WebSocket conversation to analyze
            
        Returns:
            List of detected parameters as dictionaries
        """
        conv = conversation if isinstance(conversation, Fuzzable) else Fuzzable.from_conversation(conversation)
        if not conv.validate():
            logger.error("Failed to validate conversation for parameter detection")
            return []
            
        all_params = []
        for msg in conv.messages:
            if msg.type != MessageType.TEXT:
                continue
                
            # Detect parameters
            detected_params = self.parameter_detector.detect_parameters(msg)
            all_params.extend([p.to_dict() for p in detected_params])
            
        return all_params

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
        
    def _get_default_payloads(self) -> List[str]:
        """Get default payloads for auto-detected parameters."""
        return [
            # SQL Injection
            "' OR '1'='1",
            "1; DROP TABLE users",
            "1 UNION SELECT username, password FROM users",
            
            # XSS
            "<script>alert(1)</script>",
            "<img src=x onerror=alert(1)>",
            
            # Command Injection
            "$(cat /etc/passwd)",
            "; ls -la",
            "| cat /etc/shadow",
            
            # Path Traversal
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            
            # Special Characters
            "!@#$%^&*()_+{}[]|\\:;\"'<>,.?/",
            
            # Numeric
            "0",
            "-1",
            "99999999",
            "3.14159",
            
            # Boolean
            "true",
            "false",
            
            # Empty/Null
            "",
            "null",
            
            # Long strings
            "A" * 1000,
            
            # Format strings
            "%s%s%s%s%s",
            "%x%x%x%x",
            "%n%n%n%n",
            
            # Unicode
            "你好",
            "Привет",
            "😀😈👾"
        ]
        
    def _replace_parameter_with_payload(self, data: str, param: DetectedParameter, payload: str) -> Optional[str]:
        """Replace a parameter with a payload in the message data.
        
        Args:
            data: Original message data
            param: Parameter to replace
            payload: Payload to inject
            
        Returns:
            Modified data with parameter replaced by payload
        """
        try:
            # Handle different parameter types
            if param.param_type.name == "JSON_KEY" or param.param_type.name == "JSON_VALUE":
                return self._replace_in_json(data, param, payload)
            elif param.param_type.name == "URL_QUERY":
                return self._replace_in_url(data, param, payload)
            elif param.param_type.name == "FORM_DATA":
                return self._replace_in_form(data, param, payload)
            else:
                # For custom parameters, try simple string replacement
                original_value = str(param.value)
                if original_value in data:
                    return data.replace(original_value, payload)
                return None
        except Exception as e:
            logger.error(f"Error replacing parameter {param.name} with payload: {e}")
            return None
            
    def _replace_in_json(self, data: str, param: DetectedParameter, payload: str) -> Optional[str]:
        """Replace a parameter in JSON data.
        
        Args:
            data: JSON string
            param: Parameter to replace
            payload: Payload to inject
            
        Returns:
            Modified JSON string
        """
        try:
            json_data = json.loads(data)
            
            # Parse the JSON path
            path_parts = param.path.split('.')
            if path_parts[0] == '$':
                path_parts = path_parts[1:]
                
            # Navigate to the target location
            current = json_data
            parent = None
            last_key = None
            
            for i, part in enumerate(path_parts):
                # Handle array indices
                if '[' in part and ']' in part:
                    key, idx_str = part.split('[', 1)
                    idx = int(idx_str.rstrip(']'))
                    
                    if key:
                        if i == len(path_parts) - 1:
                            parent = current
                            last_key = key
                        current = current[key]
                        current[idx] = payload if i == len(path_parts) - 1 else current[idx]
                    else:
                        if i == len(path_parts) - 1:
                            parent = current
                            last_key = idx
                        current[idx] = payload if i == len(path_parts) - 1 else current[idx]
                else:
                    if i == len(path_parts) - 1:
                        parent = current
                        last_key = part
                    else:
                        current = current[part]
            
            # Replace the value
            if parent is not None and last_key is not None:
                if isinstance(last_key, int):
                    parent[last_key] = payload
                else:
                    parent[last_key] = payload
                    
            return json.dumps(json_data)
        except Exception as e:
            logger.error(f"Error replacing in JSON: {e}")
            return None
            
    def _replace_in_url(self, data: str, param: DetectedParameter, payload: str) -> str:
        """Replace a parameter in URL query string.
        
        Args:
            data: URL or query string
            param: Parameter to replace
            payload: Payload to inject
            
        Returns:
            Modified URL or query string
        """
        param_name = param.name
        
        # Handle URL query parameters
        if '?' in data:
            base_url, query = data.split('?', 1)
            params = urllib.parse.parse_qs(query)
            
            if param_name in params:
                params[param_name] = [payload]
                new_query = urllib.parse.urlencode(params, doseq=True)
                return f"{base_url}?{new_query}"
        
        # Simple replacement for other cases
        pattern = f"{param_name}={param.value}"
        replacement = f"{param_name}={payload}"
        return data.replace(pattern, replacement)
            
    def _replace_in_form(self, data: str, param: DetectedParameter, payload: str) -> str:
        """Replace a parameter in form data.
        
        Args:
            data: Form encoded data
            param: Parameter to replace
            payload: Payload to inject
            
        Returns:
            Modified form data
        """
        param_name = param.name
        
        # Parse form data
        params = urllib.parse.parse_qs(data)
        
        if param_name in params:
            params[param_name] = [payload]
            return urllib.parse.urlencode(params, doseq=True)
        
        # Simple replacement for other cases
        pattern = f"{param_name}={param.value}"
        replacement = f"{param_name}={payload}"
        return data.replace(pattern, replacement)
