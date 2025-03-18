"""WebSocket fuzzing implementation."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import random
import json
import uuid
from datetime import datetime

from .types import WSMessage, MessageType, MessageDirection
from .conversation import WSConversation

@dataclass
class WSFuzzer:
    """WebSocket fuzzer for testing message variations."""
    
    is_enabled: bool = False
    config: Dict[str, Any] = field(default_factory=dict)

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure fuzzer settings."""
        self.config.update(config or {})
        self.is_enabled = config.get('enabled', False)

    async def fuzz_conversation(self, conversation: WSConversation) -> List[WSMessage]:
        """Fuzz a conversation's messages.
        
        Args:
            conversation: WebSocket conversation to fuzz
            
        Returns:
            List of fuzzed messages
        """
        if not self.is_enabled:
            return []

        fuzzed_messages: List[WSMessage] = []
        for msg in conversation.messages:
            fuzzed_msg = WSMessage(
                id=uuid.uuid4(),
                type=msg.type,
                data=self._fuzz_data(msg.data),
                direction=msg.direction,
                timestamp=datetime.now(),
                metadata={"fuzzed": True, **msg.metadata}
            )
            fuzzed_messages.append(fuzzed_msg)
        return fuzzed_messages

    def _fuzz_data(self, data: Any) -> Any:
        """Apply fuzzing mutations to message data."""
        if isinstance(data, str):
            try:
                # Try to parse as JSON
                json_data = json.loads(data)
                fuzzed_json = self._fuzz_json(json_data)
                return json.dumps(fuzzed_json)
            except:
                # Not JSON, fuzz as string
                return self._fuzz_string(data)
        elif isinstance(data, bytes):
            return self._fuzz_bytes(data)
        return data

    def _fuzz_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fuzz JSON data by modifying values."""
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
            lambda s: s + "\u0000",
            lambda s: s[::-1]
        ]
        mutation = random.choice(mutations)
        return mutation(data)

    def _fuzz_bytes(self, data: bytes) -> bytes:
        """Apply byte fuzzing mutations."""
        mutations = [
            lambda b: b + b"\x00",
            lambda b: b + b"\xff",
            lambda b: b[::-1],
            lambda b: bytes([x ^ 0xff for x in b])
        ]
        mutation = random.choice(mutations)
        return mutation(data)
