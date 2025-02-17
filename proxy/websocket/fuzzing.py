"""
WebSocket fuzzing functionality.
"""
import json
import random
from typing import Any, Dict, List, Union

from .types import WSMessage, WSMessageType
from .conversation import WSConversation

class WSFuzzer:
    """Fuzzes WebSocket messages for security testing."""

    # Common SQL injection payloads
    SQL_PAYLOADS = [
        "' OR '1'='1",
        "1; DROP TABLE users --",
        "' UNION SELECT * FROM users --",
        "admin'--",
        '" OR ""="',
        "1' ORDER BY 1--+",
        "1' AND 1=CONVERT(int,@@version)--",
    ]

    # Common XSS payloads
    XSS_PAYLOADS = [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        "javascript:alert(1)",
        '<svg onload=alert(1)>',
        '"><script>alert(1)</script>',
        '"onmouseover="alert(1)',
    ]

    # Protocol-level modifications
    FRAME_FLAGS = {
        'fin': [True, False],
        'rsv1': [True, False],
        'rsv2': [True, False],
        'rsv3': [True, False],
        'masked': [True, False],
    }

    async def fuzz_conversation(self, conversation: WSConversation) -> List[WSMessage]:
        """Fuzz all messages in a conversation.
        
        Args:
            conversation: The WebSocket conversation to fuzz
            
        Returns:
            List of fuzzed messages
        """
        fuzzed_messages: List[WSMessage] = []
        
        for msg in conversation.messages:
            if msg.type == WSMessageType.TEXT:
                # Apply different fuzzing techniques
                fuzzed_messages.extend(self.fuzz_sql_injection(msg))
                fuzzed_messages.extend(self.fuzz_xss(msg))
                fuzzed_messages.extend(self.fuzz_json_structure(msg))
            fuzzed_messages.extend(self.fuzz_protocol(msg))
            
        return fuzzed_messages

    def fuzz_sql_injection(self, message: WSMessage) -> List[WSMessage]:
        """Generate SQL injection fuzzed variants of a message.
        
        Args:
            message: Original message to fuzz
            
        Returns:
            List of fuzzed messages
        """
        fuzzed: List[WSMessage] = []
        
        if message.type != WSMessageType.TEXT:
            return fuzzed

        try:
            # Try to parse as JSON to fuzz structured data
            data = json.loads(message.data)
            for key in data:
                for payload in self.SQL_PAYLOADS:
                    fuzzed_data = data.copy()
                    fuzzed_data[key] = payload
                    fuzzed.append(WSMessage(
                        type=WSMessageType.TEXT,
                        data=json.dumps(fuzzed_data),
                        metadata={
                            'fuzzed': True,
                            'fuzzing_type': 'sql_injection',
                            'original_data': message.data
                        }
                    ))
        except json.JSONDecodeError:
            # For non-JSON text, inject payloads directly
            for payload in self.SQL_PAYLOADS:
                fuzzed.append(WSMessage(
                    type=WSMessageType.TEXT,
                    data=payload,
                    metadata={
                        'fuzzed': True,
                        'fuzzing_type': 'sql_injection',
                        'original_data': message.data
                    }
                ))
                
        return fuzzed

    def fuzz_xss(self, message: WSMessage) -> List[WSMessage]:
        """Generate XSS fuzzed variants of a message.
        
        Args:
            message: Original message to fuzz
            
        Returns:
            List of fuzzed messages
        """
        fuzzed: List[WSMessage] = []
        
        if message.type != WSMessageType.TEXT:
            return fuzzed

        try:
            # Try to parse as JSON to fuzz structured data
            data = json.loads(message.data)
            for key in data:
                for payload in self.XSS_PAYLOADS:
                    fuzzed_data = data.copy()
                    fuzzed_data[key] = payload
                    fuzzed.append(WSMessage(
                        type=WSMessageType.TEXT,
                        data=json.dumps(fuzzed_data),
                        metadata={
                            'fuzzed': True,
                            'fuzzing_type': 'xss',
                            'original_data': message.data
                        }
                    ))
        except json.JSONDecodeError:
            # For non-JSON text, inject payloads directly
            for payload in self.XSS_PAYLOADS:
                fuzzed.append(WSMessage(
                    type=WSMessageType.TEXT,
                    data=payload,
                    metadata={
                        'fuzzed': True,
                        'fuzzing_type': 'xss',
                        'original_data': message.data
                    }
                ))
                
        return fuzzed

    def fuzz_json_structure(self, message: WSMessage) -> List[WSMessage]:
        """Generate structure-based fuzzed variants of JSON messages.
        
        Args:
            message: Original message to fuzz
            
        Returns:
            List of fuzzed messages
        """
        fuzzed: List[WSMessage] = []
        
        if message.type != WSMessageType.TEXT:
            return fuzzed

        try:
            data = json.loads(message.data)
            
            # Recursive function to mutate JSON structure
            def mutate_structure(obj: Union[Dict, List]) -> Any:
                if isinstance(obj, dict):
                    result = obj.copy()
                    # Random mutations
                    if random.random() < 0.3:  # 30% chance to modify
                        op = random.choice(['add', 'remove', 'modify'])
                        if op == 'add':
                            result['fuzzed_field'] = 'fuzzed_value'
                        elif op == 'remove' and result:
                            del result[random.choice(list(result.keys()))]
                        elif op == 'modify' and result:
                            key = random.choice(list(result.keys()))
                            result[key] = mutate_structure(result[key])
                    return result
                elif isinstance(obj, list):
                    result = obj.copy()
                    if random.random() < 0.3:
                        if result:
                            # Either duplicate an element or remove one
                            if random.random() < 0.5 and result:
                                result.append(random.choice(result))
                            else:
                                result.pop(random.randrange(len(result)))
                    return result
                else:
                    # For primitive values, occasionally modify them
                    if random.random() < 0.2:
                        if isinstance(obj, str):
                            return obj + "_fuzzed"
                        elif isinstance(obj, (int, float)):
                            return obj * 2
                    return obj

            # Generate multiple fuzzed variants
            for _ in range(3):
                fuzzed_data = mutate_structure(data)
                fuzzed.append(WSMessage(
                    type=WSMessageType.TEXT,
                    data=json.dumps(fuzzed_data),
                    metadata={
                        'fuzzed': True,
                        'fuzzing_type': 'json_structure',
                        'original_data': message.data
                    }
                ))
        except json.JSONDecodeError:
            pass
            
        return fuzzed

    def fuzz_protocol(self, message: WSMessage) -> List[WSMessage]:
        """Generate protocol-level fuzzed variants of a message.
        
        Args:
            message: Original message to fuzz
            
        Returns:
            List of fuzzed messages
        """
        fuzzed: List[WSMessage] = []
        
        # Try different message types
        msg_types = [t for t in WSMessageType if t != message.type]
        for msg_type in msg_types:
            fuzzed.append(WSMessage(
                type=msg_type,
                data=message.data,
                metadata={
                    'fuzzed': True,
                    'fuzzing_type': 'protocol',
                    'original_data': message.data,
                }
            ))

        # Try different frame flag combinations
        for _ in range(3):
            flags = {
                flag: random.choice(values)
                for flag, values in self.FRAME_FLAGS.items()
            }
            fuzzed.append(WSMessage(
                type=message.type,
                data=message.data,
                metadata={
                    'fuzzed': True,
                    'fuzzing_type': 'protocol',
                    'original_data': message.data,
                    'frame_flags': flags
                }
            ))

        return fuzzed
