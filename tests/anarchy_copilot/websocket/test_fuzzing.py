"""
Tests for WebSocket fuzzing functionality.
"""
import json
import pytest
from typing import List

from proxy.websocket.types import WSMessage, WSMessageType
from proxy.websocket.fuzzing import WSFuzzer
from proxy.websocket.conversation import WSConversation

@pytest.fixture
def ws_fuzzer() -> WSFuzzer:
    """Fixture for WebSocket fuzzer."""
    return WSFuzzer()

async def test_basic_fuzzing(ws_fuzzer: WSFuzzer, ws_conversation: WSConversation, sample_ws_messages: List[WSMessage]):
    """Test basic fuzzing of WebSocket messages."""
    # Add original messages to conversation
    for msg in sample_ws_messages:
        if msg.type == WSMessageType.TEXT:
            ws_conversation.add_message(msg)
    
    # Generate fuzzed variations
    fuzzed_messages = await ws_fuzzer.fuzz_conversation(ws_conversation)
    
    assert len(fuzzed_messages) > 0
    for msg in fuzzed_messages:
        assert msg.metadata.get('fuzzed') is True
        assert msg.type == WSMessageType.TEXT
        assert msg.data != msg.metadata.get('original_data')

def test_sql_injection_fuzzing(ws_fuzzer: WSFuzzer):
    """Test SQL injection fuzzing patterns."""
    original_msg = WSMessage(
        type=WSMessageType.TEXT,
        data='{"user_id": 123}',
        metadata={}
    )
    
    # Generate SQL injection fuzzed messages
    fuzzed = ws_fuzzer.fuzz_sql_injection(original_msg)
    
    assert len(fuzzed) > 0
    for msg in fuzzed:
        assert msg.metadata.get('fuzzing_type') == 'sql_injection'
        data = msg.data if isinstance(msg.data, str) else msg.data.decode('utf-8')
        assert any(pattern in data for pattern in [
            "' OR '1'='1",
            " UNION SELECT ",
            "1; DROP TABLE"
        ])

def test_xss_fuzzing(ws_fuzzer: WSFuzzer):
    """Test XSS fuzzing patterns."""
    original_msg = WSMessage(
        type=WSMessageType.TEXT,
        data='{"message": "Hello"}',
        metadata={}
    )
    
    # Generate XSS fuzzed messages
    fuzzed = ws_fuzzer.fuzz_xss(original_msg)
    
    assert len(fuzzed) > 0
    for msg in fuzzed:
        assert msg.metadata.get('fuzzing_type') == 'xss'
        data = msg.data if isinstance(msg.data, str) else msg.data.decode('utf-8')
        assert any(pattern in data for pattern in [
            "<script>",
            "javascript:",
            "onerror="
        ])

async def test_fuzzing_replay(
    ws_fuzzer: WSFuzzer,
    ws_conversation: WSConversation, 
    ws_test_client,
    sample_ws_messages: List[WSMessage]
):
    """Test replaying fuzzed messages."""
    # Add original messages
    for msg in sample_ws_messages:
        if msg.type == WSMessageType.TEXT:
            ws_conversation.add_message(msg)
    
    # Generate and add fuzzed messages
    fuzzed = await ws_fuzzer.fuzz_conversation(ws_conversation)
    for msg in fuzzed:
        ws_conversation.add_message(msg)
    
    # Replay conversation with fuzzed messages
    async with ws_test_client.ws_connect("/ws") as ws:
        responses = await ws_conversation.replay_messages(ws, ws)
        
        # Check responses from fuzzed messages
        assert len(responses) > 0
        fuzzed_responses = [r for r in responses 
                          if r.metadata.get('replay') and 
                          r.metadata.get('response_to', {}).get('fuzzed')]
        assert len(fuzzed_responses) > 0

def test_json_structure_fuzzing(ws_fuzzer: WSFuzzer):
    """Test fuzzing of JSON structure."""
    original_msg = WSMessage(
        type=WSMessageType.TEXT,
        data='{"nested": {"field": "value"}, "array": [1,2,3]}',
        metadata={}
    )
    
    # Generate structure fuzzed messages
    fuzzed = ws_fuzzer.fuzz_json_structure(original_msg)
    
    assert len(fuzzed) > 0
    for msg in fuzzed:
        assert msg.metadata.get('fuzzing_type') == 'json_structure'
        # Verify structure is modified but still valid JSON
        try:
            json_data = json.loads(msg.data)
            assert isinstance(json_data, dict)
            assert json_data != json.loads(original_msg.data)
        except json.JSONDecodeError:
            pytest.fail("Fuzzed message is not valid JSON")

def test_protocol_fuzzing(ws_fuzzer: WSFuzzer):
    """Test WebSocket protocol-level fuzzing."""
    original_msg = WSMessage(
        type=WSMessageType.TEXT,
        data="Hello",
        metadata={}
    )
    
    # Generate protocol fuzzed messages
    fuzzed = ws_fuzzer.fuzz_protocol(original_msg)
    
    assert len(fuzzed) > 0
    for msg in fuzzed:
        assert msg.metadata.get('fuzzing_type') == 'protocol'
        # Verify message type or frame modifications
        assert (msg.type != original_msg.type or 
                msg.data != original_msg.data or
                msg.metadata.get('frame_flags') != original_msg.metadata.get('frame_flags'))
