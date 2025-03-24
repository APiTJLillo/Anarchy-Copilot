"""
Tests for WebSocket conversation functionality.
"""
import pytest
from typing import List

from proxy.websocket.types import WSMessage, WSMessageType
from proxy.websocket.conversation import WSConversation, SecurityAnalyzer

async def test_conversation_add_message(ws_conversation: WSConversation, sample_ws_messages: List[WSMessage]):
    """Test adding messages to a conversation."""
    for msg in sample_ws_messages:
        ws_conversation.add_message(msg)
    
    assert len(ws_conversation.messages) == len(sample_ws_messages)
    assert ws_conversation.messages == sample_ws_messages

def test_security_analyzer_sensitive_data(security_analyzer: SecurityAnalyzer):
    """Test security analyzer detection of sensitive data."""
    test_messages = [
        WSMessage(
            type=WSMessageType.TEXT,
            data='{"api_key": "secret123"}',
            metadata={}
        ),
        WSMessage(
            type=WSMessageType.TEXT,
            data='{"user": {"password": "test123"}}',
            metadata={}
        )
    ]
    
    for msg in test_messages:
        findings = security_analyzer.analyze_message(msg)
        assert len(findings) > 0
        assert any(f['severity'] == 'high' for f in findings)

async def test_conversation_replay(ws_conversation: WSConversation, ws_test_client, sample_ws_messages: List[WSMessage]):
    """Test replaying conversation messages."""
    # Add messages to conversation
    for msg in sample_ws_messages:
        ws_conversation.add_message(msg)
    
    # Connect to test websocket
    async with ws_test_client.ws_connect("/ws") as ws:
        # Replay text messages only
        responses = await ws_conversation.replay_messages(ws, ws)
        
        # Verify responses
        assert len(responses) > 0
        for resp in responses:
            assert resp.type == WSMessageType.TEXT
            assert resp.metadata.get('replay') is True

def test_conversation_statistics(ws_conversation: WSConversation, sample_ws_messages: List[WSMessage]):
    """Test conversation statistics calculation."""
    # Add messages with different types
    for msg in sample_ws_messages:
        ws_conversation.add_message(msg)
    
    stats = ws_conversation.get_message_stats()
    
    assert stats['total_messages'] == len(sample_ws_messages)
    assert stats['text_messages'] == len([m for m in sample_ws_messages if m.type == WSMessageType.TEXT])
    assert stats['binary_messages'] == len([m for m in sample_ws_messages if m.type == WSMessageType.BINARY])
    assert 'security_findings' in stats

def test_conversation_pattern_matching(ws_conversation: WSConversation):
    """Test pattern matching in conversation messages."""
    test_messages = [
        WSMessage(
            type=WSMessageType.TEXT,
            data='test message 123',
            metadata={}
        ),
        WSMessage(
            type=WSMessageType.TEXT,
            data='another test 456',
            metadata={}
        ),
        WSMessage(
            type=WSMessageType.TEXT,
            data='no match here',
            metadata={}
        )
    ]
    
    for msg in test_messages:
        ws_conversation.add_message(msg)
    
    # Test pattern matching
    matches = ws_conversation.find_patterns(r'\d+')
    assert len(matches) == 2
    
    # Test no matches
    matches = ws_conversation.find_patterns(r'nonexistent')
    assert len(matches) == 0
