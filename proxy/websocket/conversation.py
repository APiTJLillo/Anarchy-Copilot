"""
WebSocket conversation tracking and analysis.
"""
# Disable Pylance type checking for regex pattern matching operations
# which trigger false positive 'Expected class' errors
# pyright: reportGeneralTypeIssues=false
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, FrozenSet, List, Optional, Pattern, Set, Tuple
import aiohttp
from aiohttp import web

from .types import WSMessage, WSMessageType

logger = logging.getLogger(__name__)

class SecurityPattern:
    """Represents a security pattern to match in WebSocket messages."""
    
    def __init__(self, name: str, pattern: str, severity: str = "medium", description: str = ""):
        """Initialize a security pattern.
        
        Args:
            name: Name of the pattern
            pattern: Regular expression pattern
            severity: Severity level (low/medium/high)
            description: Description of what the pattern detects
        """
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.severity = severity
        self.description = description

class SecurityAnalyzer:
    """Analyzes WebSocket messages for security issues."""

    # Common security patterns to detect
    PATTERNS = [
        SecurityPattern(
            "jwt_token",
            r'eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*',
            "high",
            "JWT token detected in message"
        ),
        SecurityPattern(
            "api_key",
            r'[Aa]pi[_-]?[Kk]ey["\']?\s*[:=]\s*["\']?[\w\-]+["\']?',
            "high",
            "API key detected in message"
        ),
        SecurityPattern(
            "sql_injection",
            r'\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP)\b.*\b(FROM|TABLE|DATABASE)\b',
            "high",
            "Possible SQL injection pattern"
        ),
        SecurityPattern(
            "xss",
            r'<script>|javascript:|<img[^>]+onerror=',
            "high",
            "Possible XSS payload"
        ),
        SecurityPattern(
            "path_traversal",
            r'\.\./|\\\.\\',
            "medium",
            "Directory traversal attempt"
        ),
        SecurityPattern(
            "debug_info",
            r'(stack trace:|debug:|error:)',
            "low",
            "Debug/error information leak"
        ),
    ]

    def analyze_message(self, message: WSMessage) -> List[Dict]:
        """Analyze a message for security issues.
        
        Args:
            message: WebSocket message to analyze
            
        Returns:
            List of security findings
        """
        findings = []
        
        if message.is_text():
            data = message.data if isinstance(message.data, str) else message.data.decode('utf-8', errors='ignore')
            
            # Check all security patterns
            for pattern in self.PATTERNS:
                if pattern.pattern.search(data):
                    findings.append({
                        'type': 'pattern_match',
                        'name': pattern.name,
                        'severity': pattern.severity,
                        'description': pattern.description,
                        'timestamp': message.timestamp.isoformat()
                    })
            
            # JSON-specific analysis
            try:
                json_data = json.loads(data)
                findings.extend(self._analyze_json(json_data))
            except json.JSONDecodeError:
                pass
                
        return findings
    
    # Regex pattern for sensitive fields
    SENSITIVE_PATTERN: Pattern[str] = re.compile(
        r'(?:password|token|key|secret|auth)',
        re.IGNORECASE
    )

    def _matches_pattern(self, pattern: Pattern[str], text: str) -> bool:
        """Check if a text matches a regex pattern.
        
        Args:
            pattern: The pattern to check
            text: Text to check
        
        Returns:
            True if pattern matches, False otherwise
        """
        match = pattern.search(text)
        if match is None:
            return False
        return True

    def _contains_sensitive_term(self, text: str) -> bool:
        """Check if text contains any sensitive field names.
        
        Args:
            text: Text to check
            
        Returns:
            True if sensitive field found, False otherwise
        """
        return self._matches_pattern(self.SENSITIVE_PATTERN, text)
        
    def _analyze_json(self, data: Dict) -> List[Dict]:
        """Analyze JSON data for security issues.
        
        Args:
            data: JSON data to analyze
            
        Returns:
            List of security findings
        """
        findings = []
        
        def check_value(value: any, path: str = ""):
            if isinstance(value, dict):
                for k, v in value.items():
                    check_value(v, f"{path}.{k}" if path else k)
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    check_value(v, f"{path}[{i}]")
            elif isinstance(value, str):
                # Check for sensitive field names
                if self._contains_sensitive_term(path):
                    findings.append({
                        'type': 'sensitive_field',
                        'field': path,
                        'severity': 'high',
                        'description': f'Sensitive data found in field: {path}'
                    })
                
                # Check for serialized data
                if value.startswith('{"') and value.endswith('}'):
                    try:
                        nested = json.loads(value)
                        findings.extend(self._analyze_json(nested))
                    except json.JSONDecodeError:
                        pass
        
        check_value(data)
        return findings

@dataclass
class WSConversation:
    """Represents a WebSocket conversation history."""
    id: str
    url: str
    messages: List[WSMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    security_findings: List[Dict] = field(default_factory=list)
    _analyzer: SecurityAnalyzer = field(default_factory=SecurityAnalyzer)
    
    def add_message(self, message: WSMessage) -> None:
        """Add a message to the conversation history and analyze it.
        
        Args:
            message: WebSocket message to add
        """
        self.messages.append(message)
        findings = self._analyzer.analyze_message(message)
        if findings:
            self.security_findings.extend(findings)
    
    async def replay_messages(
        self,
        client_ws: web.WebSocketResponse,
        server_ws: aiohttp.ClientWebSocketResponse,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        delay: float = 0.0
    ) -> List[WSMessage]:
        """Replay a sequence of messages from the conversation history.
        
        Args:
            client_ws: Client WebSocket connection
            server_ws: Server WebSocket connection
            start_idx: Starting message index to replay
            end_idx: Ending message index to replay (exclusive)
            delay: Delay between messages in seconds
            
        Returns:
            List of responses received from replaying messages
        """
        responses: List[WSMessage] = []
        messages_to_replay = self.messages[start_idx:end_idx]
        
        for msg in messages_to_replay:
            try:
                # Add delay if specified
                if delay > 0:
                    await asyncio.sleep(delay)
                
                # Send the message
                if msg.type == WSMessageType.TEXT:
                    await client_ws.send_str(msg.data)
                elif msg.type == WSMessageType.BINARY:
                    await client_ws.send_bytes(msg.data)
                elif msg.type == WSMessageType.PING:
                    await client_ws.ping(msg.data)
                elif msg.type == WSMessageType.PONG:
                    await client_ws.pong(msg.data)
                elif msg.type == WSMessageType.CLOSE:
                    await client_ws.close()
                    break
                
                # Wait for response with timeout
                resp = await server_ws.receive()
                if resp.type in {WSMessageType.TEXT.value, WSMessageType.BINARY.value}:
                    response = WSMessage(
                        WSMessageType(resp.type),
                        resp.data,
                        metadata={'replay': True}
                    )
                    responses.append(response)
                    
            except Exception as e:
                logger.error(f"Error replaying message: {e}")
                break
            
        return responses
    
    def find_patterns(self, pattern: str) -> List[WSMessage]:
        """Find messages matching a regex pattern.
        
        Args:
            pattern: Regular expression pattern to match
            
        Returns:
            List of matching messages
        """
        matches = []
        try:
            regex = re.compile(pattern)
            for msg in self.messages:
                if msg.type == WSMessageType.TEXT:
                    data = msg.data if isinstance(msg.data, str) else msg.data.decode('utf-8', errors='ignore')
                    if regex.search(data):
                        matches.append(msg)
        except re.error as e:
            logger.error(f"Invalid regex pattern: {e}")
        
        return matches
    
    def get_message_stats(self) -> Dict:
        """Get statistics about the conversation messages.
        
        Returns:
            Dictionary of message statistics
        """
        stats = {
            'total_messages': len(self.messages),
            'text_messages': len([m for m in self.messages if m.type == WSMessageType.TEXT]),
            'binary_messages': len([m for m in self.messages if m.type == WSMessageType.BINARY]),
            'control_messages': len([m for m in self.messages if m.is_control()]),
            'fuzzed_messages': len([m for m in self.messages if m.is_fuzzing]),
            'security_findings': len(self.security_findings),
            'duration': (self.closed_at - self.created_at).total_seconds() if self.closed_at else None,
        }
        
        # Add severity counts
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for finding in self.security_findings:
            severity = finding.get('severity', 'medium')
            severity_counts[severity] += 1
        stats['security_findings_by_severity'] = severity_counts
        
        return stats
    
    def to_dict(self) -> dict:
        """Convert conversation to a dictionary format.
        
        Returns:
            Dictionary representation of the conversation
        """
        return {
            'id': self.id,
            'url': self.url,
            'created_at': self.created_at.isoformat(),
            'closed_at': self.closed_at.isoformat() if self.closed_at else None,
            'messages': [msg.to_dict() for msg in self.messages],
            'security_findings': self.security_findings,
            'stats': self.get_message_stats()
        }
