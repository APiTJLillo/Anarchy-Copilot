"""
WebSocket interceptor functionality.
"""
import json
import logging
from typing import Optional, Tuple
import aiohttp
from aiohttp import web

from .types import WSMessage, WSMessageType
from .conversation import WSConversation
from ..analysis.analyzer import TrafficAnalyzer

logger = logging.getLogger(__name__)

class WebSocketInterceptor:
    def __init__(self):
        self.is_enabled = True

    async def validate_handshake(self, request: web.Request) -> Tuple[bool, Optional[str]]:
        """Validate WebSocket handshake headers.
        
        Args:
            request: The HTTP upgrade request that initiated the WebSocket
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        headers = request.headers
        
        # Check required headers
        if headers.get('Upgrade', '').lower() != 'websocket':
            return False, "Missing or invalid Upgrade header"
        
        if 'Sec-WebSocket-Key' not in headers:
            return False, "Missing Sec-WebSocket-Key header"
        
        if 'Sec-WebSocket-Version' not in headers:
            return False, "Missing Sec-WebSocket-Version header"
        
        # Check security headers
        if not headers.get('Origin'):
            logger.warning("Missing Origin header in WebSocket request")
        
        return True, None
    
    async def on_connect(self, request: web.Request) -> bool:
        """Called when a WebSocket connection is established.
        
        Args:
            request: The HTTP upgrade request that initiated the WebSocket
            
        Returns:
            True to allow the connection, False to reject it
        """
        is_valid, error = await self.validate_handshake(request)
        if not is_valid:
            logger.error(f"Invalid WebSocket handshake: {error}")
            return False
        return True
    
    async def on_message(self, message: WSMessage, conversation: WSConversation) -> Optional[WSMessage]:
        """Process an intercepted WebSocket message.
        
        Args:
            message: The intercepted message
            conversation: The WebSocket conversation this message belongs to
            
        Returns:
            Modified message or None to drop the message
        """
        return message
    
    async def on_close(self, conversation: WSConversation) -> None:
        """Called when a WebSocket connection is closed.
        
        Args:
            conversation: The completed WebSocket conversation
        """
        pass

class DebugInterceptor(WebSocketInterceptor):
    """Debug interceptor that logs all WebSocket activity."""
    
    async def on_connect(self, request: web.Request) -> bool:
        """Log WebSocket connections."""
        logger.info(f"WebSocket connection established to {request.url}")
        return await super().on_connect(request)
    
    async def on_message(self, message: WSMessage, conversation: WSConversation) -> WSMessage:
        """Log WebSocket messages."""
        if message.type == WSMessageType.TEXT:
            try:
                # Try to pretty print if JSON
                if isinstance(message.data, str):
                    data = json.loads(message.data)
                    logger.info(f"WebSocket message on {conversation.url}:\n{json.dumps(data, indent=2)}")
                else:
                    logger.info(f"WebSocket message on {conversation.url}: {message.data}")
            except json.JSONDecodeError:
                logger.info(f"WebSocket message on {conversation.url}: {message.data}")
        else:
            logger.info(f"WebSocket {message.type.value} message on {conversation.url}")
        return message
    
    async def on_close(self, conversation: WSConversation) -> None:
        """Log WebSocket disconnections."""
        logger.info(f"WebSocket connection closed for {conversation.url}")
        logger.info(f"Total messages: {len(conversation.messages)}")

class SecurityInterceptor(WebSocketInterceptor):
    """Security-focused interceptor for WebSocket traffic."""
    
    def __init__(self):
        """Initialize security interceptor."""
        super().__init__()
        self.analyzer = TrafficAnalyzer()
    
    async def validate_handshake(self, request: web.Request) -> Tuple[bool, Optional[str]]:
        """Enhanced security validation of WebSocket handshake."""
        is_valid, error = await super().validate_handshake(request)
        if not is_valid:
            return is_valid, error
            
        headers = request.headers
        
        # Check for secure connection
        if not request.url.scheme == 'wss':
            logger.warning("WebSocket connection is not secure (ws:// instead of wss://)")
        
        # Validate Origin
        origin = headers.get('Origin')
        if origin:
            # Add your origin validation logic here
            # Example: check against allowed_origins list
            pass
        
        # Check for suspicious headers
        suspicious_headers = {
            'X-Forwarded-For',
            'X-Real-IP',
            'CF-Connecting-IP'
        }
        for header in suspicious_headers:
            if header in headers:
                logger.warning(f"Suspicious header found in WebSocket request: {header}")
        
        return True, None
    
    async def on_message(self, message: WSMessage, conversation: WSConversation) -> Optional[WSMessage]:
        """Security checks for WebSocket messages."""
        if message.type == WSMessageType.TEXT and isinstance(message.data, str):
            # Check message size
            if len(message.data) > 1024 * 1024:  # 1MB
                logger.warning(f"Large WebSocket message detected: {len(message.data)} bytes")
            
            # Deep security analysis
            request_like = {
                'id': message.id,
                'url': conversation.url,
                'headers': {},
                'body': message.data.encode() if isinstance(message.data, str) else message.data
            }
            
            findings = self.analyzer.analyze_request(request_like)
            if findings:
                message.metadata['security_issues'] = [
                    {
                        'severity': finding.severity,
                        'description': finding.description,
                        'evidence': finding.evidence,
                        'rule_name': finding.rule_name
                    }
                    for finding in findings
                ]
                for finding in findings:
                    logger.warning(f"Security issue detected: {finding.description}")
            
        return message

    async def on_close(self, conversation: WSConversation) -> None:
        """Generate security report when connection closes."""
        report = self.analyzer.get_security_report()
        conversation.metadata['security_report'] = report
        if report['finding_count'] > 0:
            logger.warning(f"Security issues found in conversation {conversation.id}: {report['finding_count']} findings")

class RateLimitInterceptor(WebSocketInterceptor):
    """Rate limiting interceptor for WebSocket connections."""
    
    def __init__(self, max_messages_per_second: int = 10):
        """Initialize the rate limiter.
        
        Args:
            max_messages_per_second: Maximum number of messages allowed per second
        """
        super().__init__()
        self.max_messages_per_second = max_messages_per_second
        self._message_counts = {}
    
    async def on_message(self, message: WSMessage, conversation: WSConversation) -> Optional[WSMessage]:
        """Apply rate limiting to messages."""
        # Simple rolling window rate limiting
        now = message.timestamp
        window_start = now.replace(microsecond=0)
        
        if conversation.id not in self._message_counts:
            self._message_counts[conversation.id] = []
        
        # Remove old timestamps
        self._message_counts[conversation.id] = [
            ts for ts in self._message_counts[conversation.id]
            if (now - ts).total_seconds() < 1
        ]
        
        # Check rate limit
        if len(self._message_counts[conversation.id]) >= self.max_messages_per_second:
            logger.warning(f"Rate limit exceeded for conversation {conversation.id}")
            return None
        
        self._message_counts[conversation.id].append(now)
        return message
    
    async def on_close(self, conversation: WSConversation) -> None:
        """Clean up rate limiting data."""
        if conversation.id in self._message_counts:
            del self._message_counts[conversation.id]
