"""Security analysis interceptor for WebSocket traffic."""
from typing import Optional
from .interceptor import WebSocketInterceptor
from .types import WSMessage
from .conversation import WSConversation
from ..analysis.analyzer import TrafficAnalyzer

class SecurityAnalysisInterceptor(WebSocketInterceptor):
    """Interceptor that performs security analysis on WebSocket messages."""

    def __init__(self):
        """Initialize the security analysis interceptor."""
        super().__init__()
        self.analyzer = TrafficAnalyzer()

    async def on_message(self, message: WSMessage, conversation: WSConversation) -> Optional[WSMessage]:
        """
        Analyze message for security issues.
        
        Args:
            message: The WebSocket message to analyze
            conversation: The current WebSocket conversation
            
        Returns:
            The original message with added security findings metadata
        """
        if not self.is_enabled:
            return message

        # Create a request-like object for the analyzer
        request_like = {
            'id': message.id,
            'url': conversation.url,
            'headers': {},
            'body': message.data if isinstance(message.data, bytes) else message.data.encode()
        }

        # Analyze the message
        findings = self.analyzer.analyze_request(request_like)

        # Add any findings to message metadata
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

        return message

    async def on_close(self, conversation: WSConversation) -> None:
        """Generate final security report when conversation closes."""
        if self.is_enabled:
            report = self.analyzer.get_security_report()
            conversation.metadata['security_report'] = report
