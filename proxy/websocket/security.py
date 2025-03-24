"""Security analysis for WebSocket conversations."""
from typing import Dict, List, Any, Optional
from .types import WSMessage, MessageType
from .conversation import WSConversation

class SecurityAnalyzer:
    """Analyzes WebSocket conversations for security concerns."""

    def __init__(self):
        """Initialize the security analyzer."""
        self._sensitive_patterns = [
            "password",
            "token",
            "secret",
            "key",
            "auth",
            "credentials"
        ]
        self._sql_patterns = [
            "SELECT", "INSERT", "UPDATE", "DELETE",
            "DROP", "UNION", "WHERE", "FROM"
        ]
        self._xss_patterns = [
            "<script>",
            "javascript:",
            "onerror=",
            "onload="
        ]
        self._injection_patterns = [
            "eval(",
            "setTimeout(",
            "setInterval(",
            "Function(",
            "constructor"
        ]
        self._path_traversal_patterns = [
            "../", "..\\",
            "/etc/passwd",
            "c:\\windows",
            "file:///"
        ]

    def analyze_message(self, message: WSMessage) -> Dict[str, Any]:
        """Analyze a single message for security concerns.
        
        Args:
            message: Message to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        if message.type != MessageType.TEXT:
            return {"safe": True, "concerns": []}

        concerns = []
        data = str(message.data).lower()

        # Check for sensitive data
        if any(pattern in data for pattern in self._sensitive_patterns):
            concerns.append("sensitive_data")

        # Check for SQL injection
        if any(pattern.lower() in data for pattern in self._sql_patterns):
            concerns.append("sql_injection")

        # Check for XSS
        if any(pattern.lower() in data for pattern in self._xss_patterns):
            concerns.append("xss")

        # Check for code injection
        if any(pattern.lower() in data for pattern in self._injection_patterns):
            concerns.append("code_injection")

        # Check for path traversal
        if any(pattern.lower() in data for pattern in self._path_traversal_patterns):
            concerns.append("path_traversal")

        return {
            "safe": len(concerns) == 0,
            "concerns": concerns,
            "message_id": str(message.id),
            "timestamp": message.timestamp.isoformat() if message.timestamp else None
        }

    def analyze_conversation(self, conversation: WSConversation) -> Dict[str, Any]:
        """Analyze a complete conversation for security concerns.
        
        Args:
            conversation: Conversation to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        results = []
        concerns = set()
        binary_messages = 0
        large_messages = 0
        message_count = len(conversation.messages)

        for message in conversation.messages:
            if message.type == MessageType.BINARY:
                binary_messages += 1
                continue

            if message.type == MessageType.TEXT:
                if len(str(message.data)) > 8192:  # Flag large messages
                    large_messages += 1

            analysis = self.analyze_message(message)
            if not analysis["safe"]:
                results.append(analysis)
                concerns.update(analysis["concerns"])

        return {
            "safe": len(concerns) == 0,
            "concerns": list(concerns),
            "suspicious_messages": results,
            "stats": {
                "total_messages": message_count,
                "binary_messages": binary_messages,
                "large_messages": large_messages,
                "suspicious_message_count": len(results)
            },
            "conversation_id": conversation.id,
            "url": conversation.url,
            "metadata": conversation.metadata
        }

    def get_remediation_advice(self, concerns: List[str]) -> Dict[str, List[str]]:
        """Get remediation advice for identified concerns.
        
        Args:
            concerns: List of identified security concerns
            
        Returns:
            Dictionary mapping concerns to remediation steps
        """
        advice: Dict[str, List[str]] = {}
        
        if "sensitive_data" in concerns:
            advice["sensitive_data"] = [
                "Avoid sending sensitive data in plain text",
                "Use secure protocols for authentication",
                "Consider encrypting sensitive payloads"
            ]
            
        if "sql_injection" in concerns:
            advice["sql_injection"] = [
                "Use parameterized queries",
                "Validate and sanitize input",
                "Implement proper input escaping"
            ]
            
        if "xss" in concerns:
            advice["xss"] = [
                "Sanitize user input",
                "Use Content Security Policy headers",
                "Encode output appropriately"
            ]
            
        if "code_injection" in concerns:
            advice["code_injection"] = [
                "Avoid using eval() and similar functions",
                "Implement strict input validation",
                "Use safer alternatives like JSON.parse()"
            ]
            
        if "path_traversal" in concerns:
            advice["path_traversal"] = [
                "Validate file paths",
                "Use path canonicalization",
                "Implement proper access controls"
            ]
            
        return advice
