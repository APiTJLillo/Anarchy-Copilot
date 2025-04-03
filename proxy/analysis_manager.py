"""
Traffic Analysis Manager for Anarchy Copilot proxy module.

This module provides a central manager for all traffic analysis functionality,
coordinating between different analysis modules and providing a unified API.
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from .interceptor import InterceptedRequest, InterceptedResponse, ProxyInterceptor
from .analysis import SecurityIssue, TrafficAnalyzer, RealTimeAnalysisInterceptor
from .analysis.monitoring.trend_analysis import TrendAnalyzer, TrendConfig
from .analysis.monitoring.anomaly_analysis import AnomalyDetector, AnomalyConfig
from .websocket.types import WebSocketMessage

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Represents the result of a traffic analysis operation."""
    timestamp: datetime = field(default_factory=datetime.now)
    security_issues: List[SecurityIssue] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    behavior_patterns: Dict[str, Any] = field(default_factory=dict)
    anomalies: Dict[str, Any] = field(default_factory=dict)
    trends: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert analysis result to dictionary format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "security_issues": [issue.to_dict() for issue in self.security_issues],
            "performance_metrics": self.performance_metrics,
            "behavior_patterns": self.behavior_patterns,
            "anomalies": self.anomalies,
            "trends": self.trends,
            "metadata": self.metadata
        }

@dataclass
class BehaviorPattern:
    """Represents a detected behavior pattern in traffic."""
    pattern_type: str  # e.g., "session_hijacking", "data_exfiltration"
    confidence: float  # 0.0 to 1.0
    description: str
    evidence: List[str]
    request_ids: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert pattern to dictionary format."""
        return {
            "pattern_type": self.pattern_type,
            "confidence": self.confidence,
            "description": self.description,
            "evidence": self.evidence,
            "request_ids": self.request_ids,
            "timestamp": self.timestamp.isoformat()
        }

class BehaviorAnalyzer:
    """Analyzes traffic behavior patterns across multiple requests/responses."""
    
    def __init__(self):
        """Initialize the behavior analyzer."""
        self._patterns: List[BehaviorPattern] = []
        self._session_data: Dict[str, Dict[str, Any]] = {}  # session_id -> data
        self._request_sequences: Dict[str, List[str]] = {}  # session_id -> [request_ids]
        
    async def analyze_request_sequence(self, session_id: str) -> List[BehaviorPattern]:
        """Analyze a sequence of requests for behavior patterns."""
        if session_id not in self._request_sequences:
            return []
            
        request_ids = self._request_sequences[session_id]
        if len(request_ids) < 3:  # Need at least 3 requests to analyze patterns
            return []
            
        patterns = []
        
        # Check for rapid succession requests (potential brute force)
        if len(request_ids) > 10:
            # Implementation would analyze timestamps and patterns
            # This is a placeholder for the actual implementation
            patterns.append(BehaviorPattern(
                pattern_type="rapid_succession",
                confidence=0.7,
                description="Multiple requests in rapid succession detected",
                evidence=[f"Sequence of {len(request_ids)} requests"],
                request_ids=request_ids
            ))
            
        # Check for session anomalies
        if session_id in self._session_data:
            session_data = self._session_data[session_id]
            # Implementation would analyze session data for anomalies
            # This is a placeholder for the actual implementation
            
        return patterns
        
    async def track_request(self, request: InterceptedRequest, session_id: Optional[str] = None):
        """Track a request for behavior analysis."""
        if not session_id:
            # Extract session ID from cookies or other identifiers
            session_id = self._extract_session_id(request)
            
        if session_id:
            if session_id not in self._request_sequences:
                self._request_sequences[session_id] = []
            self._request_sequences[session_id].append(request.id)
            
            # Update session data
            if session_id not in self._session_data:
                self._session_data[session_id] = {}
            
            # Extract and store relevant data from request
            self._update_session_data(session_id, request)
    
    def _extract_session_id(self, request: InterceptedRequest) -> Optional[str]:
        """Extract session ID from request."""
        # Check cookies for session identifier
        if "cookie" in request.headers:
            cookie_header = request.headers["cookie"]
            cookies = {
                k.strip(): v.strip() 
                for k, v in [
                    cookie.split("=", 1) 
                    for cookie in cookie_header.split(";")
                ]
            }
            
            # Common session cookie names
            for name in ["sessionid", "session", "sid", "PHPSESSID", "JSESSIONID"]:
                if name in cookies:
                    return cookies[name]
        
        # Check authorization header
        if "authorization" in request.headers:
            auth_header = request.headers["authorization"]
            if auth_header.startswith("Bearer "):
                return auth_header[7:]  # Use token as session ID
                
        return None
        
    def _update_session_data(self, session_id: str, request: InterceptedRequest):
        """Update session data with information from request."""
        session_data = self._session_data[session_id]
        
        # Track endpoints accessed
        if "endpoints" not in session_data:
            session_data["endpoints"] = []
        session_data["endpoints"].append(request.path)
        
        # Track user agent
        if "user-agent" in request.headers:
            session_data["user_agent"] = request.headers["user-agent"]
            
        # Track IP address
        if hasattr(request, "client_address"):
            session_data["ip_address"] = request.client_address
            
        # Track request methods
        if "methods" not in session_data:
            session_data["methods"] = []
        if request.method not in session_data["methods"]:
            session_data["methods"].append(request.method)

class TrafficAnalysisManager:
    """Central manager for all traffic analysis functionality."""
    
    def __init__(self):
        """Initialize the traffic analysis manager."""
        self.security_analyzer = TrafficAnalyzer()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.trend_analyzer = None  # Will be initialized when needed
        self.anomaly_detector = None  # Will be initialized when needed
        
        self._analysis_results: Dict[str, AnalysisResult] = {}  # request_id -> result
        self._websocket_messages: Dict[str, List[WebSocketMessage]] = {}  # connection_id -> messages
        
    async def analyze_request(self, request: InterceptedRequest) -> AnalysisResult:
        """Analyze a request for security issues and behavior patterns."""
        # Create analysis result
        result = AnalysisResult(
            metadata={
                "request_id": request.id,
                "method": request.method,
                "path": request.path,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Security analysis
        security_issues = self.security_analyzer.analyze_request(request)
        result.security_issues.extend(security_issues)
        
        # Behavior tracking
        await self.behavior_analyzer.track_request(request)
        
        # Store result
        self._analysis_results[request.id] = result
        
        return result
        
    async def analyze_response(self, response: InterceptedResponse, request: InterceptedRequest) -> AnalysisResult:
        """Analyze a response for security issues and behavior patterns."""
        # Get or create analysis result
        if request.id in self._analysis_results:
            result = self._analysis_results[request.id]
        else:
            result = AnalysisResult(
                metadata={
                    "request_id": request.id,
                    "method": request.method,
                    "path": request.path,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Security analysis
        security_issues = self.security_analyzer.analyze_response(response, request)
        result.security_issues.extend(security_issues)
        
        # Update result
        self._analysis_results[request.id] = result
        
        return result
        
    async def analyze_websocket_message(self, message: WebSocketMessage, connection_id: str) -> AnalysisResult:
        """Analyze a WebSocket message for security issues and behavior patterns."""
        # Track message
        if connection_id not in self._websocket_messages:
            self._websocket_messages[connection_id] = []
        self._websocket_messages[connection_id].append(message)
        
        # Create analysis result
        result = AnalysisResult(
            metadata={
                "connection_id": connection_id,
                "message_id": message.id,
                "direction": message.direction,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Security analysis
        # Convert WebSocket message to a format that can be analyzed
        if message.content:
            try:
                # Try to parse as JSON
                content = json.loads(message.content)
                # Analyze JSON content for security issues
                # This is a placeholder for actual implementation
            except json.JSONDecodeError:
                # Analyze as text
                if isinstance(message.content, bytes):
                    text_content = message.content.decode('utf-8', errors='ignore')
                else:
                    text_content = message.content
                # Analyze text content for security issues
                # This is a placeholder for actual implementation
        
        # Store result
        self._analysis_results[message.id] = result
        
        return result
        
    async def analyze_behavior_patterns(self, session_id: Optional[str] = None) -> List[BehaviorPattern]:
        """Analyze behavior patterns across multiple requests."""
        if session_id:
            return await self.behavior_analyzer.analyze_request_sequence(session_id)
        
        # If no session ID provided, analyze all sessions
        all_patterns = []
        for session_id in self.behavior_analyzer._request_sequences.keys():
            patterns = await self.behavior_analyzer.analyze_request_sequence(session_id)
            all_patterns.extend(patterns)
            
        return all_patterns
        
    async def get_analysis_result(self, request_id: str) -> Optional[AnalysisResult]:
        """Get analysis result for a specific request."""
        return self._analysis_results.get(request_id)
        
    async def get_all_analysis_results(self) -> Dict[str, AnalysisResult]:
        """Get all analysis results."""
        return self._analysis_results
        
    async def get_security_issues(self, severity: Optional[str] = None) -> List[SecurityIssue]:
        """Get all detected security issues, optionally filtered by severity."""
        issues = []
        for result in self._analysis_results.values():
            if severity:
                issues.extend([issue for issue in result.security_issues if issue.severity == severity])
            else:
                issues.extend(result.security_issues)
                
        return issues
        
    async def get_behavior_patterns(self, pattern_type: Optional[str] = None) -> List[BehaviorPattern]:
        """Get all detected behavior patterns, optionally filtered by type."""
        patterns = await self.analyze_behavior_patterns()
        
        if pattern_type:
            return [pattern for pattern in patterns if pattern.pattern_type == pattern_type]
        return patterns
        
    async def clear_analysis_data(self):
        """Clear all analysis data."""
        self._analysis_results.clear()
        self._websocket_messages.clear()
        self.behavior_analyzer._session_data.clear()
        self.behavior_analyzer._request_sequences.clear()
        self.behavior_analyzer._patterns.clear()

class EnhancedAnalysisInterceptor(ProxyInterceptor):
    """Enhanced interceptor that performs comprehensive analysis of proxy traffic."""
    
    def __init__(self):
        """Initialize the enhanced analysis interceptor."""
        self.manager = TrafficAnalysisManager()
    
    async def intercept_request(self, request: InterceptedRequest) -> InterceptedRequest:
        """Analyze intercepted requests."""
        await self.manager.analyze_request(request)
        return request
    
    async def intercept_response(self, response: InterceptedResponse, request: InterceptedRequest) -> InterceptedResponse:
        """Analyze intercepted responses."""
        await self.manager.analyze_response(response, request)
        return response
    
    async def get_analysis_results(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analysis results."""
        if request_id:
            result = await self.manager.get_analysis_result(request_id)
            return result.to_dict() if result else {}
        
        results = {}
        for req_id, result in (await self.manager.get_all_analysis_results()).items():
            results[req_id] = result.to_dict()
            
        return results
    
    async def get_security_issues(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get security issues."""
        issues = await self.manager.get_security_issues(severity)
        return [issue.to_dict() for issue in issues]
    
    async def get_behavior_patterns(self, pattern_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get behavior patterns."""
        patterns = await self.manager.get_behavior_patterns(pattern_type)
        return [pattern.to_dict() for pattern in patterns]
