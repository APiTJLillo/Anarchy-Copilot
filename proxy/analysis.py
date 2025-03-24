"""
Real-time traffic analysis for Anarchy Copilot proxy module.

This module provides real-time analysis of proxy traffic to identify potential
security issues, patterns, and anomalies.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

from .interceptor import InterceptedRequest, InterceptedResponse, ProxyInterceptor

logger = logging.getLogger(__name__)

@dataclass
class SecurityIssue:
    """Represents a potential security issue found in traffic."""
    severity: str  # high, medium, low
    type: str  # e.g., "Sensitive Data Exposure", "SQL Injection Pattern"
    description: str
    evidence: str
    request_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert issue to dictionary format."""
        return {
            "severity": self.severity,
            "type": self.type,
            "description": self.description,
            "evidence": self.evidence,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat()
        }

class TrafficAnalyzer:
    """Analyzes proxy traffic for security issues and patterns."""
    
    def __init__(self):
        """Initialize the traffic analyzer."""
        self._issues: List[SecurityIssue] = []
        self._sensitive_patterns = {
            'jwt': re.compile(r'eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*'),
            'api_key': re.compile(r'(?i)(api[-_]?key|access[-_]?token)["\']?\s*[:=]\s*["\']?([^"\'\s]+)'),
            'password': re.compile(r'(?i)(password|passwd|pwd)["\']?\s*[:=]\s*["\']?([^"\'\s]+)'),
            'ssn': re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'),
            'sql_injection': re.compile(r'(?i)(union\s+select|select\s+.*\s+from|insert\s+into|update\s+.*\s+set|delete\s+from)'),
            'xss': re.compile(r'(?i)(<script>|javascript:|onerror=|onload=|\balert\s*\()')
        }

    def analyze_request(self, request: InterceptedRequest) -> List[SecurityIssue]:
        """Analyze a request for potential security issues.
        
        Args:
            request: The intercepted request to analyze
            
        Returns:
            List of identified security issues
        """
        issues = []
        
        # Analyze URL parameters
        if request.query_params:
            for param_name, param_values in request.query_params.items():
                for value in param_values:
                    issues.extend(self._check_parameter(param_name, value, request.id))

        # Analyze request body
        if request.body:
            try:
                # Try parsing as JSON
                json_data = json.loads(request.body)
                issues.extend(self._analyze_json_data(json_data, request.id))
            except json.JSONDecodeError:
                # Analyze as plain text/form data
                if isinstance(request.body, bytes):
                    body_text = request.body.decode('utf-8', errors='ignore')
                else:
                    body_text = request.body
                issues.extend(self._analyze_text(body_text, request.id))

        return issues

    def analyze_response(self, response: InterceptedResponse, request: InterceptedRequest) -> List[SecurityIssue]:
        """Analyze a response for potential security issues.
        
        Args:
            response: The intercepted response to analyze
            request: The original request that generated this response
            
        Returns:
            List of identified security issues
        """
        issues = []
        
        # Check for sensitive headers
        for header, value in response.headers.items():
            if header.lower() in ['server', 'x-powered-by']:
                issues.append(SecurityIssue(
                    severity="low",
                    type="Information Disclosure",
                    description=f"Response includes version information in {header} header",
                    evidence=f"{header}: {value}",
                    request_id=request.id
                ))

        # Check response body
        if response.body:
            try:
                # Try parsing as JSON
                json_data = json.loads(response.body)
                issues.extend(self._analyze_json_data(json_data, request.id))
            except json.JSONDecodeError:
                # Analyze as plain text
                if isinstance(response.body, bytes):
                    body_text = response.body.decode('utf-8', errors='ignore')
                else:
                    body_text = response.body
                issues.extend(self._analyze_text(body_text, request.id))

        # Check for error messages
        error_patterns = [
            (r'(?i)(sql|mysql|postgresql|oracle)\s+error', "SQL Error Disclosure"),
            (r'(?i)stack\s+trace', "Stack Trace Disclosure"),
            (r'(?i)(exception|error)\s+details?:', "Error Details Disclosure"),
            (r'(?i)debug\s+information', "Debug Information Disclosure")
        ]
        
        if response.body:
            body_text = response.body.decode('utf-8', errors='ignore') if isinstance(response.body, bytes) else response.body
            for pattern, issue_type in error_patterns:
                if re.search(pattern, body_text, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        severity="medium",
                        type=issue_type,
                        description=f"Response contains {issue_type.lower()}",
                        evidence=self._extract_context(body_text, pattern),
                        request_id=request.id
                    ))

        return issues

    def _check_parameter(self, name: str, value: str, request_id: str) -> List[SecurityIssue]:
        """Check a parameter for potential security issues."""
        issues = []
        
        # Check for common attack patterns
        for pattern_name, pattern in self._sensitive_patterns.items():
            if pattern.search(value):
                severity = "high" if pattern_name in ['jwt', 'api_key', 'password'] else "medium"
                issues.append(SecurityIssue(
                    severity=severity,
                    type=f"Sensitive Data: {pattern_name}",
                    description=f"Parameter '{name}' contains sensitive {pattern_name} pattern",
                    evidence=f"{name}={value}",
                    request_id=request_id
                ))
                
        return issues

    def _analyze_json_data(self, data: dict, request_id: str) -> List[SecurityIssue]:
        """Recursively analyze JSON data for security issues."""
        issues = []
        
        def analyze_value(key: str, value: any):
            if isinstance(value, str):
                for pattern_name, pattern in self._sensitive_patterns.items():
                    if pattern.search(value):
                        severity = "high" if pattern_name in ['jwt', 'api_key', 'password'] else "medium"
                        issues.append(SecurityIssue(
                            severity=severity,
                            type=f"Sensitive Data: {pattern_name}",
                            description=f"JSON field '{key}' contains sensitive {pattern_name} pattern",
                            evidence=f"{key}: {value}",
                            request_id=request_id
                        ))
            elif isinstance(value, dict):
                for k, v in value.items():
                    analyze_value(f"{key}.{k}", v)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    analyze_value(f"{key}[{i}]", item)
        
        for key, value in data.items():
            analyze_value(key, value)
        
        return issues

    def _analyze_text(self, text: str, request_id: str) -> List[SecurityIssue]:
        """Analyze text content for security issues."""
        issues = []
        
        for pattern_name, pattern in self._sensitive_patterns.items():
            for match in pattern.finditer(text):
                severity = "high" if pattern_name in ['jwt', 'api_key', 'password'] else "medium"
                issues.append(SecurityIssue(
                    severity=severity,
                    type=f"Sensitive Data: {pattern_name}",
                    description=f"Content contains sensitive {pattern_name} pattern",
                    evidence=self._extract_context(text, match.group()),
                    request_id=request_id
                ))
        
        return issues

    def _extract_context(self, text: str, pattern: str, context_chars: int = 50) -> str:
        """Extract surrounding context for a matched pattern."""
        match_pos = text.find(pattern)
        if match_pos == -1:
            return pattern
        
        start = max(0, match_pos - context_chars)
        end = min(len(text), match_pos + len(pattern) + context_chars)
        
        context = text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
            
        return context

class RealTimeAnalysisInterceptor(ProxyInterceptor):
    """Interceptor that performs real-time analysis of proxy traffic."""
    
    def __init__(self):
        """Initialize the analysis interceptor."""
        self.analyzer = TrafficAnalyzer()
        self._issues: Dict[str, List[SecurityIssue]] = {}
    
    async def intercept_request(self, request: InterceptedRequest) -> InterceptedRequest:
        """Analyze intercepted requests."""
        issues = self.analyzer.analyze_request(request)
        if issues:
            self._issues[request.id] = issues
            for issue in issues:
                logger.warning(f"Security issue detected in request: {issue.type} ({issue.severity})")
        return request
    
    async def intercept_response(self, response: InterceptedResponse, request: InterceptedRequest) -> InterceptedResponse:
        """Analyze intercepted responses."""
        issues = self.analyzer.analyze_response(response, request)
        if issues:
            if request.id not in self._issues:
                self._issues[request.id] = []
            self._issues[request.id].extend(issues)
            for issue in issues:
                logger.warning(f"Security issue detected in response: {issue.type} ({issue.severity})")
        return response
    
    def get_issues(self, request_id: Optional[str] = None) -> List[SecurityIssue]:
        """Get detected security issues.
        
        Args:
            request_id: Optional request ID to filter issues for
            
        Returns:
            List of security issues
        """
        if request_id:
            return self._issues.get(request_id, [])
        return [issue for issues in self._issues.values() for issue in issues]
