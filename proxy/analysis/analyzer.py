"""Proxy traffic analyzer."""
from typing import Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
import json

@dataclass
class SecurityFinding:
    """Security finding details."""
    rule_name: str
    severity: str
    evidence: str
    description: str
    request_id: str
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        """Convert finding to dictionary format."""
        return {
            'rule_name': self.rule_name,
            'severity': self.severity,
            'evidence': self.evidence,
            'description': self.description,
            'request_id': self.request_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class TrafficAnalyzer:
    """Analyzes proxy traffic for security issues."""

    def __init__(self):
        self.findings: List[SecurityFinding] = []

    def analyze_request(self, request) -> List[SecurityFinding]:
        """
        Analyze an intercepted request for security issues.
        
        Args:
            request: The intercepted request to analyze
            
        Returns:
            List of security findings
        """
        findings = []

        # Check URL for SQL injection
        if any(pattern in request.url.lower() for pattern in 
               ['union select', "' or '", '" or "', 'select from', 'delete from']):
            findings.append(SecurityFinding(
                rule_name="SQL_INJECTION",
                severity="high",
                evidence=request.url,
                description="SQL injection pattern detected in URL",
                request_id=request.id
            ))

        # Check body for various issues
        if request.body:
            try:
                body_content = request.body.decode()
                
                # Check for JSON content
                if request.headers.get('Content-Type', '').startswith('application/json'):
                    body_data = json.loads(body_content)
                    for key, value in self._flatten_dict(body_data).items():
                        # Check for sensitive data
                        if self._is_sensitive_field(key, str(value)):
                            findings.append(SecurityFinding(
                                rule_name="SENSITIVE_DATA",
                                severity="high",
                                evidence=f"{key}: {value}",
                                description=f"Sensitive data found in field '{key}'",
                                request_id=request.id
                            ))

                # Check for XSS patterns
                if any(pattern in body_content.lower() for pattern in 
                      ['<script', 'javascript:', 'onerror=', 'onload=', 'alert(']):
                    findings.append(SecurityFinding(
                        rule_name="XSS",
                        severity="high",
                        evidence=body_content,
                        description="XSS pattern detected in request body",
                        request_id=request.id
                    ))

            except (UnicodeDecodeError, json.JSONDecodeError):
                pass  # Skip binary or invalid JSON data

        self.findings.extend(findings)
        return findings

    def analyze_response(self, response, request) -> List[SecurityFinding]:
        """
        Analyze an intercepted response for security issues.
        
        Args:
            response: The intercepted response to analyze
            request: The original request
            
        Returns:
            List of security findings
        """
        findings = []

        # Check for error disclosures
        if response.body:
            try:
                body_content = response.body.decode()

                if "stack trace" in body_content.lower():
                    findings.append(SecurityFinding(
                        rule_name="STACK_TRACE_DISCLOSURE",
                        severity="medium",
                        evidence=body_content,
                        description="Stack trace disclosed in error response",
                        request_id=request.id
                    ))

                if any(error in body_content for error in ["MySQL Error", "ORA-", "SQLSTATE"]):
                    findings.append(SecurityFinding(
                        rule_name="SQL_ERROR_DISCLOSURE",
                        severity="medium",
                        evidence=body_content,
                        description="SQL error message disclosed in response",
                        request_id=request.id
                    ))

            except UnicodeDecodeError:
                pass

        # Check headers for version disclosure
        for header, value in response.headers.items():
            if header.lower() in ['server', 'x-powered-by', 'x-aspnet-version']:
                findings.append(SecurityFinding(
                    rule_name="VERSION_DISCLOSURE",
                    severity="medium",
                    evidence=f"{header}: {value}",
                    description=f"Software version disclosed in {header} header",
                    request_id=request.id
                ))

        self.findings.extend(findings)
        return findings

    def get_security_report(self) -> Dict:
        """
        Generate a security report from all findings.
        
        Returns:
            Security report containing all findings
        """
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'finding_count': len(self.findings),
            'findings': [finding.to_dict() for finding in self.findings],
            'summary': {
                'critical': len([f for f in self.findings if f.severity == 'critical']),
                'high': len([f for f in self.findings if f.severity == 'high']),
                'medium': len([f for f in self.findings if f.severity == 'medium']),
                'low': len([f for f in self.findings if f.severity == 'low'])
            }
        }

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _is_sensitive_field(self, key: str, value: str) -> bool:
        """Check if a field contains sensitive information."""
        sensitive_keys = {
            'password', 'secret', 'api_key', 'token', 'auth',
            'ssn', 'social', 'credit_card', 'card_number', 'cvv',
            'access_token', 'refresh_token', 'private_key'
        }
        return any(s in key.lower() for s in sensitive_keys)
