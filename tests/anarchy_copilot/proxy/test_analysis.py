"""Test proxy traffic analysis functionality."""
import pytest
import json
from copy import deepcopy

from proxy.analysis.analyzer import TrafficAnalyzer
from proxy.interceptor import InterceptedRequest, InterceptedResponse

def test_json_analysis():
    """Test analysis of JSON payloads."""
    analyzer = TrafficAnalyzer()
    
    # Create a request with sensitive JSON data
    request = InterceptedRequest(
        id="test-1",
        method="POST",
        url="http://example.com/api",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer abc123"
        },
        body=json.dumps({
            "username": "test",
            "password": "secret123",
            "api_key": "12345",
            "credit_card": "4111-1111-1111-1111"
        }).encode()
    )
    
    # Analyze request
    results = analyzer.analyze_request(request)
    assert len(results) >= 2  # Should find password and credit card

    # Verify password was found
    password_result = next(r for r in results if "password" in r.evidence.lower())
    assert password_result.severity == "high"
    assert "secret123" in password_result.evidence

    # Verify credit card was found
    cc_result = next(r for r in results if "credit_card" in r.evidence.lower())
    assert cc_result.severity == "high"
    assert "4111-1111-1111-1111" in cc_result.evidence

def test_sql_injection_analysis():
    """Test detection of SQL injection attempts."""
    analyzer = TrafficAnalyzer()
    
    # Create a request with SQL injection pattern
    request = InterceptedRequest(
        id="test-2",
        method="GET",
        url="http://example.com/users?id=1 UNION SELECT username,password FROM users--",
        headers={},
        body=None
    )
    
    # Analyze request
    results = analyzer.analyze_request(request)
    assert len(results) >= 1

    # Verify SQL injection was detected
    sql_result = next(r for r in results if "SQL" in r.rule_name)
    assert sql_result.severity == "high"
    assert "UNION SELECT" in sql_result.evidence

def test_error_disclosure_analysis():
    """Test detection of error message disclosure."""
    analyzer = TrafficAnalyzer()
    
    # Create response with error details
    request = InterceptedRequest(
        id="test-3",
        method="GET",
        url="http://example.com/api",
        headers={},
        body=None
    )
    
    response = InterceptedResponse(
        status_code=500,
        headers={"Content-Type": "text/plain"},
        body=b"""Internal Server Error
        Stack trace:
        Error in /var/www/app.py, line 123
        MySQL Error [1045]: Access denied for user 'app'@'localhost'"""
    )
    
    # Analyze response
    results = analyzer.analyze_response(response, request)
    assert len(results) >= 2  # Should find stack trace and SQL error

    # Verify error disclosure was detected
    stack_result = next(r for r in results if "Stack trace" in r.evidence)
    assert stack_result.severity == "medium"
    assert "app.py" in stack_result.evidence

    sql_result = next(r for r in results if "MySQL Error" in r.evidence)
    assert sql_result.severity == "medium"
    assert "Access denied" in sql_result.evidence

def test_xss_analysis():
    """Test detection of XSS attempts."""
    analyzer = TrafficAnalyzer()
    
    # Create request with XSS pattern
    request = InterceptedRequest(
        id="test-4",
        method="POST",
        url="http://example.com/comment",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        body=b"comment=<script>alert(document.cookie)</script>"
    )
    
    # Analyze request
    results = analyzer.analyze_request(request)
    assert len(results) >= 1

    # Verify XSS was detected
    xss_result = next(r for r in results if "XSS" in r.rule_name)
    assert xss_result.severity == "high"
    assert "<script>" in xss_result.evidence
    assert "alert" in xss_result.evidence

def test_information_disclosure():
    """Test detection of information disclosure in headers."""
    analyzer = TrafficAnalyzer()
    
    # Create response with sensitive headers
    request = InterceptedRequest(
        id="test-5",
        method="GET",
        url="http://example.com",
        headers={},
        body=None
    )
    
    response = InterceptedResponse(
        status_code=200,
        headers={
            "Server": "Apache/2.4.41 (Ubuntu)",
            "X-Powered-By": "PHP/7.4.3",
            "X-AspNet-Version": "4.0.30319",
            "Set-Cookie": "PHPSESSID=abc123; path=/"
        },
        body=None
    )
    
    # Analyze response
    results = analyzer.analyze_response(response, request)
    assert len(results) >= 2  # Should find version disclosures

    # Verify server version disclosure was detected
    server_result = next(r for r in results if "Server" in r.evidence)
    assert server_result.severity == "medium"
    assert "Apache/2.4.41" in server_result.evidence

    # Verify framework version disclosure was detected
    framework_result = next(r for r in results if "PHP" in r.evidence)
    assert framework_result.severity == "medium"
    assert "PHP/7.4.3" in framework_result.evidence

def test_multiple_findings():
    """Test handling multiple security findings in one request."""
    analyzer = TrafficAnalyzer()
    
    # Create request with multiple issues
    request = InterceptedRequest(
        id="test-6",
        method="POST",
        url="http://example.com/users?id=1' OR '1'='1",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        },
        body=json.dumps({
            "comment": "<script>alert(1)</script>",
            "ssn": "123-45-6789"
        }).encode()
    )
    
    # Analyze request
    results = analyzer.analyze_request(request)
    
    # Should find SQL injection, XSS, and sensitive data
    assert len(results) >= 3
    
    # Check for each type of finding
    findings = {r.rule_name: r for r in results}
    assert any("SQL" in name for name in findings)
    assert any("XSS" in name for name in findings)
    assert any("Sensitive" in name for name in findings)

    # Check result metadata
    for result in results:
        assert result.request_id == "test-6"
        assert result.timestamp is not None
        assert result.evidence is not None
