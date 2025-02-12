"""Tests for vulnerability module models."""

import pytest
from datetime import datetime
from vuln_module.models import (
    VulnResult,
    VulnSeverity,
    PayloadType,
    Payload,
    PayloadResult,
    ScanConfig
)

@pytest.fixture
def sample_payload():
    """Create a sample payload for testing."""
    return Payload(
        content="<script>alert(1)</script>",
        type=PayloadType.XSS,
        encoding="none",
        context={"location": "parameter"},
        metadata={"generator": "test"},
        generated_by="test_suite",
        effectiveness_score=0.8
    )

@pytest.fixture
def sample_payload_result(sample_payload):
    """Create a sample payload result for testing."""
    return PayloadResult(
        payload=sample_payload,
        success=True,
        response_data={"status": 200, "body": "<script>alert(1)</script>"},
        context={"url": "http://example.com"}
    )

@pytest.fixture
def sample_vuln_result(sample_payload_result):
    """Create a sample vulnerability result for testing."""
    return VulnResult(
        name="Test XSS",
        type="XSS",
        severity=VulnSeverity.HIGH,
        description="Cross-site scripting vulnerability",
        endpoint="http://example.com/search",
        payloads=[sample_payload_result]
    )

def test_payload_creation():
    """Test Payload class creation and attributes."""
    payload = Payload(
        content="test",
        type=PayloadType.XSS
    )
    assert payload.content == "test"
    assert payload.type == PayloadType.XSS
    assert payload.metadata == {}

def test_payload_result_creation(sample_payload):
    """Test PayloadResult class creation and attributes."""
    result = PayloadResult(
        payload=sample_payload,
        success=True,
        response_data={}
    )
    assert result.success
    assert isinstance(result.timestamp, datetime)
    assert result.error is None

def test_vuln_result_serialization(sample_vuln_result):
    """Test VulnResult serialization and deserialization."""
    # Convert to dict
    data = sample_vuln_result.to_dict()
    
    # Verify dict content
    assert data["name"] == "Test XSS"
    assert data["type"] == "XSS"
    assert data["severity"] == "HIGH"
    
    # Convert back to VulnResult
    reconstructed = VulnResult.from_dict(data)
    
    # Verify object reconstruction
    assert reconstructed.name == sample_vuln_result.name
    assert reconstructed.type == sample_vuln_result.type
    assert reconstructed.severity == sample_vuln_result.severity
    assert len(reconstructed.payloads) == len(sample_vuln_result.payloads)

def test_scan_config_defaults():
    """Test ScanConfig default values."""
    config = ScanConfig(
        target="http://example.com",
        payload_types={PayloadType.XSS}
    )
    assert config.max_depth == 3
    assert config.threads == 10
    assert config.timeout == 30
    assert config.verify_ssl
    assert config.follow_redirects
    assert config.ai_assistance

def test_scan_config_custom_values():
    """Test ScanConfig with custom values."""
    config = ScanConfig(
        target="http://example.com",
        payload_types={PayloadType.XSS, PayloadType.SQLI},
        max_depth=5,
        threads=20,
        timeout=60,
        verify_ssl=False,
        follow_redirects=False,
        ai_assistance=False
    )
    assert config.max_depth == 5
    assert config.threads == 20
    assert config.timeout == 60
    assert not config.verify_ssl
    assert not config.follow_redirects
    assert not config.ai_assistance

def test_vuln_severity_comparison():
    """Test VulnSeverity enum comparison."""
    assert VulnSeverity.CRITICAL > VulnSeverity.HIGH
    assert VulnSeverity.HIGH > VulnSeverity.MEDIUM
    assert VulnSeverity.MEDIUM > VulnSeverity.LOW
    assert VulnSeverity.LOW > VulnSeverity.INFO

def test_payload_type_validation():
    """Test PayloadType validation."""
    with pytest.raises(ValueError):
        # This should fail as INVALID is not a PayloadType
        Payload(content="test", type="INVALID")
