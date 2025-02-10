"""Shared fixtures and helpers for vulnerability module tests."""

import pytest
import time
from typing import Dict, Any
from datetime import datetime

from ..vuln_module.models import (
    VulnResult,
    VulnSeverity,
    PayloadType,
    Payload,
    PayloadResult,
    ScanConfig
)

class TimeHelpers:
    """Helper functions for time-based tests."""
    
    @staticmethod
    def time() -> float:
        """Get current time in seconds."""
        return time.time()

    @staticmethod
    def sleep(seconds: float) -> None:
        """Sleep for specified seconds."""
        time.sleep(seconds)

@pytest.fixture
def time_helpers() -> TimeHelpers:
    """Provide time helper methods."""
    return TimeHelpers()

# Register time helpers globally
pytest.helpers = TimeHelpers  # type: ignore

@pytest.fixture
def basic_scan_config() -> ScanConfig:
    """Basic scan configuration for testing."""
    return ScanConfig(
        target="http://example.com",
        payload_types={PayloadType.XSS, PayloadType.SQLI},
        max_depth=2,
        threads=1,
        timeout=5
    )

@pytest.fixture
def nuclei_response_data() -> Dict[str, Any]:
    """Sample Nuclei scan response data."""
    return {
        "template-id": "test-vulnerability",
        "type": "http",
        "host": "example.com",
        "matched-at": "http://example.com/test",
        "severity": "high",
        "info": {
            "name": "Test Vulnerability",
            "description": "Test vulnerability description",
            "severity": "high",
            "reference": ["https://example.com/ref"]
        },
        "matcher-name": "test-matcher",
        "extracted-values": {},
        "request": "GET /test HTTP/1.1",
        "response": "HTTP/1.1 200 OK",
        "curl-command": "curl -X GET http://example.com/test",
        "timestamp": datetime.now().isoformat(),
        "matched-line": "<script>alert(1)</script>"
    }

@pytest.fixture
def test_payloads() -> Dict[PayloadType, List[str]]:
    """Collection of test payloads by type."""
    return {
        PayloadType.XSS: [
            "<script>alert(1)</script>",
            "javascript:alert(document.domain)"
        ],
        PayloadType.SQLI: [
            "' OR '1'='1",
            "UNION SELECT NULL--"
        ],
        PayloadType.PATH_TRAVERSAL: [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\win.ini"
        ],
        PayloadType.COMMAND_INJECTION: [
            "$(id)",
            "`whoami`"
        ]
    }

@pytest.fixture
def mock_templates_dir(tmp_path) -> str:
    """Create a temporary directory for test templates."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    return str(templates_dir)

@pytest.fixture
def mock_output_file(tmp_path) -> str:
    """Create a temporary file for test output."""
    return str(tmp_path / "nuclei_output.json")

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
