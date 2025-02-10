"""Common test fixtures and utilities for API tests."""

import pytest
import asyncio
from typing import Dict, Any, AsyncGenerator, Generator
from fastapi import FastAPI
from fastapi.testclient import TestClient
import logging
from pathlib import Path
import tempfile
import os

from anarchy_copilot.api import create_app

# Test constants
TEST_API_KEY = "test_api_key_123"
TEST_USER_ID = "test_user"
TEST_PROJECT_ID = 1

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_env() -> Dict[str, Any]:
    """Create test environment configuration."""
    return {
        "API_KEY": TEST_API_KEY,
        "USER_ID": TEST_USER_ID,
        "PROJECT_ID": TEST_PROJECT_ID,
        "DEBUG": True,
        "TESTING": True,
        "ENVIRONMENT": "test",
        "DOCS_ENABLED": True
    }

@pytest.fixture(scope="session")
def app(test_env: Dict[str, Any]) -> FastAPI:
    """Create FastAPI test application."""
    return create_app(test_env)

@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    """Create test client with authentication."""
    with TestClient(app) as client:
        client.headers.update({
            "X-API-Key": TEST_API_KEY,
            "User-Agent": "AnarchyCopilot-TestClient/1.0"
        })
        yield client

@pytest.fixture(scope="session")
def test_data_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        os.environ["TEST_DATA_DIR"] = str(path)
        yield path
        if "TEST_DATA_DIR" in os.environ:
            del os.environ["TEST_DATA_DIR"]

@pytest.fixture
def mock_nuclei_response() -> Dict[str, Any]:
    """Create mock Nuclei scanner response."""
    return {
        "template-id": "test-vulnerability",
        "info": {
            "name": "Test Vulnerability",
            "author": ["test"],
            "severity": "high",
            "description": "Test vulnerability detection",
            "reference": ["https://example.com"]
        },
        "host": "example.com",
        "matched-at": "http://example.com/test",
        "type": "http",
        "timestamp": "2025-02-10T12:00:00.000Z",
        "matcher-name": "test-matcher",
        "extracted-values": {},
        "curl-command": "curl -X GET http://example.com/test",
        "ip": "93.184.216.34"
    }

@pytest.fixture
def mock_scan_result() -> Dict[str, Any]:
    """Create mock scan result data."""
    return {
        "scan_id": "test-scan-123",
        "target": "http://example.com",
        "timestamp": "2025-02-10T12:00:00.000Z",
        "status": "completed",
        "findings": [
            {
                "type": "vulnerability",
                "name": "Test Vulnerability",
                "severity": "high",
                "description": "Test vulnerability found",
                "evidence": "<script>alert(1)</script>",
                "location": "http://example.com/test",
                "confidence": "high"
            }
        ],
        "statistics": {
            "requests": 100,
            "duration": 60,
            "findings_count": 1
        }
    }

@pytest.fixture
def caplog_debug(caplog):
    """Enable debug logging for tests."""
    caplog.set_level(logging.DEBUG)
    return caplog

# Utility functions
def verify_json_response(data: Dict[str, Any]) -> None:
    """Verify common JSON response format."""
    assert isinstance(data, dict)
    assert "status" in data
    if "error" in data:
        assert "message" in data
        assert "code" in data

def verify_error_response(data: Dict[str, Any], expected_code: str) -> None:
    """Verify error response format."""
    verify_json_response(data)
    assert data["status"] == "error"
    assert data["code"] == expected_code
    assert "message" in data

def verify_success_response(data: Dict[str, Any]) -> None:
    """Verify success response format."""
    verify_json_response(data)
    assert data["status"] == "success"

def verify_pagination_response(data: Dict[str, Any]) -> None:
    """Verify paginated response format."""
    verify_success_response(data)
    assert "items" in data
    assert isinstance(data["items"], list)
    assert "total" in data
    assert "page" in data
    assert "per_page" in data

# Register markers
def pytest_configure(config):
    """Configure test markers."""
    config.addinivalue_line(
        "markers",
        "api: mark test as API test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
