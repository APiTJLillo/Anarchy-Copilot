"""Tests for the Anarchy Copilot API."""

import pytest
from typing import Any, Dict, Generator
from fastapi import FastAPI
from fastapi.testclient import TestClient
from contextlib import contextmanager
import os
import tempfile

# Test constants
TEST_API_KEY = "test_api_key_123"
TEST_USER_ID = "test_user"

@pytest.fixture(scope="session")
def test_env() -> Dict[str, Any]:
    """Create test environment configuration."""
    return {
        "API_KEY": TEST_API_KEY,
        "USER_ID": TEST_USER_ID,
        "DEBUG": True,
        "TESTING": True,
        "ENVIRONMENT": "test",
    }

@pytest.fixture(scope="session")
def test_app(test_env: Dict[str, Any]) -> FastAPI:
    """Create FastAPI test application."""
    from api import create_app

    app = create_app(test_env)
    return app

@pytest.fixture
def test_client(test_app: FastAPI) -> TestClient:
    """Create test client with authentication."""
    with TestClient(test_app) as client:
        client.headers.update({
            "X-API-Key": TEST_API_KEY,
            "User-Agent": "AnarchyCopilot-TestClient/1.0"
        })
        yield client

@pytest.fixture(scope="session")
def test_data_dir() -> Generator[str, None, None]:
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["TEST_DATA_DIR"] = tmpdir
        yield tmpdir
        if "TEST_DATA_DIR" in os.environ:
            del os.environ["TEST_DATA_DIR"]

@contextmanager
def mock_env(**env_vars: str) -> Generator[None, None, None]:
    """Temporarily modify environment variables."""
    original = {}
    for key, value in env_vars.items():
        if key in os.environ:
            original[key] = os.environ[key]
        os.environ[key] = value

    try:
        yield
    finally:
        for key in env_vars:
            if key in original:
                os.environ[key] = original[key]
            else:
                del os.environ[key]

# Test utilities for common API operations
async def create_test_data(client: TestClient) -> Dict[str, Any]:
    """Create test data in the API."""
    # Add test data creation logic here
    return {}

async def cleanup_test_data(client: TestClient) -> None:
    """Clean up test data from the API."""
    # Add test data cleanup logic here
    pass

# Common test verification functions
def verify_response_format(response_data: Dict[str, Any]) -> None:
    """Verify common response format requirements."""
    assert isinstance(response_data, dict)
    assert "status" in response_data
    if "error" in response_data:
        assert "message" in response_data
        assert "code" in response_data

def verify_pagination(response_data: Dict[str, Any]) -> None:
    """Verify pagination response format."""
    assert "page" in response_data
    assert "per_page" in response_data
    assert "total" in response_data
    assert "items" in response_data
    assert isinstance(response_data["items"], list)

def verify_timestamp(timestamp_str: str) -> None:
    """Verify timestamp format."""
    from datetime import datetime
    try:
        datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError as e:
        pytest.fail(f"Invalid timestamp format: {e}")

# Test markers
pytest.mark.api = pytest.mark.api  # Mark tests as API tests
pytest.mark.integration = pytest.mark.integration  # Mark tests as integration tests
