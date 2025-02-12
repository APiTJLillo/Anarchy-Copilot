"""Test fixtures for API tests."""
import os
import pytest
import asyncio
from typing import Generator
from unittest.mock import AsyncMock, Mock
from fastapi.testclient import TestClient

from api import app
from proxy.core import ProxyServer

@pytest.fixture
def test_client() -> Generator:
    """Create a test client for the API."""
    with TestClient(app) as client:
        yield client

@pytest.fixture(autouse=True)
def test_env(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("NUCLEI_PATH", "/usr/local/bin/nuclei")
    monkeypatch.setenv("TEST_MODE", "true")

@pytest.fixture
def test_port(unused_tcp_port):
    """Get an unused TCP port for testing."""
    return unused_tcp_port

@pytest.fixture
def mock_proxy_server(test_port):
    """Create mock proxy server."""
    mock = AsyncMock(spec=ProxyServer)
    mock._is_running = True
    
    # Configure config
    mock.config = Mock()
    mock.config.port = test_port
    mock.config.intercept_requests = True
    mock.config.intercept_responses = True
    mock.config.allowed_hosts = set()
    mock.config.excluded_hosts = set()
    
    # Configure session
    mock.session = Mock()
    mock.session.get_history = Mock(return_value=[])
    mock.session.find_entry = Mock(return_value=None)
    mock.session.add_entry_tag = Mock(return_value=True)
    mock.session.set_entry_note = Mock(return_value=True)
    mock.session.clear_history = Mock()
    
    # Configure async methods
    mock.start = AsyncMock()
    mock.stop = AsyncMock()
    
    # Patch the property
    type(mock).is_running = property(lambda _: True)
    return mock

@pytest.fixture(autouse=True)
def cleanup_proxy():
    """Clean up proxy server after each test."""
    yield
    # Clean up after test
    import api.proxy
    if api.proxy.proxy_server and api.proxy.proxy_server.is_running:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(api.proxy.proxy_server.stop())
    api.proxy.proxy_server = None

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()
