"""
Test fixtures and configuration for the Anarchy Copilot proxy module tests.
"""
import os
import pytest
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def test_certs_dir():
    """Create a temporary directory for test certificates."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="session")
def temp_storage_dir():
    """Create a temporary directory for proxy storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(autouse=True)
def clean_test_files(request, test_certs_dir):
    """Clean up any test files after each test."""
    def cleanup():
        # Remove any test certificate files
        for file in test_certs_dir.glob("*.pem"):
            file.unlink()
        for file in test_certs_dir.glob("*.crt"):
            file.unlink()
        for file in test_certs_dir.glob("*.key"):
            file.unlink()
    
    request.addfinalizer(cleanup)

@pytest.fixture
def proxy_test_port():
    """Get a test port from environment or use default."""
    return int(os.getenv("PROXY_TEST_PORT", "8081"))

@pytest.fixture
def proxy_test_host():
    """Get test host from environment or use default."""
    return os.getenv("PROXY_TEST_HOST", "127.0.0.1")

@pytest.fixture
def test_server_url(proxy_test_host, proxy_test_port):
    """Construct the test server URL."""
    return f"http://{proxy_test_host}:{proxy_test_port}"
