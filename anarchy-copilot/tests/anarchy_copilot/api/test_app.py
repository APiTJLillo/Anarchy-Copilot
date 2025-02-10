"""Tests for FastAPI application factory."""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from typing import Dict, Any, Generator

from anarchy_copilot.api import create_app, get_app

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Create test configuration."""
    return {
        "DEBUG": True,
        "TESTING": True,
        "API_KEY": "test_key",
        "ENVIRONMENT": "test"
    }

@pytest.fixture
def app(test_config: Dict[str, Any]) -> FastAPI:
    """Create test FastAPI application."""
    return create_app(test_config)

@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    """Create test client."""
    with TestClient(app) as client:
        yield client

def test_app_creation(app: FastAPI):
    """Test basic application creation."""
    assert app.title == "Anarchy Copilot"
    assert app.version == "0.1.0"
    assert app.docs_url == "/docs"
    assert app.redoc_url == "/redoc"

def test_config_loading(app: FastAPI, test_config: Dict[str, Any]):
    """Test configuration loading."""
    assert app.state.config["DEBUG"] is True
    assert app.state.config["TESTING"] is True
    assert app.state.config["API_KEY"] == "test_key"
    assert app.state.config["ENVIRONMENT"] == "test"

def test_middleware_registration(app: FastAPI):
    """Test middleware registration."""
    middleware_classes = [m.__class__.__name__ for m in app.middleware]
    assert "CORSMiddleware" in middleware_classes
    assert "GZipMiddleware" in middleware_classes

def test_cors_configuration(client: TestClient):
    """Test CORS configuration."""
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type"
        }
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers
    assert "access-control-allow-headers" in response.headers

def test_auth_middleware_disabled_in_test_mode(client: TestClient):
    """Test authentication middleware is disabled in test mode."""
    response = client.get("/health")
    assert response.status_code != 401

def test_auth_middleware_enabled_in_production():
    """Test authentication middleware is enabled in production."""
    app = create_app({"TESTING": False, "API_KEY": "prod_key"})
    client = TestClient(app)

    # Request without API key should fail
    response = client.get("/health")
    assert response.status_code == 401
    assert response.json()["code"] == "invalid_api_key"

    # Request with invalid API key should fail
    response = client.get("/health", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401

    # Request with correct API key should succeed
    response = client.get("/health", headers={"X-API-Key": "prod_key"})
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_startup_event(app: FastAPI):
    """Test startup event handler."""
    # Trigger startup event
    await app.router.startup()
    # Add assertions for startup tasks

@pytest.mark.asyncio
async def test_shutdown_event(app: FastAPI):
    """Test shutdown event handler."""
    # Trigger shutdown event
    await app.router.shutdown()
    # Add assertions for cleanup tasks

def test_error_handler(client: TestClient):
    """Test global error handler."""
    # Create a route that raises an exception
    @client.app.get("/test-error")
    async def error_route():
        raise ValueError("Test error")

    response = client.get("/test-error")
    assert response.status_code == 500
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Internal server error"
    assert data["code"] == "internal_error"
    # In test mode, error details should be included
    assert "Test error" in data["details"]

def test_get_app():
    """Test get_app function."""
    app = get_app()
    assert isinstance(app, FastAPI)
    assert app.title == "Anarchy Copilot"

def test_debug_logging(caplog):
    """Test debug logging middleware."""
    app = create_app({"DEBUG": True})
    client = TestClient(app)

    with caplog.at_level("DEBUG"):
        response = client.get("/health")
        assert any("Request: GET" in record.message for record in caplog.records)
        assert any(f"Response: {response.status_code}" in record.message
                for record in caplog.records)

@pytest.mark.parametrize("config,expected_docs", [
    ({"DOCS_ENABLED": True}, True),
    ({"DOCS_ENABLED": False}, False),
    ({}, True)  # Default case
])
def test_docs_configuration(config: Dict[str, Any], expected_docs: bool):
    """Test API documentation configuration."""
    app = create_app(config)
    assert bool(app.docs_url) == expected_docs
    assert bool(app.redoc_url) == expected_docs
