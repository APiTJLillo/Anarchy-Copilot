"""Test utilities for the API tests."""
from typing import AsyncGenerator, Generator
import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient

from api import create_app

@pytest.fixture
def test_app() -> FastAPI:
    """Create a test FastAPI application."""
    return create_app()

@pytest.fixture
def test_client(test_app: FastAPI) -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(test_app)

@pytest.fixture
async def async_test_client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI application."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
