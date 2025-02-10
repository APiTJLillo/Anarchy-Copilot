from fastapi.testclient import TestClient
from .main import app, get_db
from .models import Base, engine, SessionLocal
import pytest

# Create a new database for testing
Base.metadata.create_all(bind=engine)

# Dependency override for testing
def override_get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(scope="module")
def test_db():
    # Setup: Create a new database for testing
    Base.metadata.create_all(bind=engine)
    yield
    # Teardown: Drop the database tables after tests
    Base.metadata.drop_all(bind=engine)

def test_create_project(test_db):
    response = client.post(
        "/projects/",
        json={"name": "Test Project", "scope_details": "Test Scope"}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Test Project"

def test_read_project(test_db):
    response = client.get("/projects/1")
    assert response.status_code == 200
    assert response.json()["name"] == "Test Project"
