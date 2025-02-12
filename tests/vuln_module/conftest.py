"""Test fixtures for vulnerability module."""
import pytest
import shutil
import tempfile
import time
from pathlib import Path

@pytest.fixture
def mock_time(monkeypatch):
    """Mock time functions for rate limiting tests."""
    class MockTime:
        def __init__(self):
            self.current_time = 0

        def sleep(self, seconds):
            self.current_time += seconds
            return self.current_time

        def time(self):
            return self.current_time

    mock = MockTime()
    
    async def mock_async_sleep(seconds):
        mock.sleep(seconds)

    monkeypatch.setattr("time.time", mock.time)
    monkeypatch.setattr("time.sleep", mock.sleep)
    monkeypatch.setattr("asyncio.sleep", mock_async_sleep)
    return mock

@pytest.fixture
def time_helper(mock_time):
    """Helper functions for time-based tests."""
    class TimeHelper:
        @staticmethod
        def get_time():
            return mock_time.time()

    return TimeHelper()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(str(temp_path), ignore_errors=True)

@pytest.fixture(autouse=True)
def mock_nuclei_installation(monkeypatch):
    """Mock nuclei binary presence."""
    monkeypatch.setenv("PATH", "/mock/path")
    monkeypatch.setattr("shutil.which", lambda x: "/mock/path/nuclei" if x == "nuclei" else None)

@pytest.fixture(autouse=True)
def mock_temp_directory(temp_dir, monkeypatch):
    """Mock temporary directory creation to use test directory."""
    def mock_mkdtemp(*args, **kwargs):
        test_dir = temp_dir / "nuclei_test"
        test_dir.mkdir(exist_ok=True)
        return str(test_dir)

    monkeypatch.setattr("tempfile.mkdtemp", mock_mkdtemp)
    monkeypatch.setattr("shutil.rmtree", lambda x, **kwargs: None)
