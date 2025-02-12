"""Common test fixtures and utilities."""
import pytest
import tempfile
import asyncio
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Generator
from unittest.mock import Mock, AsyncMock, create_autospec

class TimingTestHelper:
    """Helper for testing time-based operations."""
    def __init__(self):
        """Initialize timing helper."""
        self._current_time = 0.0
        self._time_calls: List[Tuple[float, float]] = []

    def reset(self) -> None:
        """Reset time tracking."""
        self._current_time = 0.0
        self._time_calls.clear()

    def get_time(self) -> float:
        """Get current mock time."""
        return self._current_time

    def advance_time(self, seconds: float) -> None:
        """Advance mock time by given seconds."""
        self._current_time += seconds

    def record_time_call(self, elapsed: float, sleep_time: float) -> None:
        """Record a time-related function call."""
        self._time_calls.append((elapsed, sleep_time))

@pytest.fixture
def time_helper() -> TimingTestHelper:
    """Create timing test helper."""
    helper = TimingTestHelper()
    return helper

@pytest.fixture
def mock_time(time_helper: TimingTestHelper) -> Mock:
    """Create mock time object."""
    mock = Mock()
    
    def get_time() -> float:
        return time_helper.get_time()
    
    async def sleep(seconds: float) -> None:
        time_helper.advance_time(seconds)
        time_helper.record_time_call(time_helper.get_time(), seconds)
    
    mock.__call__ = Mock(side_effect=get_time)
    mock.sleep = AsyncMock(side_effect=sleep)
    return mock

@pytest.fixture
def mock_datetime(mock_time: Mock) -> Mock:
    """Create mock datetime object."""
    now_mock = Mock()
    now_mock.timestamp = Mock(side_effect=mock_time)

    datetime_mock = Mock()
    datetime_mock.now = Mock(return_value=now_mock)
    return datetime_mock

@pytest.fixture(scope="session")
def temp_test_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        shutil.rmtree(str(temp_dir))

@pytest.fixture
async def dummy_config() -> Dict[str, Any]:
    """Create dummy test configuration."""
    return {
        "testing": True,
        "debug": True,
        "environment": "test",
        "log_level": "DEBUG"
    }

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

def pytest_configure(config):
    """Configure test markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", 
        "slow: mark test as slow running"
    )
