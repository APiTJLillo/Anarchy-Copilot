"""Test fixtures for vulnerability scanning module."""
import pytest
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@pytest.fixture
def mock_time():
    """Provide access to controlled time values for testing."""
    @dataclass
    class TimeHelper:
        _current_time: float = 0.0

        def now(self) -> datetime:
            return datetime.fromtimestamp(self._current_time)

        def timestamp(self) -> float:
            return self._current_time

        def advance(self, seconds: float) -> None:
            self._current_time += seconds

        def set(self, timestamp: float) -> None:
            self._current_time = timestamp

    return TimeHelper()

@pytest.fixture
def mock_rate_limiter(mock_time):
    """Mock rate limiter with time control."""
    from vuln_module.rate_limiter import RateLimiter

    class MockRateLimiter(RateLimiter):
        def __init__(self, rate: Optional[int] = None):
            # Convert None to 0 for base class
            super().__init__(0 if rate is None else rate)
            self._time = mock_time
            self._rate = rate

        def _now(self) -> float:
            return self._time.timestamp()

        async def acquire(self) -> None:
            if self._rate is None or self._rate <= 0:
                return
            await super().acquire()

    return MockRateLimiter

@pytest.fixture
def mock_scan_results():
    """Mock scan results for testing."""
    return [{
        "template-id": "test-xss",
        "type": "http",
        "severity": "high",
        "info": {"description": "Test XSS"},
        "matched-at": "http://example.com/test",
        "matcher-name": "<script>alert(1)</script>",
        "extracted-values": {},
        "ip": "1.2.3.4",
        "host": "example.com",
        "request": "GET /test",
        "response": "<html>test</html>"
    }]

@pytest.fixture
def mock_target_data():
    """Mock target data for testing."""
    return {
        "url": "http://example.com",
        "headers": {"User-Agent": "test"},
        "cookies": {"session": "123"},
        "depth": 3
    }

class AsyncIteratorWrapper:
    """Wrapper for async iterators that implements __aiter__."""
    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)

@pytest.fixture
def async_gen():
    """Create async iterators for testing."""
    def create_iterator(items):
        return AsyncIteratorWrapper(items.copy())
    return create_iterator

@pytest.fixture
def test_templates_dir(tmp_path):
    """Create temporary templates directory and parent dirs."""
    templates_dir = tmp_path / "test" / "templates"
    templates_dir.parent.mkdir(exist_ok=True)
    templates_dir.mkdir(exist_ok=True)
    return templates_dir

@pytest.fixture
def test_output_file(tmp_path):
    """Create temporary output file."""
    output_file = tmp_path / "output.json"
    output_file.parent.mkdir(exist_ok=True)
    output_file.touch()
    return output_file
