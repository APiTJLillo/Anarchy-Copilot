"""Global test configuration and fixtures."""

import os
import pytest
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator, Optional
import asyncio
import signal
from contextlib import contextmanager
import time

# Setup test logging
def pytest_configure(config):
    """Configure test environment."""
    logging.basicConfig(
        level=logging.DEBUG if config.option.verbose > 0 else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Register custom markers
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "api: mark test as API test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "network: mark test as requiring network access")

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the test session."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_dir() -> Generator[Path, None, None]:
    """Create base temporary directory for all tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        yield path

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Get test data directory."""
    return project_root / "tests" / "data"

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture(scope="session")
def nuclei_templates_dir(test_data_dir: Path) -> Path:
    """Get Nuclei templates directory."""
    return test_data_dir / "nuclei_templates"

@pytest.fixture(scope="session")
def mock_env() -> Dict[str, str]:
    """Mock environment variables."""
    return {
        "ANARCHY_COPILOT_DEBUG": "true",
        "ANARCHY_COPILOT_TESTING": "true",
        "ANARCHY_COPILOT_ENV": "test"
    }

@pytest.fixture(autouse=True)
def setup_test_env(mock_env: Dict[str, str]) -> Generator[None, None, None]:
    """Set up test environment variables."""
    original = {}
    for key, value in mock_env.items():
        if key in os.environ:
            original[key] = os.environ[key]
        os.environ[key] = value

    yield

    for key in mock_env:
        if key in original:
            os.environ[key] = original[key]
        else:
            del os.environ[key]

@contextmanager
def timeout(seconds: int):
    """Context manager for timing out long operations."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Register timeout handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore original handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

@pytest.fixture
def strict_timeout():
    """Fixture to enforce strict timeouts in tests."""
    return timeout

class TimingTestHelper:
    """Helper for timing-sensitive tests."""

    @staticmethod
    def wait_for(condition: callable, timeout: float = 5.0, interval: float = 0.1) -> bool:
        """Wait for a condition to be true."""
        end_time = time.time() + timeout
        while time.time() < end_time:
            if condition():
                return True
            time.sleep(interval)
        return False

    @staticmethod
    @contextmanager
    def timed_execution(expected_duration: float, tolerance: float = 0.1):
        """Verify execution time of a block."""
        start_time = time.time()
        yield
        duration = time.time() - start_time
        assert abs(duration - expected_duration) <= tolerance, \
            f"Expected duration {expected_duration}s but took {duration}s"

@pytest.fixture
def timing_helper() -> TimingTestHelper:
    """Provide timing test helper."""
    return TimingTestHelper()

def pytest_runtest_setup(item):
    """Perform test setup based on markers."""
    # Skip network tests if offline mode is enabled
    if item.get_closest_marker("network") and os.environ.get("OFFLINE_TESTS"):
        pytest.skip("Network tests disabled in offline mode")

    # Skip slow tests unless explicitly enabled
    if item.get_closest_marker("slow") and not item.config.getoption("--run-slow"):
        pytest.skip("Slow tests skipped. Use --run-slow to run.")

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--offline",
        action="store_true",
        default=False,
        help="Run in offline mode (skip network tests)"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on configuration."""
    if config.getoption("--offline"):
        os.environ["OFFLINE_TESTS"] = "1"
        skip_network = pytest.mark.skip(reason="Network tests disabled in offline mode")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)
