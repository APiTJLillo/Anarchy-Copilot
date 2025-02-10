"""Example test demonstrating test utilities and fixtures usage."""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any
import logging

# Get module logger
logger = logging.getLogger(__name__)

#
# Basic Test Examples
#

def test_basic_fixtures(temp_dir: Path, test_data_dir: Path):
    """Demonstrate basic fixture usage."""
    # temp_dir is a temporary directory that is cleaned up after the test
    assert temp_dir.exists()
    assert isinstance(temp_dir, Path)

    # test_data_dir points to the tests/data directory
    assert test_data_dir.exists()
    assert (test_data_dir / "nuclei_templates").exists()

def test_environment_setup(mock_env: Dict[str, str]):
    """Demonstrate environment variable handling."""
    import os
    # Environment variables are automatically set up
    assert os.environ["ANARCHY_COPILOT_TESTING"] == "true"
    assert os.environ["ANARCHY_COPILOT_DEBUG"] == "true"

#
# Async Test Examples
#

@pytest.mark.asyncio
async def test_async_operation():
    """Demonstrate async test functionality."""
    # Simulated async operation
    await asyncio.sleep(0.1)
    assert True

@pytest.mark.asyncio
async def test_timing_utilities(timing_helper):
    """Demonstrate timing utilities."""
    # Test timed execution
    with timing_helper.timed_execution(expected_duration=0.1, tolerance=0.05):
        await asyncio.sleep(0.1)

    # Test wait_for condition
    flag = False
    async def set_flag():
        nonlocal flag
        await asyncio.sleep(0.1)
        flag = True

    asyncio.create_task(set_flag())
    assert timing_helper.wait_for(lambda: flag, timeout=1.0)

#
# Test Categories Examples
#

@pytest.mark.unit
def test_unit_example():
    """Demonstrate unit test marking."""
    assert True

@pytest.mark.integration
def test_integration_example(nuclei_templates_dir: Path):
    """Demonstrate integration test marking and template usage."""
    # Verify test templates are available
    assert nuclei_templates_dir.exists()
    assert (nuclei_templates_dir / "test_xss.yaml").exists()

@pytest.mark.slow
def test_slow_operation(timing_helper):
    """Demonstrate slow test marking and timing."""
    with timing_helper.timed_execution(expected_duration=1.0, tolerance=0.1):
        import time
        time.sleep(1.0)

@pytest.mark.network
def test_network_operation():
    """Demonstrate network test marking."""
    # This test will be skipped in offline mode
    import socket
    socket.create_connection(("8.8.8.8", 53), timeout=1)

#
# Timeout Handling Example
#

def test_timeout_handling(strict_timeout):
    """Demonstrate timeout handling."""
    # This operation should complete within 1 second
    with strict_timeout(1):
        import time
        time.sleep(0.1)

    # This would timeout
    with pytest.raises(TimeoutError):
        with strict_timeout(1):
            time.sleep(2)

#
# Common Test Patterns
#

class TestExamplePatterns:
    """Demonstrate common test patterns."""

    @pytest.fixture(autouse=True)
    def setup(self, temp_dir: Path):
        """Set up test resources."""
        self.temp_dir = temp_dir
        self.test_file = temp_dir / "test.txt"
        
        # Create test file
        self.test_file.write_text("test content")
        
        yield
        
        # Cleanup is handled automatically by temp_dir fixture

    def test_file_operations(self):
        """Test file operations pattern."""
        assert self.test_file.exists()
        assert self.test_file.read_text() == "test content"

    @pytest.mark.parametrize("content,expected", [
        ("test1", 5),
        ("test22", 6),
        ("", 0)
    ])
    def test_parameterized_example(self, content: str, expected: int):
        """Demonstrate parameterized testing."""
        self.test_file.write_text(content)
        assert len(self.test_file.read_text()) == expected

    def test_error_handling(self):
        """Demonstrate error handling testing."""
        # Test expected exceptions
        with pytest.raises(FileNotFoundError):
            (self.temp_dir / "nonexistent.txt").read_text()

        # Test error messages
        try:
            (self.temp_dir / "nonexistent.txt").read_text()
        except FileNotFoundError as e:
            assert "No such file" in str(e)

#
# Logging Example
#

def test_logging(caplog):
    """Demonstrate logging capture and verification."""
    with caplog.at_level(logging.DEBUG):
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")

        assert "Debug message" in caplog.text
        assert "Info message" in caplog.text
        assert "Warning message" in caplog.text

        # Check specific log records
        assert any(
            record.levelname == "WARNING" and "Warning message" in record.message
            for record in caplog.records
        )
