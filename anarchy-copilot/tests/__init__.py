"""Test suite for Anarchy Copilot."""

from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Test constants
TEST_TIMEOUT = 30  # seconds
MAX_CONCURRENT_TESTS = 10

# Common test paths
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_TEMP_DIR = Path(__file__).parent / "temp"
TEST_TEMP_DIR.mkdir(exist_ok=True)

# Common test URLs and domains
TEST_TARGETS = {
    "http": "http://example.com",
    "https": "https://example.com",
    "domain": "example.com",
    "ip": "93.184.216.34",  # example.com IP
}

# Test categories (used in pytest marks)
TEST_CATEGORIES = {
    "unit": "Unit tests that don't require external services",
    "integration": "Tests that require external services or tools",
    "slow": "Tests that take longer than 1 second to complete",
    "network": "Tests that require network connectivity",
    "async": "Tests for asynchronous functionality",
}

# Common test utilities
def get_test_path(name: str) -> Path:
    """Get path to test data file."""
    return TEST_DATA_DIR / name

def get_temp_path(name: str) -> Path:
    """Get path for temporary test file."""
    return TEST_TEMP_DIR / name

# Cleanup on exit
import atexit
import shutil

@atexit.register
def cleanup():
    """Clean up temporary test files."""
    if TEST_TEMP_DIR.exists():
        shutil.rmtree(TEST_TEMP_DIR)
