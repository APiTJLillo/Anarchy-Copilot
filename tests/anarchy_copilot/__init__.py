"""Tests for the Anarchy Copilot package."""

from typing import Dict, Any, Optional
import pytest
import os
import tempfile
from pathlib import Path
import logging
from typing import Generator

# Configure test logging
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("TEST_DEBUG") else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("anarchy_copilot.tests")

# Test paths
TEST_ROOT = Path(__file__).parent
TEST_DATA_DIR = TEST_ROOT / "data"
TEMPLATE_DIR = TEST_DATA_DIR / "nuclei_templates"

# Test constants
TEST_PROJECT_ID = 1
TEST_API_KEY = "test_api_key_123"
TEST_USER_ID = "test_user"

class TestHelper:
    """Shared test utilities."""

    @staticmethod
    def create_temp_dir() -> Path:
        """Create a temporary directory for tests."""
        return Path(tempfile.mkdtemp(prefix="anarchy_copilot_test_"))

    @staticmethod
    def create_temp_file(content: str = "") -> Path:
        """Create a temporary file with content."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(content)
            return Path(f.name)

    @staticmethod
    def get_test_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get test configuration."""
        config = {
            "DEBUG": True,
            "TESTING": True,
            "API_KEY": TEST_API_KEY,
            "USER_ID": TEST_USER_ID,
            "PROJECT_ID": TEST_PROJECT_ID,
            "ENVIRONMENT": "test",
        }
        if overrides:
            config.update(overrides)
        return config

    @staticmethod
    def setup_test_env(config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Set up test environment variables."""
        test_env = {
            "ANARCHY_COPILOT_DEBUG": "true",
            "ANARCHY_COPILOT_TESTING": "true",
            "ANARCHY_COPILOT_API_KEY": TEST_API_KEY,
            "ANARCHY_COPILOT_USER_ID": TEST_USER_ID,
            "ANARCHY_COPILOT_PROJECT_ID": str(TEST_PROJECT_ID),
            "ANARCHY_COPILOT_ENV": "test"
        }
        if config:
            test_env.update({
                f"ANARCHY_COPILOT_{k.upper()}": str(v)
                for k, v in config.items()
            })
        for k, v in test_env.items():
            os.environ[k] = v
        return test_env

    @staticmethod
    def cleanup_test_env(env_vars: Dict[str, str]) -> None:
        """Clean up test environment variables."""
        for k in env_vars:
            if k in os.environ:
                del os.environ[k]

# Common test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.api = pytest.mark.api
pytest.mark.slow = pytest.mark.slow

# Common test fixtures
@pytest.fixture(scope="session")
def test_helper() -> TestHelper:
    """Provide test helper instance."""
    return TestHelper()

@pytest.fixture(scope="session")
def test_env(test_helper: TestHelper) -> Generator[Dict[str, str], None, None]:
    """Set up test environment."""
    env_vars = test_helper.setup_test_env()
    yield env_vars
    test_helper.cleanup_test_env(env_vars)

@pytest.fixture(scope="session")
def test_config(test_helper: TestHelper) -> Dict[str, Any]:
    """Get test configuration."""
    return test_helper.get_test_config()

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get test data directory."""
    return TEST_DATA_DIR

@pytest.fixture(scope="session")
def template_dir() -> Path:
    """Get Nuclei templates directory."""
    return TEMPLATE_DIR

@pytest.fixture
def temp_dir(test_helper: TestHelper) -> Generator[Path, None, None]:
    """Create temporary directory for test."""
    temp_dir = test_helper.create_temp_dir()
    yield temp_dir
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
