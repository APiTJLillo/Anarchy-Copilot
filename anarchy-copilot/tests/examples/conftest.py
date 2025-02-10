"""Example-specific test fixtures and utilities."""

import pytest
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Generator
import tempfile
import shutil

# Example data
EXAMPLE_DATA = {
    "targets": [
        "http://example.com",
        "https://test.example.com",
        "example.org"
    ],
    "scan_configs": [
        {
            "name": "basic_scan",
            "templates": ["xss", "sqli"],
            "concurrency": 10
        },
        {
            "name": "full_scan",
            "templates": ["xss", "sqli", "path_traversal"],
            "concurrency": 20
        }
    ]
}

class ExampleTestHelper:
    """Helper class for example tests."""

    @staticmethod
    def create_example_file(content: str, temp_dir: Path, name: str = "example.txt") -> Path:
        """Create a file with example content."""
        file_path = temp_dir / name
        file_path.write_text(content)
        return file_path

    @staticmethod
    def create_example_structure(temp_dir: Path) -> Dict[str, Path]:
        """Create an example directory structure."""
        paths = {
            "config": temp_dir / "config",
            "data": temp_dir / "data",
            "output": temp_dir / "output"
        }
        
        for path in paths.values():
            path.mkdir(exist_ok=True)
            
        return paths

    @staticmethod
    def get_example_config() -> Dict[str, Any]:
        """Get example configuration."""
        return {
            "targets": EXAMPLE_DATA["targets"],
            "scan": EXAMPLE_DATA["scan_configs"][0],
            "output_dir": "example_output",
            "debug": True
        }

# Fixtures
@pytest.fixture
def example_helper() -> ExampleTestHelper:
    """Provide example test helper."""
    return ExampleTestHelper()

@pytest.fixture
def example_dir(temp_dir: Path) -> Generator[Path, None, None]:
    """Create example test directory structure."""
    example_dir = temp_dir / "example"
    example_dir.mkdir()
    
    # Create subdirectories
    (example_dir / "input").mkdir()
    (example_dir / "output").mkdir()
    (example_dir / "config").mkdir()
    
    yield example_dir
    
    # Cleanup is handled by temp_dir fixture

@pytest.fixture
def example_config() -> Dict[str, Any]:
    """Provide example configuration."""
    return ExampleTestHelper.get_example_config()

@pytest.fixture
def example_file(example_dir: Path) -> Generator[Path, None, None]:
    """Create example test file."""
    file_path = example_dir / "test.txt"
    file_path.write_text("Example test content")
    yield file_path

@pytest.fixture
def example_logger() -> logging.Logger:
    """Create logger for examples."""
    logger = logging.getLogger("anarchy_copilot.examples")
    logger.setLevel(logging.DEBUG)
    return logger

@pytest.fixture
async def example_server(example_dir: Path) -> Generator[str, None, None]:
    """Create a simple test server for examples."""
    from aiohttp import web
    
    app = web.Application()
    
    async def handle_test(request):
        return web.Response(text="Test server response")
    
    app.router.add_get("/test", handle_test)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 0)
    await site.start()
    
    # Get the assigned port
    port = site._server.sockets[0].getsockname()[1]
    url = f"http://localhost:{port}"
    
    yield url
    
    await runner.cleanup()

@pytest.fixture
def mock_target_data() -> List[Dict[str, Any]]:
    """Provide mock target data for examples."""
    return [
        {
            "url": "http://example.com",
            "ip": "93.184.216.34",
            "ports": [80, 443],
            "services": ["http", "https"]
        },
        {
            "url": "https://test.example.com",
            "ip": "93.184.216.35",
            "ports": [443],
            "services": ["https"]
        }
    ]

@pytest.fixture
def mock_scan_results() -> List[Dict[str, Any]]:
    """Provide mock scan results for examples."""
    return [
        {
            "id": "test-1",
            "target": "http://example.com",
            "findings": [
                {
                    "type": "xss",
                    "severity": "high",
                    "description": "Example XSS finding"
                }
            ]
        },
        {
            "id": "test-2",
            "target": "https://test.example.com",
            "findings": [
                {
                    "type": "sqli",
                    "severity": "critical",
                    "description": "Example SQLi finding"
                }
            ]
        }
    ]

# Custom markers for examples
pytest.mark.example = pytest.mark.example
pytest.mark.example_integration = pytest.mark.example_integration
pytest.mark.example_slow = pytest.mark.example_slow
