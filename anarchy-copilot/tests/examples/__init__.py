"""Example test package for demonstrating testing patterns and utilities."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

# Configure example test logging
logger = logging.getLogger("anarchy_copilot.tests.examples")

# Example test constants
EXAMPLE_TIMEOUT = 30  # seconds
EXAMPLE_CONCURRENCY = 5
EXAMPLE_TARGETS = [
    "http://example.com",
    "https://test.example.com",
    "example.org"
]

@dataclass
class ExampleScanConfig:
    """Example scan configuration."""
    name: str
    templates: List[str]
    concurrency: int = EXAMPLE_CONCURRENCY
    timeout: int = EXAMPLE_TIMEOUT
    options: Optional[Dict[str, Any]] = None

# Example configurations
EXAMPLE_CONFIGS = {
    "basic": ExampleScanConfig(
        name="basic_scan",
        templates=["xss", "sqli"],
        concurrency=5
    ),
    "full": ExampleScanConfig(
        name="full_scan",
        templates=["xss", "sqli", "path_traversal"],
        concurrency=10
    ),
    "safe": ExampleScanConfig(
        name="safe_scan",
        templates=["xss"],
        concurrency=2,
        options={"safe": True}
    )
}

class ExampleTestError(Exception):
    """Base exception for example tests."""
    pass

class ExampleTestTimeout(ExampleTestError):
    """Raised when an example test times out."""
    pass

class ExampleTestSetupError(ExampleTestError):
    """Raised when example test setup fails."""
    pass

def get_example_assets_dir() -> Path:
    """Get the directory containing example test assets."""
    return Path(__file__).parent / "assets"

def get_example_config(name: str) -> ExampleScanConfig:
    """Get a predefined example configuration."""
    if name not in EXAMPLE_CONFIGS:
        raise ValueError(f"Unknown example config: {name}")
    return EXAMPLE_CONFIGS[name]

def create_example_report(
    results: List[Dict[str, Any]],
    output_dir: Path,
    name: str = "example_report"
) -> Path:
    """Create an example test report."""
    import json
    from datetime import datetime

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "name": name,
        "results": results,
        "statistics": {
            "total": len(results),
            "success": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False))
        }
    }

    output_file = output_dir / f"{name}.json"
    output_file.write_text(json.dumps(report, indent=2))
    return output_file

# Example test decorators
def example_test(name: str):
    """Decorator to mark a function as an example test."""
    import pytest
    return pytest.mark.example(name=name)

def slow_example():
    """Decorator to mark an example test as slow."""
    import pytest
    return pytest.mark.example_slow

def integration_example():
    """Decorator to mark an example test as requiring integration."""
    import pytest
    return pytest.mark.example_integration

__all__ = [
    'ExampleScanConfig',
    'ExampleTestError',
    'ExampleTestTimeout',
    'ExampleTestSetupError',
    'EXAMPLE_CONFIGS',
    'EXAMPLE_TARGETS',
    'EXAMPLE_TIMEOUT',
    'EXAMPLE_CONCURRENCY',
    'get_example_assets_dir',
    'get_example_config',
    'create_example_report',
    'example_test',
    'slow_example',
    'integration_example'
]
