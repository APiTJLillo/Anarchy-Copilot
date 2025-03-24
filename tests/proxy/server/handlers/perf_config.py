"""Configuration settings for performance tests."""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
import logging
from pydantic import BaseSettings, Field

logger = logging.getLogger(__name__)

class PerformanceSettings(BaseSettings):
    """Performance test settings."""
    # Test execution settings
    concurrent_connections: int = Field(
        default=100,
        description="Number of concurrent connections to test"
    )
    request_count: int = Field(
        default=1000,
        description="Number of requests per test iteration"
    )
    warmup_requests: int = Field(
        default=100,
        description="Number of warmup requests before testing"
    )
    cooldown_time: int = Field(
        default=5,
        description="Cooldown time in seconds between tests"
    )

    # Performance thresholds
    min_throughput: float = Field(
        default=1000.0,
        description="Minimum acceptable requests per second"
    )
    max_latency_p95: float = Field(
        default=0.01,
        description="Maximum acceptable P95 latency in seconds"
    )
    max_memory_mb: float = Field(
        default=100.0,
        description="Maximum acceptable memory usage in MB"
    )
    max_cpu_percent: float = Field(
        default=80.0,
        description="Maximum acceptable CPU usage percentage"
    )

    # Regression thresholds
    throughput_regression_threshold: float = Field(
        default=0.15,
        description="Maximum acceptable throughput degradation (15%)"
    )
    latency_regression_threshold: float = Field(
        default=0.15,
        description="Maximum acceptable latency increase (15%)"
    )
    memory_regression_threshold: float = Field(
        default=0.20,
        description="Maximum acceptable memory usage increase (20%)"
    )

    # Test data settings
    data_sizes: List[int] = Field(
        default=[1024, 64*1024, 1024*1024],
        description="List of data sizes to test (in bytes)"
    )
    data_distributions: Dict[str, float] = Field(
        default={
            "small": 0.7,   # 70% small requests
            "medium": 0.2,  # 20% medium requests
            "large": 0.1    # 10% large requests
        },
        description="Distribution of request sizes"
    )

    # Output settings
    output_dir: Path = Field(
        default=Path("performance_results"),
        description="Directory for test results"
    )
    baseline_dir: Path = Field(
        default=Path(".performance_baselines"),
        description="Directory for baseline data"
    )
    report_format: str = Field(
        default="html",
        description="Output format for reports (html, json, or both)"
    )

    # System settings
    process_priority: Optional[int] = Field(
        default=None,
        description="Process priority for tests (nice value)"
    )
    thread_count: Optional[int] = Field(
        default=None,
        description="Number of threads to use (None for auto)"
    )
    disable_gc: bool = Field(
        default=False,
        description="Disable garbage collection during tests"
    )

    # Visualization settings
    plot_style: str = Field(
        default="seaborn-darkgrid",
        description="Matplotlib style for plots"
    )
    chart_size: tuple = Field(
        default=(12, 6),
        description="Default size for charts (width, height)"
    )
    save_raw_data: bool = Field(
        default=True,
        description="Save raw test data for later analysis"
    )

    class Config:
        """Pydantic config."""
        env_prefix = "PERF_"
        case_sensitive = False

    @classmethod
    def load_from_file(cls, path: Path) -> "PerformanceSettings":
        """Load settings from YAML file."""
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
                return cls(**data)
        except Exception as e:
            logger.error(f"Error loading performance settings: {e}")
            return cls()

    def save_to_file(self, path: Path):
        """Save settings to YAML file."""
        data = self.dict(exclude={"output_dir", "baseline_dir"})
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def get_test_profile(self, name: str) -> Dict[str, Any]:
        """Get predefined test profile settings."""
        profiles = {
            "quick": {
                "concurrent_connections": 10,
                "request_count": 100,
                "warmup_requests": 10,
                "data_sizes": [1024]
            },
            "standard": {
                "concurrent_connections": self.concurrent_connections,
                "request_count": self.request_count,
                "warmup_requests": self.warmup_requests,
                "data_sizes": self.data_sizes
            },
            "stress": {
                "concurrent_connections": self.concurrent_connections * 2,
                "request_count": self.request_count * 2,
                "warmup_requests": self.warmup_requests * 2,
                "data_sizes": self.data_sizes + [4 * 1024 * 1024]
            }
        }
        return profiles.get(name, profiles["standard"])

# Default configuration
default_config = PerformanceSettings()

def load_config(path: Optional[Path] = None) -> PerformanceSettings:
    """Load configuration from file or use defaults."""
    if path and path.exists():
        return PerformanceSettings.load_from_file(path)
    return default_config

# Example performance test profiles
PERFORMANCE_PROFILES = {
    "ci": {
        "description": "Quick tests for CI pipeline",
        "settings": {
            "concurrent_connections": 50,
            "request_count": 500,
            "warmup_requests": 50
        }
    },
    "baseline": {
        "description": "Standard baseline tests",
        "settings": {
            "concurrent_connections": 100,
            "request_count": 1000,
            "warmup_requests": 100
        }
    },
    "stress": {
        "description": "Heavy load stress testing",
        "settings": {
            "concurrent_connections": 200,
            "request_count": 2000,
            "warmup_requests": 200,
            "data_sizes": [1024, 64*1024, 1024*1024, 4*1024*1024]
        }
    }
}

if __name__ == "__main__":
    # Example usage
    config = PerformanceSettings()
    print(f"Default settings:")
    print(f"Concurrent connections: {config.concurrent_connections}")
    print(f"Request count: {config.request_count}")
    print(f"Throughput threshold: {config.min_throughput} req/s")
    
    # Save example config
    config.save_to_file(Path("example_perf_config.yml"))
