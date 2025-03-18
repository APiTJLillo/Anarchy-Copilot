"""Proxy server configuration."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

@dataclass
class ProxyConfig:
    """Proxy server configuration."""
    host: str = "127.0.0.1"
    port: int = 8083
    
    # Connection settings
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_timeout: int = 30
    history_size: int = 1000
    buffer_size: int = 262144  # 256KB default buffer size
    
    # Tunnel settings
    metrics_interval: float = 0.1  # Interval for metrics collection
    write_limit: int = 1048576  # 1MB write limit
    write_interval: float = 0.0001  # Write interval for rate limiting
    
    # Feature flags
    websocket_support: bool = True
    http2_support: bool = False
    intercept_responses: bool = True
    intercept_requests: bool = True

    # SSL/TLS settings
    ca_cert_path: Optional[Path] = None
    ca_key_path: Optional[Path] = None

    # Filtering
    allowed_hosts: List[str] = None
    excluded_hosts: List[str] = None

    # Memory monitoring settings
    memory_sample_interval: float = 10.0  # Seconds between memory samples
    memory_growth_threshold: int = 10 * 1024 * 1024  # 10MB threshold for growth alerts
    memory_sample_retention: int = 3600  # Keep 1 hour of memory samples
    memory_log_level: str = "INFO"  # Logging level for memory metrics
    memory_alert_level: str = "WARNING"  # Logging level for memory alerts
    
    # Leak detection settings
    leak_detection_threshold: float = 0.8  # 80% confidence threshold for leak detection
    leak_detection_samples: int = 10  # Minimum samples needed for leak detection
    leak_growth_rate: float = 0.1  # 10% growth rate threshold for leak warning
    
    # Cleanup thresholds
    cleanup_timeout: float = 5.0  # Seconds to wait for cleanup operations
    force_cleanup_threshold: int = 100 * 1024 * 1024  # 100MB threshold for forced cleanup
    cleanup_retry_delay: float = 0.5  # Delay between cleanup retries

    def __post_init__(self):
        """Initialize default lists and validate paths."""
        if self.allowed_hosts is None:
            self.allowed_hosts = []
        if self.excluded_hosts is None:
            self.excluded_hosts = []

        # Convert string paths to Path objects
        if isinstance(self.ca_cert_path, str):
            self.ca_cert_path = Path(self.ca_cert_path)
        if isinstance(self.ca_key_path, str):
            self.ca_key_path = Path(self.ca_key_path)

        # Normalize logging levels
        self.memory_log_level = self.memory_log_level.upper()
        self.memory_alert_level = self.memory_alert_level.upper()

        # Validate thresholds
        if self.leak_detection_threshold < 0 or self.leak_detection_threshold > 1:
            raise ValueError("Leak detection threshold must be between 0 and 1")
        if self.leak_growth_rate < 0:
            raise ValueError("Leak growth rate must be non-negative")
        if self.memory_sample_interval <= 0:
            raise ValueError("Memory sample interval must be positive")

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ProxyConfig':
        """Create config from dictionary, handling both snake_case and camelCase."""
        # Convert camelCase to snake_case
        converted_dict = {}
        for key, value in config_dict.items():
            # Convert camelCase to snake_case (e.g., interceptRequests -> intercept_requests)
            snake_key = ''.join(['_' + c.lower() if c.isupper() else c.lower() for c in key]).lstrip('_')
            converted_dict[snake_key] = value

        # Only include keys that match our fields
        filtered_dict = {
            k: v for k, v in converted_dict.items()
            if k in cls.__dataclass_fields__
        }
        return cls(**filtered_dict)

    def update(self, new_config: dict) -> None:
        """Update config with new values."""
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_memory_settings(self) -> dict:
        """Get all memory-related settings as a dictionary."""
        return {
            'sample_interval': self.memory_sample_interval,
            'growth_threshold': self.memory_growth_threshold,
            'sample_retention': self.memory_sample_retention,
            'log_level': self.memory_log_level,
            'alert_level': self.memory_alert_level,
            'leak_detection': {
                'threshold': self.leak_detection_threshold,
                'min_samples': self.leak_detection_samples,
                'growth_rate': self.leak_growth_rate
            },
            'cleanup': {
                'timeout': self.cleanup_timeout,
                'force_threshold': self.force_cleanup_threshold,
                'retry_delay': self.cleanup_retry_delay
            }
        }
