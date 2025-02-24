"""API configuration settings."""
import logging
import json
from typing import Any, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field, computed_field

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """API configuration settings.

    This class manages API settings with support for environment variables.
    Environment variables are prefixed with ANARCHY_ and can be:
    - Simple values: ANARCHY_DEBUG=true
    - Comma-separated lists: ANARCHY_CORS_ORIGINS=http://localhost:3000,http://localhost:8000
    """
    # API Settings
    api_title: str = "Anarchy Copilot API"
    api_version: str = "0.1.0"
    debug: bool = False
    cors_origins_input: str = Field(
        default="http://localhost:3000",
        description="Comma-separated list of allowed CORS origins",
        alias="cors_origins",
    )

    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        if not self.cors_origins_input:
            return ["http://localhost:3000"]
        return [
            origin.strip() 
            for origin in self.cors_origins_input.split(",") 
            if origin.strip()
        ]

    # Proxy Settings
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 8080
    proxy_intercept_requests: bool = True
    proxy_intercept_responses: bool = True
    proxy_max_connections: int = 100
    proxy_max_keepalive_connections: int = 20
    proxy_keepalive_timeout: int = 30

    model_config = SettingsConfigDict(
        env_prefix="ANARCHY_",
        validate_default=True,
        case_sensitive=False
    )

    def __init__(self, **data):
        """Initialize settings and log the configuration."""
        super().__init__(**data)
        logger.info(f"Initialized Settings: {self.model_dump()}")
