"""Setup file for Anarchy Copilot."""
from setuptools import setup, find_packages

import os
import re
from pathlib import Path

def get_version() -> str:
    """Get version from version.py."""
    version_file = Path(__file__).parent / "anarchy_copilot" / "version.py"
    if not version_file.exists():
        return "0.1.0"
    
    content = version_file.read_text()
    version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)
    if version_match:
        return version_match.group(1)
    return "0.1.0"

version = get_version()

setup(
    name="anarchy_copilot",
    version=version,  # Version is read from anarchy_copilot/version.py
    package_data={
        "anarchy_copilot": ["py.typed", "*.pyi", "**/*.pyi"],
        "api": ["py.typed", "*.pyi", "**/*.pyi"],
        "proxy": ["py.typed", "*.pyi", "**/*.pyi"],
        "recon_module": ["py.typed", "*.pyi", "**/*.pyi"],
        "vuln_module": ["py.typed", "*.pyi", "**/*.pyi"],
    },
    packages=find_packages(include=[
        "anarchy_copilot",
        "anarchy_copilot.*",
        "api",
        "api.*",
        "proxy",
        "proxy.*",
        "recon_module",
        "recon_module.*",
        "vuln_module",
        "vuln_module.*"
    ]),
    install_requires=[
        "fastapi>=0.100.0,<1.0.0",
        "uvicorn>=0.22.0,<1.0.0",
        "sqlalchemy>=2.0.0,<3.0.0",
        "alembic>=1.11.1,<2.0.0",
        "asyncpg>=0.28.0,<1.0.0",
        "pydantic>=2.0.0,<3.0.0",
        "email-validator>=2.0.0",
        "aiohttp>=3.8.0",
        "python-multipart>=0.0.5",
        "cryptography>=41.0.0",
        "debugpy>=1.6.7"
    ]
)
