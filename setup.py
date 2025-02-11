"""Setup file for Anarchy Copilot."""

import os
from setuptools import setup, find_packages

# Read version from __init__.py
with open(os.path.join("anarchy_copilot", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip("'").strip('"')
            break

# Read README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="anarchy_copilot",
    version=version,
    description="AI-powered bug bounty suite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anarchy Copilot Team",
    author_email="info@anarchycopilot.org",
    url="https://github.com/anarchycopilot/anarchycopilot",
    packages=find_packages(include=[
        "anarchy_copilot", 
        "anarchy_copilot.*",
        "recon_module",
        "recon_module.*",
        "vuln_module",
        "vuln_module.*"
    ]),
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.68.0",
        "sqlalchemy>=1.4.0",
        "uvicorn>=0.15.0",
        "aiohttp>=3.8.0",
        "asyncio>=3.4.3",
        "typing_extensions>=4.0.0",
        "pydantic>=1.8.0",
        "pyppeteer>=1.0.2",
        "psutil>=5.8.0",
        "python-multipart>=0.0.5",
        "dnspython>=2.1.0",
        "mitmproxy>=9.0.0",
        "python-nmap>=0.7.1",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.15.0",
            "pytest-html>=3.1.1",
            "mypy>=0.910",
            "black>=21.0",
            "isort>=5.0.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "anarchy-copilot=anarchy_copilot.cli:main",
        ],
    },
    package_data={
        "anarchy_copilot": ["py.typed", "*.pyi", "**/*.pyi"],
        "recon_module": ["py.typed", "*.pyi", "**/*.pyi"],
        "vuln_module": ["py.typed", "*.pyi", "**/*.pyi"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Security",
    ],
)
