# Anarchy Copilot

AI-powered bug bounty suite that integrates modern technologies to streamline the bug bounty process. The suite includes a custom web proxy with advanced interception capabilities, along with comprehensive reconnaissance and vulnerability scanning features.

## Features

- **Proxy Client**: A client application that polls for proxy settings and applies them as needed.
- **Local Proxy Checkbox**: A UI checkbox to enable the local proxy, which only activates when the Docker proxy is running.
- **Proxy Settings Endpoint**: A new `/settings` endpoint to provide current proxy settings to the client.


### Reconnaissance Module ✅
- Automated subdomain discovery
- Port scanning and service detection
- Web endpoint analysis
- Continuous monitoring with change detection
- Scheduling system for recurring scans
- AI-assisted pattern recognition for interesting findings
- Real-time progress tracking

### Vulnerability Discovery Module 🔄
- **Advanced Proxy Platform**:
  - Custom web attack proxy with request/response interception
  - Real-time traffic analysis and pattern matching
  - Session management and replay capabilities
  - HTTP/HTTPS/HTTP2 and WebSocket support
  - Plugin system for custom interceptors
  - Automated security testing workflows

- **Integrated Scanning**:
  - Nuclei integration as primary scanning engine
  - Auxiliary tool integration (SQLMap, XSSHunter, etc.)
  - AI-driven parameter mutation and fuzzing
  - Automated vulnerability verification
  - Advanced false positive reduction with ML
  - JavaScript deobfuscation and analysis

- **Performance & Scale**:
  - Efficient traffic handling and connection pooling
  - Distributed scanning capabilities
  - Advanced resource management
  - Real-time monitoring and alerting

## Installation

```bash
# Development installation
pip install -e ".[dev]"

# Regular installation
pip install .
```

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/anarchycopilot/anarchycopilot.git
cd anarchycopilot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Run environment validation:
```bash
python -m tests.tools.validate_environment
```

5. Run tests:
```bash
make test
```

## Key Components

```
anarchy-copilot/
├── models/              # Database models
│   ├── base.py         # Core models (User, Project)
│   ├── recon.py        # Reconnaissance models
│   └── vulnerability.py # Vulnerability and reporting models
├── api/                # API endpoints and handlers
│   ├── projects/       # Project management
│   └── proxy/         # Web Proxy Module
│       ├── core.py    # Proxy server implementation
│       ├── interceptor.py # Request/response interception
│       ├── session.py # Session management
│       └── config.py  # Proxy configuration
├── recon_module/         # Reconnaissance module
│   ├── models.py        # Data structures
│   ├── recon_manager.py # Main interface
│   ├── scheduler.py     # Scheduling system
│   └── tools/          # Scanner implementations
├── vuln_module/         # Vulnerability module
│   ├── models.py       # Vulnerability models
│   ├── vuln_manager.py # Vulnerability interface
│   └── scanner/       # Scanner implementations
├── tests/              # Test suite ✨
│   ├── tools/         # Test utilities
│   ├── examples/      # Example tests
│   └── data/         # Test data & assets
├── examples/          # Usage examples
└── docker-compose.yml # Container orchestration
```

## Components

### Web Proxy Module
- Custom HTTP/HTTPS proxy with advanced features
- Request/response interception and modification
- Session management and history tracking
- Plugin system for custom security checks
- Certificate authority for HTTPS inspection
- Real-time traffic analysis

## Testing

### Quick Start
```bash
# Validate environment
python -m tests.tools.validate_environment

# Run all tests
make test

# Run specific test types
make test-unit         # Unit tests
make test-integration  # Integration tests
make test-api         # API tests

# View test reports
open test-reports/report.html
```

### Test Categories
- Unit tests (fast, no external dependencies)
- Integration tests (external tool interactions)
- API tests (endpoint testing)
- Example tests (usage patterns)
- Security tests (scanning & validation)

### Test Reports
- HTML test reports with coverage metrics
- JUnit XML reports for CI integration
- Performance profiling results
- Security scan summaries

For detailed testing information, see `tests/README.md`.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Set up development environment:
   ```bash
   make dev-setup
   python -m tests.tools.validate_environment
   ```
4. Make your changes
5. Run tests and checks:
   ```bash
   make validate  # Runs formatting, linting, type checking, and tests
   ```
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
