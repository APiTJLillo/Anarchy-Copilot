# Anarchy Copilot

AI-powered bug bounty suite that integrates modern technologies to streamline the bug bounty process.

## Features

### Reconnaissance Module ✅
- Automated subdomain discovery
- Port scanning and service detection
- Web endpoint analysis
- Continuous monitoring with change detection
- Scheduling system for recurring scans
- AI-assisted pattern recognition for interesting findings
- Real-time progress tracking

### Vulnerability Discovery Module 🔄
- Integrated vulnerability scanners (Nuclei)
- Custom payload testing support
- Configurable scan settings
- Real-time scan monitoring
- False positive handling
- Extensible scanner interface for custom integrations
- Built-in rate limiting and resource management

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

## Project Structure

```
anarchy-copilot/
├── anarchy_copilot/        # Main package
├── recon_module/          # Reconnaissance module
│   ├── models.py         # Data structures
│   ├── recon_manager.py  # Main interface
│   ├── scheduler.py      # Scheduling system
│   └── tools/           # Scanner implementations
├── vuln_module/         # Vulnerability module
│   ├── models.py       # Vulnerability models
│   ├── vuln_manager.py # Vulnerability interface
│   └── scanner/       # Scanner implementations
├── tests/            # Test suite ✨
│   ├── tools/       # Test utilities
│   │   ├── templates/  # Report templates
│   │   ├── validate_environment.py
│   │   └── test_summary.py
│   ├── examples/    # Example tests
│   ├── data/       # Test data & assets
│   └── README.md   # Test documentation
├── examples/       # Usage examples
└── README.md
```

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
