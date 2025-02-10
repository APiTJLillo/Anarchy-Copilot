# Anarchy Copilot Test Suite

This directory contains the test suite for the Anarchy Copilot project. The test infrastructure is designed to support unit tests, integration tests, API tests, and security validation tests.

## Requirements

- Python 3.8 or higher
- Nuclei 2.8.0 or higher
- Docker (optional, for containerized testing)

Install test dependencies:
```bash
make install-test
```

## Directory Structure

```
tests/
├── anarchy_copilot/      # Tests for core package
├── data/                 # Test data files
│   └── nuclei_templates/ # Nuclei test templates
├── examples/             # Example tests and usage patterns
├── tools/               # Test utilities and helpers
│   ├── templates/       # Report templates
│   └── validate_environment.py
├── vuln_module/         # Vulnerability module tests
├── conftest.py         # Global test configuration
├── requirements-test.txt # Test dependencies
└── README.md           # This file
```

## Running Tests

### Quick Start

1. Validate your test environment:
```bash
python -m tests.tools.validate_environment
```

2. Run all tests:
```bash
make test
```

### Test Categories

- Unit tests: `make test-unit`
- Integration tests: `make test-integration`
- API tests: `make test-api`
- Security tests: `make security-check`

### Running Specific Tests

```bash
# Run tests in a specific file
make test-file TEST_FILE=tests/vuln_module/test_models.py

# Run tests matching a pattern
make test-pattern PATTERN="test_scan"

# Run tests with a specific marker
make test-marked MARKER="integration"
```

### Test Coverage

Generate coverage report:
```bash
make test-coverage
```

View the HTML coverage report:
```bash
open test-reports/coverage/index.html
```

## Test Configuration

### Environment Variables

- `TESTING=true`: Enables test mode
- `TEST_DEBUG=true`: Enables debug logging
- `OFFLINE_TESTS=1`: Skip tests requiring network access
- `COVERAGE_THRESHOLD=80`: Minimum required coverage percentage

### pytest Configuration

See `pytest.ini` for detailed configuration settings including:
- Test discovery patterns
- Markers
- Coverage settings
- Timeout settings
- Report generation

## Test Reports

Test results are written to `test-reports/` and include:
- JUnit XML report (`junit.xml`)
- Coverage report (`coverage.xml`, `htmlcov/`)
- HTML test report (`report.html`)

### Generating Reports

```bash
# Generate comprehensive test report
make test-report

# View the report
open test-reports/report.html
```

## Writing Tests

### Test Categories (Markers)

- `@pytest.mark.unit`: Unit tests (fast, no external dependencies)
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.api`: API endpoint tests
- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.network`: Tests requiring network access
- `@pytest.mark.security`: Security-related tests

### Example Test

```python
import pytest
from anarchy_copilot.vuln_module import VulnScanner

@pytest.mark.integration
async def test_vulnerability_scan(example_target):
    scanner = VulnScanner()
    results = await scanner.scan(example_target)
    assert results.status == "completed"
    assert len(results.findings) > 0
```

### Test Utilities

Common test utilities are available in `tests.tools`:

```python
from tests.tools import (
    validate_environment,
    validate_templates,
    generate_test_summary
)
```

### Fixtures

Common fixtures are defined in `conftest.py`:

```python
@pytest.fixture
def example_target(temp_dir):
    # Setup test target
    yield target
    # Cleanup
```

## Development Workflow

1. Set up your environment:
```bash
make dev-setup
```

2. Run environment validation:
```bash
python -m tests.tools.validate_environment
```

3. Write tests for new features:
```bash
touch tests/module_name/test_feature.py
```

4. Run tests and check coverage:
```bash
make test-coverage
```

5. Generate and review test report:
```bash
make test-report
```

## Continuous Integration

Tests are automatically run in CI for:
- Pull requests to main/develop
- Push to main/develop
- Scheduled security scans

See `.github/workflows/test.yml` for CI configuration.

## Contributing Tests

1. Follow the existing test structure
2. Include both positive and negative test cases
3. Add appropriate markers
4. Maintain minimum 80% coverage
5. Add to relevant test requirements if needed
6. Include test documentation
7. Validate templates if adding new ones

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
```bash
pip install -r tests/requirements-test.txt
```

2. **Environment Setup**
```bash
python -m tests.tools.validate_environment
```

3. **Test Failures**
- Check test reports in `test-reports/`
- Enable debug logging: `TEST_DEBUG=true`
- Run specific test with -v: `pytest -v test_file.py`

### Getting Help

- Check test documentation in docstrings
- Review test examples in `tests/examples/`
- Run environment validation tool
- Check CI build logs
