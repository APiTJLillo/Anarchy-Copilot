# Anarchy Copilot Makefile

.PHONY: install test test-unit test-integration test-api test-coverage clean docs lint format help

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
COVERAGE := coverage
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy
PRE_COMMIT := pre-commit

# Directories
SRC_DIRS := anarchy_copilot recon_module vuln_module
TEST_DIR := tests
DOCS_DIR := docs
REPORTS_DIR := test-reports

# Test settings
TEST_ARGS ?= -v
COVERAGE_THRESHOLD := 80

help:
	@echo "Anarchy Copilot Development Commands"
	@echo "==================================="
	@echo "make install          Install development dependencies"
	@echo "make install-test     Install test dependencies"
	@echo "make test            Run all tests"
	@echo "make test-unit       Run unit tests"
	@echo "make test-integration Run integration tests"
	@echo "make test-api        Run API tests"
	@echo "make test-coverage   Run tests with coverage report"
	@echo "make lint            Run linting checks"
	@echo "make format          Format code using black and isort"
	@echo "make typecheck       Run type checking"
	@echo "make clean           Clean build and test artifacts"
	@echo "make docs            Build documentation"
	@echo "make validate        Run full validation suite"

# Installation targets
install:
	$(PIP) install -e .[dev]

install-test:
	$(PIP) install -r tests/requirements-test.txt

# Test targets
test:
	$(PYTEST) $(TEST_ARGS)

test-unit:
	$(PYTEST) -m "unit" $(TEST_ARGS)

test-integration:
	$(PYTEST) -m "integration" $(TEST_ARGS)

test-api:
	$(PYTEST) -m "api" $(TEST_ARGS)

test-coverage:
	$(PYTEST) --cov=$(SRC_DIRS) \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-report=xml \
		--cov-fail-under=$(COVERAGE_THRESHOLD) \
		$(TEST_ARGS)

# Development targets
lint:
	$(FLAKE8) $(SRC_DIRS) $(TEST_DIR)
	$(PRE_COMMIT) run --all-files

format:
	$(BLACK) $(SRC_DIRS) $(TEST_DIR)
	$(ISORT) $(SRC_DIRS) $(TEST_DIR)

typecheck:
	$(MYPY) $(SRC_DIRS)

validate: format lint typecheck test

# Documentation targets
docs:
	cd $(DOCS_DIR) && make html

# Utility targets
clean:
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	rm -rf $(REPORTS_DIR)/*
	rm -rf **/__pycache__
	rm -rf build/ dist/ *.egg-info

# Development environment setup
dev-setup: install install-test
	$(PRE_COMMIT) install
	mkdir -p $(REPORTS_DIR)

# Docker targets
docker-build:
	docker build -t anarchy-copilot:dev -f Dockerfile.dev .

docker-test:
	docker build -t anarchy-copilot:test -f Dockerfile.test .
	docker run --rm anarchy-copilot:test

# Continuous Integration targets
ci: 
	make install-test
	make lint
	make typecheck
	make test-coverage

# Watch mode for development
watch-tests:
	$(PYTEST) --looponfail $(TEST_ARGS)

# Run specific test file or directory
test-file:
	$(PYTEST) $(TEST_FILE) $(TEST_ARGS)

# Run tests by marker
test-marked:
	$(PYTEST) -m "$(MARKER)" $(TEST_ARGS)

# Run tests matching pattern
test-pattern:
	$(PYTEST) -k "$(PATTERN)" $(TEST_ARGS)

# Generate test report
test-report: test-coverage
	$(PYTHON) tests/tools/templates/__init__.py generate-report

# Security checks
security-check:
	bandit -r $(SRC_DIRS)
	safety check
	pip-audit

# Performance profiling
profile-tests:
	$(PYTEST) --profile $(TEST_ARGS)

# Coverage enforcement
coverage-enforce:
	$(COVERAGE) report --fail-under=$(COVERAGE_THRESHOLD)

# Initialize test environment
init-test-env:
	mkdir -p $(REPORTS_DIR)
	cp tests/tools/templates/test_report.html $(REPORTS_DIR)/template.html
	cp tests/tools/templates/styles/report.css $(REPORTS_DIR)/styles/report.css

# Default target
default: help
