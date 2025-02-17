#!/bin/bash
set -e

# Create test results directory if it doesn't exist
mkdir -p test-results

# Build and run tests in Docker
echo "Building test environment..."
docker-compose -f docker-compose.test.yml build

echo "Running proxy API tests..."
docker-compose -f docker-compose.test.yml run --rm test pytest \
    -v \
    --tb=short \
    --html=/app/test-results/report.html \
    --junitxml=/app/test-results/junit.xml \
    --cov=api/proxy \
    --cov-report=html:/app/test-results/coverage \
    tests/anarchy_copilot/api/test_proxy*.py

# Store exit code
EXIT_CODE=$?

# Cleanup
echo "Cleaning up..."
docker-compose -f docker-compose.test.yml down -v

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Tests failed!"
fi

echo "Test results available at:"
echo " - HTML Report: ./test-results/report.html"
echo " - Coverage Report: ./test-results/coverage/index.html"
echo " - JUnit XML: ./test-results/junit.xml"

# Exit with test status
exit $EXIT_CODE
