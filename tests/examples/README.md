# Test Examples

This directory contains example tests demonstrating how to use the testing infrastructure and utilities provided by the Anarchy Copilot framework.

## Running Examples

```bash
# Run all example tests
pytest tests/examples/

# Run with detailed output
pytest -v tests/examples/

# Run specific example
pytest tests/examples/test_example.py::test_async_operation

# Run with coverage
pytest tests/examples/ --cov
```

## Example Categories

### 1. Basic Test Examples
- Basic fixture usage
- Environment setup
- Basic assertions

```python
def test_basic_fixtures(temp_dir, test_data_dir):
    assert temp_dir.exists()
    assert (test_data_dir / "nuclei_templates").exists()
```

### 2. Async Test Examples
- Async test functions
- Timing utilities
- Asyncio operations

```python
@pytest.mark.asyncio
async def test_async_operation():
    await asyncio.sleep(0.1)
    assert True
```

### 3. Test Categories
- Unit tests (`@pytest.mark.unit`)
- Integration tests (`@pytest.mark.integration`)
- Slow tests (`@pytest.mark.slow`)
- Network tests (`@pytest.mark.network`)

### 4. Timeout Handling
- Using timeout context managers
- Testing long-running operations

```python
def test_timeout_handling(strict_timeout):
    with strict_timeout(1):
        time.sleep(0.1)
```

### 5. Common Test Patterns
- Class-based tests
- Fixture usage
- Parameterized tests
- Error handling

```python
class TestExamplePatterns:
    @pytest.fixture(autouse=True)
    def setup(self, temp_dir):
        self.test_file = temp_dir / "test.txt"
        self.test_file.write_text("test content")
        yield

    @pytest.mark.parametrize("content,expected", [
        ("test1", 5),
        ("test22", 6),
        ("", 0)
    ])
    def test_parameterized_example(self, content, expected):
        self.test_file.write_text(content)
        assert len(self.test_file.read_text()) == expected
```

### 6. Logging Tests
- Log capture
- Log level testing
- Log message verification

```python
def test_logging(caplog):
    with caplog.at_level(logging.DEBUG):
        logger.debug("Debug message")
        assert "Debug message" in caplog.text
```

## Available Fixtures

1. **Basic Fixtures**
   - `temp_dir`: Temporary directory
   - `test_data_dir`: Test data directory
   - `project_root`: Project root path
   - `nuclei_templates_dir`: Nuclei templates directory

2. **Environment Fixtures**
   - `mock_env`: Mock environment variables
   - `setup_test_env`: Auto-configure test environment

3. **Timing Fixtures**
   - `timing_helper`: Timing test utilities
   - `strict_timeout`: Timeout context manager

4. **Logging Fixtures**
   - `caplog`: Log capture fixture

## Test Categories

Use markers to categorize tests:

```python
@pytest.mark.unit            # Unit tests
@pytest.mark.integration    # Integration tests
@pytest.mark.slow          # Slow tests (>1s)
@pytest.mark.network       # Network-requiring tests
@pytest.mark.api          # API tests
```

Run specific categories:
```bash
# Run only unit tests
pytest -v -m unit tests/examples/

# Skip slow tests
pytest -v -m "not slow" tests/examples/

# Run integration tests
pytest -v -m integration tests/examples/
```

## Best Practices

1. **Use Type Hints**
```python
def test_example(temp_dir: Path, mock_env: Dict[str, str]) -> None:
    assert temp_dir.exists()
```

2. **Clear Test Names**
- Use descriptive test names
- Follow `test_<what>_<condition>` pattern

3. **Test Organization**
- Group related tests in classes
- Use fixtures for common setup
- Keep tests focused and simple

4. **Error Handling**
- Test both success and failure cases
- Use `pytest.raises` for exceptions
- Check error messages

5. **Async Testing**
- Use `@pytest.mark.asyncio`
- Handle cleanup properly
- Use timing utilities for async operations

## Common Issues and Solutions

1. **Async Test Failures**
```python
# Wrong
def test_async():  # Missing asyncio mark
    async_function()

# Correct
@pytest.mark.asyncio
async def test_async():
    await async_function()
```

2. **Fixture Scope Issues**
```python
# Inefficient - creates for each test
@pytest.fixture
def expensive_resource(): ...

# Better - reuses across session
@pytest.fixture(scope="session")
def expensive_resource(): ...
```

3. **Cleanup Handling**
```python
# Using yield for cleanup
@pytest.fixture
def test_file(temp_dir):
    file = temp_dir / "test.txt"
    file.write_text("test")
    yield file
    file.unlink()  # Cleanup
```

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-Asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
