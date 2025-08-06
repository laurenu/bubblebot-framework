# Testing Guide 🧪

This directory contains all tests for the Bubblebot API backend.

## Test Structure

- `test_document_processor.py` - Tests for document processing functionality
- `test_embedding_retrieval.py` - Tests for embeddings and retrieval functionality
- `test_integration_*.py` - Integration tests (require external services)
- `__init__.py` - Makes the tests directory a Python package

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Virtual environment activated
- Development dependencies installed: `pip install -r requirements-dev.txt`

### 🎯 Recommended: Use the Test Runner Scripts

#### Running Unit Tests (Default)

For most testing needs, use the convenient `run_tests.sh` script from the api directory:

```bash
# From the api directory
./run_tests.sh                    # Run all unit tests
./run_tests.sh -v                 # Run with verbose output
./run_tests.sh -c                 # Run with coverage report
./run_tests.sh -c -h              # Run with coverage + HTML report
./run_tests.sh -f test_document_processor.py  # Run specific file
./run_tests.sh -k 'document_type'             # Run tests matching pattern
./run_tests.sh --help             # Show all options
```

#### Running Integration Tests

⚠️ **Important**: Integration tests make real API calls to external services (like OpenAI) and may incur costs.

To run integration tests, use the dedicated integration test runner:

```bash
# From the api directory
chmod +x run_integration_tests.sh  # Make script executable if needed
./run_integration_tests.sh         # Run all integration tests
```

The script will:
1. Show a warning about potential costs
2. List which tests will be executed
3. Ask for confirmation before proceeding
4. Run the integration tests with coverage reporting

### Running Tests Manually

If you prefer to run pytest directly:

```bash
# Run all unit tests (excludes integration tests)
pytest -k "not integration"

# Run integration tests (requires valid API keys)
pytest tests/test_integration_*.py -v

# Run with coverage report
pytest --cov=app

# Run with coverage and generate HTML report
pytest --cov=app --cov-report=html
```

## Test Organization

- **Unit Tests**: Test individual components in isolation with all external dependencies mocked.
- **Integration Tests**: Test interactions between components and with external services.
  - Located in files matching `test_integration_*.py`
  - May make real API calls to external services
  - Should be run separately from unit tests

## Writing Tests

- Place new unit tests in appropriate test files
- Prefix integration test files with `test_integration_`
- Use descriptive test names and docstrings
- Mock external API calls in unit tests
- Document any required setup for integration tests

### Running Specific Tests

```bash
# Run a specific test file
pytest tests/test_document_processor.py

# Run a specific test class
pytest tests/test_document_processor.py::TestDocumentProcessor

# Run a specific test method
pytest tests/test_document_processor.py::TestDocumentProcessor::test_document_type_detection

# Run tests matching a pattern
pytest -k "document_type"
```

### Running Tests with Different Options

```bash
# Run tests in parallel (if pytest-xdist is installed)
pytest -n auto

# Stop on first failure
pytest -x

# Show local variables in tracebacks
pytest -l

# Run tests with asyncio support (for async tests)
pytest --asyncio-mode=auto
```

### Test Coverage

```bash
# Generate coverage report
pytest --cov=app --cov-report=term-missing

# Generate HTML coverage report (creates htmlcov/ directory)
pytest --cov=app --cov-report=html

# Generate XML coverage report (for CI/CD)
pytest --cov=app --cov-report=xml
```

**Note:** The `./run_tests.sh -c -h` command provides an easy way to generate coverage reports.

## 🐛 Debugging Tests

### Running Tests in Debug Mode

```bash
# Run with detailed output
pytest -vvv

# Run with print statement output
pytest -s

# Run a single test with debugger
python -m pytest tests/test_document_processor.py::TestDocumentProcessor::test_document_type_detection -s
```

### Common Test Issues

1. **Import Errors**: Make sure you're in the correct directory and virtual environment is activated
2. **Async Test Issues**: Ensure `pytest-asyncio` is installed and `@pytest.mark.asyncio` decorators are used
3. **File Path Issues**: Tests use temporary files, ensure proper cleanup in fixtures

## 📊 Test Results

After running tests, you'll see output similar to:

```
============================= test session starts ==============================
platform darwin -- Python 3.9.7, pytest-7.4.3, pluggy-1.3.0
rootdir: /path/to/bubblebot-framework/api
plugins: asyncio-0.21.1, cov-4.1.0
collected 15 items

tests/test_document_processor.py ................                    [100%]

============================== 15 passed in 2.34s ==============================
```

## 🔧 Test Configuration

Tests use pytest configuration. You can create a `pytest.ini` file in the api directory for custom settings:

```ini
[tool:pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

## 🤝 Contributing

When adding new tests:

1. Follow the existing naming convention: `test_*.py` for files, `test_*` for functions
2. Use descriptive test names that explain what is being tested
3. Add appropriate fixtures for reusable test data
4. Include both unit tests and integration tests where appropriate
5. Ensure tests are isolated and don't depend on external state

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Python Testing Best Practices](https://realpython.com/python-testing/) 
