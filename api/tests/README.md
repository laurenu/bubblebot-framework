# Testing Guide 🧪

This directory contains all tests for the Bubblebot API backend.

## Test Structure

- `test_document_processor.py` - Tests for document processing functionality
- `__init__.py` - Makes the tests directory a Python package

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Virtual environment activated
- Development dependencies installed: `pip install -r requirements-dev.txt`

### Running All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=app

# Run with coverage and generate HTML report
pytest --cov=app --cov-report=html
```

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
