# Testing Guide 🧪

This guide covers the testing strategy and how to run tests for the Bubblebot Framework.

## Test Types

### Unit Tests

Unit tests verify individual components in isolation using mocks. These tests:
- Are fast and reliable
- Don't make external API calls
- Should cover all major code paths
- Are run by default with `pytest`

### Integration Tests

Integration tests verify that components work together correctly. These tests:
- May make real API calls to external services
- May incur costs on your provider account
- Require valid API credentials
- Are excluded by default

## Running Tests

### Running All Unit Tests

```bash
# Run all unit tests (excludes integration tests)
./run_tests.sh

# With verbose output
./run_tests.sh -v
```

### Running Integration Tests

⚠️ **Warning**: Integration tests make real API calls that may incur costs.

```bash
# Run all tests, including integration tests
./run_integration_tests.sh

# You'll be prompted to confirm before any API calls are made
```

### Running Specific Tests

Run tests in a specific file:
```bash
pytest tests/test_embedding_service.py
```

Run a specific test class or method:
```bash
pytest tests/test_retrieval_service.py::TestRetrievalService
pytest tests/test_retrieval_service.py::TestRetrievalService::test_retrieve_context_success
```

Run tests by marker:
```bash
# Run only fast tests
pytest -m "not slow"

# Run only integration tests
pytest -m "integration"
```

### Test Coverage

Generate a coverage report:
```bash
pytest --cov=app --cov-report=term-missing
```

## Writing Tests

### Test Organization

- Unit tests are co-located with the code they test in the `tests/` directory
- Test files are named `test_*.py`
- Test classes are named `Test*`
- Test methods are named `test_*`

### Fixtures

Common test fixtures are defined in `conftest.py`. These include:

- `mock_embedding_provider`: A mock embedding provider for unit tests
- `sample_document_chunks`: Sample document chunks for testing
- `sample_embeddings`: Sample embeddings for testing

### Best Practices

1. **Isolation**: Each test should be independent
2. **Determinism**: Tests should produce the same results every time
3. **Descriptive Names**: Test names should describe what they test
4. **Mocks**: Use mocks for external dependencies in unit tests
5. **Cleanup**: Clean up any test data after each test

## Integration Test Details

### Cost Considerations

Integration tests may make real API calls to external services, which could incur costs. The test runner will warn you before running these tests.

### Environment Setup

For integration tests to work, you'll need to set up the appropriate environment variables:

```bash
# For OpenAI provider
EMBEDDING_PROVIDER=openai
EMBEDDING_PROVIDER_API_KEY=your_openai_api_key

# For Gemini provider
EMBEDDING_PROVIDER=gemini
EMBEDDING_PROVIDER_API_KEY=your_gemini_api_key
```

### Skipping Integration Tests

To skip integration tests:

```bash
# Skip all integration tests
pytest -k "not integration"

# Or use the run_tests.sh script
./run_tests.sh
```

## Debugging Tests

To drop into the debugger on test failures:

```bash
pytest --pdb
```

For more verbose output:

```bash
pytest -v
