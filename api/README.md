# Bubblebot API 🫧

This directory contains the complete Python backend for the Bubblebot Framework, including document processing services, models, utilities, and API endpoints.

## 🏗️ Architecture

The API is built with FastAPI and follows a clean architecture with these key components:

- **Document Processor**: Handles document ingestion, chunking, and text extraction
- **Embedding Service**: Manages text embeddings using pluggable providers
- **Retrieval Service**: Handles similarity search and context building
- **Provider System**: Abstract base classes for different AI providers

### Provider Architecture

The system uses a provider-agnostic design:

```
services/
└── providers/
    ├── __init__.py
    ├── base.py         # Abstract base classes
    ├── factory.py      # Provider factory
    ├── openai/         # OpenAI provider implementation
    └── gemini/         # Gemini provider implementation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip (Python package installer)

### Installation

1. **Navigate to the API directory:**
   ```bash
   cd api
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Upgrade pip:**
pip install --upgrade pip

4. **Install dependencies:**
   ```bash
   # For production
   pip install -r requirements.txt
   
   # For development (includes testing tools)
   pip install -r requirements-dev.txt
   ```
   
   **Note:** The development scripts automatically handle virtual environment activation, so you only need to install dependencies once during setup.

## 🏃‍♂️ Development Scripts

The project includes convenient scripts for common development tasks:

### **Start the API Server:**
```bash
./start_server.sh
```
- Automatically activates virtual environment
- Starts FastAPI server with auto-reload
- Runs on http://localhost:8000

### **Stop the API Server:**
```bash
./stop_server.sh
```
- Kills any running server instances
- Useful if you get "port already in use" errors

### **Run Tests:**
```bash
./run_tests.sh                    # Run all tests
./run_tests.sh -v                 # Run with verbose output
./run_tests.sh -c                 # Run with coverage report
./run_tests.sh -c -h              # Run with coverage + HTML report
./run_tests.sh --help             # Show all options
```

For detailed testing information, see [tests/README.md](tests/README.md).

## 🧪 Testing

### Unit Tests

Run all unit tests (excludes integration tests by default):

```bash
./run_tests.sh
```

### Integration Tests

Integration tests make real API calls to external services and may incur costs. You'll be prompted for confirmation before running them:

```bash
./run_integration_tests.sh
```

### Test Coverage

To run tests with coverage report:

```bash
pytest --cov=app --cov-report=term-missing tests/
```

## 🏗️ Development

### Code Quality Tools

The project includes several code quality tools:

```bash
# Format code with Black
black app/ tests/

# Sort imports with isort
isort app/ tests/

# Lint code with flake8
flake8 app/ tests/

# Type checking with mypy
mypy app/
```

### Pre-commit Hooks

If you have pre-commit installed, it will automatically run these tools:

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

### Adding a New Provider

1. Create a new provider module in `app/services/providers/`
2. Implement the required abstract methods from `BaseEmbeddingProvider`
3. Update the provider factory to support your new provider
4. Add tests for your provider implementation

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting

Run formatting and linting:

```bash
black .
isort .
flake8
```

## 📁 Project Structure

```
api/
├── app/                    # Main application code
│   ├── api/               # API endpoints
│   ├── core/              # Core functionality
│   ├── models/            # Data models
│   ├── services/          # Business logic services
│   └── utils/             # Utility functions
├── tests/                 # Test files
│   ├── README.md          # Testing guide
│   ├── test_document_processor.py
│   └── __init__.py
├── notebooks/             # Jupyter notebooks for demos
├── start_server.sh        # Server startup script
├── stop_server.sh         # Server shutdown script
├── run_tests.sh           # Test runner script
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the api directory for local development:

```bash
# Example .env file
DEBUG=True
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=10
```

## 🤝 Contributing

When contributing to the backend:

1. Follow the existing code style and patterns
2. Add appropriate tests for new functionality
3. Update documentation as needed
4. Ensure all code quality checks pass
5. Use type hints and docstrings for clarity

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Python Best Practices](https://realpython.com/python-best-practices/)
- [Testing Guide](tests/README.md) 

## 📝 License

MIT
