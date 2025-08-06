# Bubblebot Framework 🫧

A flexible, provider-agnostic framework for document processing, embedding, and retrieval with support for multiple AI providers.

## Features

- **Multi-provider Support**: Easily switch between different embedding providers (OpenAI, Gemini, etc.)
- **Document Processing**: Process various document formats (TXT, PDF, etc.) with automatic chunking
- **Vector Search**: Efficient similarity search for document chunks
- **REST API**: Easy integration with other services
- **Tested**: Comprehensive test suite with unit and integration tests

## Architecture

```
bubblebot-framework/
├── api/                   # FastAPI application
│   ├── app/              
│   │   ├── core/         # Core configuration and utilities
│   │   ├── models/       # Pydantic models
│   │   ├── services/     # Business logic
│   │   │   └── providers/ # Provider implementations
│   │   └── tests/        # Test files
│   └── requirements.txt  # Python dependencies
└── README.md            # This file
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bubblebot-framework.git
   cd bubblebot-framework/api
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running Tests

Run unit tests:
```bash
./run_tests.sh
```

Run integration tests (may incur API costs):
```bash
./run_integration_tests.sh
```

### Starting the Server

```bash
./start_server.sh
```

The API will be available at `http://localhost:8000`

## Configuration

Edit the `.env` file to configure:
- Embedding provider (OpenAI, Gemini, etc.)
- API keys
- Model settings
- Batch sizes and other parameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT

## 🚀 Quick Start

### Backend API

The Bubblebot API provides document processing, AI integration, and multi-tenant chatbot functionality.

**Getting Started:**
```bash
cd api
./start_server.sh          # Start the API server
```

**For detailed backend documentation, see [api/README.md](api/README.md)**

### Frontend Web Application

The web frontend will provide an intuitive interface for interacting with the Bubblebot Framework.

**Status:** Coming soon - Development in progress

## 🧪 Testing

### Backend Testing

The API backend includes comprehensive test suites for all functionality.

**For detailed testing information, see [api/tests/README.md](api/tests/README.md)**

**Quick test commands:**
```bash
cd api
./run_tests.sh             # Run all tests
./run_tests.sh -c          # Run with coverage
./run_tests.sh --help      # Show all options
```

## 📚 Documentation

- **[Backend API Guide](api/README.md)** - Complete backend setup and development
- **[Testing Guide](api/tests/README.md)** - Comprehensive testing documentation
- **[Architecture Docs](docs/architecture/)** - System design and architecture
- **[Setup Guide](docs/setup/)** - Installation and configuration
- **[API Documentation](docs/api/)** - API reference and examples

## 🎯 Key Features

### Backend (Current)

TBD

### Frontend (Planned)

TBD

## 📄 [License](LICENSE)


## 🔗 Links

- **Backend Documentation:** [api/README.md](api/README.md)
- **Testing Guide:** [api/tests/README.md](api/tests/README.md)
---

**Note:** This project is actively under development. 
