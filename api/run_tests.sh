#!/bin/bash
# Test runner script for Bubblebot API

# Navigate to the api directory
cd "$(dirname "$0")"

# Add the project root to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Activate virtual environment
source venv/bin/activate

# Default test options
TEST_MODE="all"
VERBOSE=false
COVERAGE=false
HTML_REPORT=false
STOP_ON_FAILURE=false
INCLUDE_INTEGRATION=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -h|--html)
            HTML_REPORT=true
            shift
            ;;
        -x|--stop-on-failure)
            STOP_ON_FAILURE=true
            shift
            ;;
        -f|--file)
            TEST_FILE="$2"
            TEST_MODE="file"
            shift 2
            ;;
        -k|--keyword)
            TEST_KEYWORD="$2"
            TEST_MODE="keyword"
            shift 2
            ;;
        --include-integration)
            INCLUDE_INTEGRATION=true
            shift
            ;;
        --help)
            echo "🧪 Bubblebot API Test Runner"
            echo ""
            echo "Usage: ./run_tests.sh [options]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose           Run tests with verbose output"
            echo "  -c, --coverage          Run tests with coverage report"
            echo "  -h, --html              Generate HTML coverage report"
            echo "  -x, --stop-on-failure   Stop on first test failure"
            echo "  -f, --file FILE         Run specific test file"
            echo "  -k, --keyword PATTERN   Run tests matching pattern"
            echo "  --include-integration    Include integration tests (not recommended for CI)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                    # Run all unit tests (excludes integration tests)"
            echo "  ./run_tests.sh --include-integration  # Run all tests including integration tests"
            echo "  ./run_tests.sh -v                 # Run all unit tests with verbose output"
            echo "  ./run_tests.sh -c                 # Run with coverage"
            echo "  ./run_tests.sh -c -h              # Run with coverage and HTML report"
            echo "  ./run_tests.sh -f test_document_processor.py"
            echo "  ./run_tests.sh -k 'document_type'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest"

# Add options based on flags
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=app"
fi

if [ "$HTML_REPORT" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov-report=html"
fi

if [ "$STOP_ON_FAILURE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -x"
fi

# Add test target based on mode
case $TEST_MODE in
    "all")
        if [ "$INCLUDE_INTEGRATION" = true ]; then
            PYTEST_CMD="$PYTEST_CMD tests/"
            echo "🧪 Running all tests (including integration tests)..."
            echo "⚠️  WARNING: Integration tests may make external API calls and incur costs"
        else
            PYTEST_CMD="$PYTEST_CMD -k 'not integration' tests/"
            echo "🧪 Running unit tests (excluding integration tests)..."
            echo "   Use --include-integration to run integration tests"
        fi
        ;;
    "file")
        PYTEST_CMD="$PYTEST_CMD tests/$TEST_FILE"
        echo "🧪 Running test file: $TEST_FILE"
        ;;
    "keyword")
        if [ "$INCLUDE_INTEGRATION" = true ]; then
            PYTEST_CMD="$PYTEST_CMD -k \"$TEST_KEYWORD\""
        else
            PYTEST_CMD="$PYTEST_CMD -k \"$TEST_KEYWORD and not integration\""
        fi
        echo "🧪 Running tests matching: $TEST_KEYWORD"
        if [ "$INCLUDE_INTEGRATION" != true ]; then
            echo "   (excluding integration tests, use --include-integration to include them)"
        fi
        ;;
esac

# Run the tests
echo "🚀 Starting tests with command: $PYTEST_CMD"
echo ""

eval $PYTEST_CMD

# Show coverage summary if HTML report was generated
if [ "$HTML_REPORT" = true ]; then
    echo ""
    echo "📊 HTML coverage report generated in htmlcov/ directory"
    echo "   Open htmlcov/index.html in your browser to view the report"
fi

echo ""
echo "✅ Tests completed!" 

if [ "$INCLUDE_INTEGRATION" != true ] && [ "$TEST_MODE" = "all" ]; then
    echo ""
    echo "💡 Tip: To run integration tests (may incur API costs), use:"
    echo "     ./run_tests.sh --include-integration"
    echo "     or"
    echo "     ./run_integration_tests.sh"
fi
