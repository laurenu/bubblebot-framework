#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print header
print_header() {
    echo -e "${YELLOW}================================================${NC}"
    echo -e "${YELLOW}  BUBBLEBOT INTEGRATION TEST RUNNER${NC}"
    echo -e "${YELLOW}================================================${NC}"
}

# Show warning about potential costs
show_warning() {
    echo -e "${YELLOW}⚠️  WARNING: Integration tests make real API calls to external services${NC}"
    echo -e "${YELLOW}   This may incur costs on your OpenAI account.${NC}\n"
    
    echo -e "The following tests will be executed:"
    echo -e "  • Document Processor Integration Tests"
    echo -e "  • Embedding & Retrieval Integration Tests\n"
    
    read -p "Do you want to continue? [y/N] " -n 1 -r
    echo    # move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        echo -e "${RED}❌ Test execution cancelled by user${NC}"
        exit 1
    fi
}

# Main execution
main() {
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi

    print_header
    show_warning

    # Run integration tests with coverage
    echo -e "${GREEN}🚀 Running integration tests...${NC}"
    python -m pytest tests/test_integration_*.py -v --cov=app --cov-report=term-missing
    
    # Check test result
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✅ All integration tests passed!${NC}"
    else
        echo -e "\n${RED}❌ Some integration tests failed${NC}"
        exit 1
    fi
}

# Run the main function
main "$@"
