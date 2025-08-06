#!/bin/bash

# --- Integration Test Runner ---
# This script runs integration tests that may incur API costs.

# Stop on first error
set -e

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to print messages
print_message() {
    echo ""
    echo "----------------------------------------------------------------"
    echo "$1"
    echo "----------------------------------------------------------------"
    echo ""
}

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    print_message "🐍 Activating Python virtual environment..."
    source "$SCRIPT_DIR/venv/bin/activate"
else
    print_message "⚠️ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# --- Warning and Confirmation ---
print_message "⚠️ WARNING: Running integration tests may incur API costs. ⚠️"
read -p "Do you want to continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    print_message "🛑 Integration tests cancelled."
    exit 1
fi

# --- Run Integration Tests ---
print_message "🚀 Running integration tests..."
pytest -v -k "integration" "$SCRIPT_DIR/tests/"

print_message "✅ Integration tests completed!"
