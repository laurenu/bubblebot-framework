#!/bin/bash
# Start script for Bubblebot API server

# Navigate to the api directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Start the server
echo "🚀 Starting Bubblebot API server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 
