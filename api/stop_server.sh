#!/bin/bash
# Stop script for Bubblebot API server

echo "🛑 Stopping Bubblebot API server..."

# Kill any uvicorn processes
pkill -f uvicorn

# Alternative: kill processes on port 8000
# lsof -ti:8000 | xargs kill -9

echo "✅ Server stopped" 
