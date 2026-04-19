#!/bin/bash
# Quick Start Guide for Paracodex Pipeline Web App
# Works from any directory — paths are relative to this script's location.

# Navigate to webapp directory (relative to this script's location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================="
echo "PARACODEX PIPELINE WEB APPLICATION - QUICK START"
echo "======================================================================="
echo ""
echo "📁 Working directory: $SCRIPT_DIR"
echo ""

# Check Python is available
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "❌ Python not found. Please install Python 3.8+ and try again."
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)

# Check if dependencies are installed
if ! $PYTHON -c "import flask" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    $PYTHON -m pip install -q -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

echo ""
echo "🚀 Starting Paracodex Pipeline Web Application..."
echo ""
echo "======================================================================="
echo "  Web Interface: http://localhost:5000"
echo "======================================================================="
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask app
$PYTHON app.py
