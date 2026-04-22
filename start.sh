#!/bin/bash

# ============================================================================
# Optimal Sample Selection System - Web Application Launcher
# ============================================================================

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

show_help() {
    echo "Usage: ./start.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  start     Start the web application (default)"
    echo "  install   Install dependencies only"
    echo "  test      Run the algorithm test suite"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./start.sh         # Start web app on port 8080"
    echo "  ./start.sh install  # Install dependencies"
    echo "  ./start.sh test    # Run test suite"
}

# Check if Python 3 is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python 3 is not installed."
        echo "Please install Python 3 and try again."
        exit 1
    fi
}

# Install dependencies
install_deps() {
    check_python
    echo "Installing dependencies..."
    python3 -m pip install --quiet --upgrade pip
    python3 -m pip install --quiet flask numpy numba
    echo "Dependencies installed successfully."
}

# Run tests
run_tests() {
    check_python
    echo "Running test suite..."
    python3 test_algorithm.py
}

# Start web app
start_webapp() {
    check_python

    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        echo "Virtual environment created."
        echo ""
    fi

    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "Virtual environment activated."
    echo ""

    # Install/update dependencies
    echo "Installing dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet flask numpy numba
    echo "Dependencies ready."
    echo ""

    # Get local IP address for mobile access
    LOCAL_IP=""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)
    else
        LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi

    echo "=========================================="
    echo "  Optimal Sample Selection System"
    echo "  Web Application"
    echo "=========================================="
    echo ""
    echo "  Local:   http://localhost:3000"
    if [ -n "$LOCAL_IP" ]; then
        echo "  Mobile:  http://$LOCAL_IP:3000"
    fi
    echo "  Network: http://0.0.0.0:3000"
    echo ""
    echo "  Press Ctrl+C to stop the server"
    echo "=========================================="
    echo ""

    # Start the Flask application
    python3 web_app.py
}

# Parse arguments
case "${1:-start}" in
    start)
        start_webapp
        ;;
    install)
        install_deps
        ;;
    test)
        run_tests
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown option: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
