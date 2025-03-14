#!/bin/bash
set -e

# API service entrypoint script

# Initialize logging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Verify development tools
verify_tools() {
    log "Verifying development tools..."
    
    # Check required tools
    if ! command_exists python; then
        log "ERROR: Python not found"
        exit 1
    fi
    
    if ! command_exists pip; then
        log "ERROR: pip not found"
        exit 1
    fi
    
    log "Required tools verification complete"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    if command_exists alembic; then
        alembic upgrade head
    else
        log "WARNING: alembic not found, skipping migrations"
    fi
}

# Setup development environment
setup_dev_env() {
    log "Setting up development environment..."
    
    # Create necessary directories
    mkdir -p /app/logs
    mkdir -p /app/data
    mkdir -p /app/tests/data/output
    
    # Install dependencies
    log "Installing package and dependencies..."
    pip install --no-cache-dir -e .
    log "Installing development dependencies..."
    pip install --no-cache-dir -e ".[dev]"
    
    # Run database migrations
    run_migrations
}

# Start API server
start_api() {
    log "Starting API server..."
    log "Current directory: $(pwd)"
    log "Python path: $PYTHONPATH"
    
    # Start the FastAPI application with debugger
    python -m debugpy --listen 0.0.0.0:5678 \
        -m uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --log-level debug \
        --proxy-headers \
        --timeout-keep-alive 75
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    pkill -f "uvicorn" || true
}

# Register cleanup handler
trap cleanup EXIT

# Main execution
case "$1" in
    "setup")
        verify_tools
        setup_dev_env
        ;;
    "start")
        verify_tools
        setup_dev_env
        start_api
        ;;
    *)
        log "Usage: $0 {setup|start}"
        log "  setup  - Set up development environment"
        log "  start  - Start API server"
        exit 1
        ;;
esac 