#!/bin/bash
set -e

# Proxy service entrypoint script

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

# Setup proxy environment
setup_proxy_env() {
    log "Setting up proxy environment..."
    
    # Create necessary directories
    mkdir -p /app/logs
    mkdir -p /app/data
    mkdir -p /app/certs
    
    # Install dependencies
    log "Installing package and dependencies..."
    pip install --no-cache-dir -e .
    
    # Run database migrations
    run_migrations
    
    # Verify CA certificate files
    if [ ! -f "$CA_CERT_PATH" ] || [ ! -f "$CA_KEY_PATH" ]; then
        log "ERROR: CA certificate files not found"
        log "Expected paths:"
        log "  CA_CERT_PATH: $CA_CERT_PATH"
        log "  CA_KEY_PATH: $CA_KEY_PATH"
        exit 1
    fi
}

# Start proxy server
start_proxy() {
    log "Starting proxy server..."
    log "Current directory: $(pwd)"
    log "Python path: $PYTHONPATH"
    
    # Start the proxy server
    PYTHONPATH=/app python -m proxy.server.start
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    pkill -f "proxy.server.start" || true
}

# Register cleanup handler
trap cleanup EXIT

# Main execution
case "$1" in
    "setup")
        verify_tools
        setup_proxy_env
        ;;
    "start")
        verify_tools
        setup_proxy_env
        start_proxy
        ;;
    *)
        log "Usage: $0 {setup|start}"
        log "  setup  - Set up proxy environment"
        log "  start  - Start proxy server with API"
        exit 1
        ;;
esac 