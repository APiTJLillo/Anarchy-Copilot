#!/bin/bash
set -e

# Development environment setup script

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
    
    # Check optional tools
    if ! command_exists nuclei; then
        log "WARNING: Nuclei not found, some features may be limited"
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

# Create test data
create_test_data() {
    log "Creating test user and project..."
    python scripts/create_test_user.py
}

# Setup development environment
setup_dev_env() {
    log "Setting up development environment..."
    
    # Create necessary directories
    mkdir -p /app/logs
    mkdir -p /app/data
    mkdir -p /app/tests/data/output
    
    # Install pre-commit hooks if git repo exists and pre-commit is available
    if [ -d ".git" ] && command_exists pre-commit; then
        log "Installing pre-commit hooks..."
        pre-commit install || { log "Pre-commit hooks installation skipped"; true; }
    else
        log "Skipping git hooks setup (git not available or not in a repo)"
    fi
    
    # Install dependencies - first install the package to get pyproject.toml configuration
    log "Installing package and dependencies..."
    pip install --no-cache-dir -e .
    log "Installing development dependencies..."
    pip install --no-cache-dir -e ".[dev]"
    log "Installing test dependencies..."
    pip install --no-cache-dir -r tests/requirements-test.txt
    
    # Verify all dependencies are installed correctly
    log "Verifying dependencies..."
    if ! pip check; then
        log "WARNING: Dependency conflicts detected"
    fi
    
    # Create package symlinks
    log "Setting up package symlinks..."
    SITE_PACKAGES="/usr/local/lib/python3.10/site-packages"
    if [ -d "$SITE_PACKAGES" ]; then
        cd "$SITE_PACKAGES"
        for module in recon_module vuln_module anarchy_copilot; do
            if [ -d "/app/$module" ]; then
                ln -sf "/app/$module" .
            else
                log "WARNING: Module directory /app/$module not found, skipping symlink"
            fi
        done
        cd /app
    else
        log "WARNING: Python site-packages directory not found, skipping symlinks"
    fi
    
    # Update Nuclei templates if available
    if command_exists nuclei; then
        log "Updating Nuclei templates..."
        nuclei -update-templates || log "WARNING: Failed to update Nuclei templates"
    fi

    # Run database migrations
    run_migrations

    # Create test data
    create_test_data
}

# Setup debugger
setup_debugger() {
    log "Configuring debugger..."
    
    # Create VSCode debugging configuration if it doesn't exist
    if [ ! -f ".vscode/launch.json" ]; then
        mkdir -p .vscode
        cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Remote Attach",
            "type": "python",
            "request": "attach",
            "host": "localhost",
            "port": 5678,
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "/app"
                }
            ]
        }
    ]
}
EOF
    fi
}

# Wait for the API to be ready
wait_for_api() {
    log "Waiting for API to be ready..."
    local retries=30
    local wait_seconds=1
    
    while [ $retries -gt 0 ]; do
        if curl -s -f "http://localhost:8000/api/health" > /dev/null 2>&1; then
            log "API is ready!"
            return 0
        fi
        retries=$((retries-1))
        log "API not ready yet, waiting... ($retries attempts left)"
        sleep $wait_seconds
    done
    
    log "ERROR: API failed to start"
    return 1
}

# Start development services
start_services() {
    log "Starting development services..."
    
    # Start API server with debugger
    log "Starting API server with debugger on port 5678..."
    log "Current directory: $(pwd)"
    log "Python path: $PYTHONPATH"
    log "Available Python packages:"
    pip list
    
    # Start the FastAPI application
    python -m debugpy --listen 0.0.0.0:5678 \
        -m uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --log-level debug \
        --proxy-headers \
        --timeout-keep-alive 75 &
    API_PID=$!

    # Start the proxy in the background
    python -m proxy.server &
    PROXY_PID=$!

    # Run the test script in the background after a short delay
    (sleep 5 && /usr/local/bin/test-proxy.sh) &
    TEST_PID=$!

    # Wait for the proxy process
    wait $PROXY_PID

    # Wait for the API process
    wait $API_PID
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
        setup_debugger
        ;;
    "start")
        verify_tools
        setup_dev_env  # Always run setup to ensure symlinks
        start_services
        ;;
    "debug")
        verify_tools
        setup_dev_env  # Always run setup to ensure symlinks
        setup_debugger
        start_services
        ;;
    *)
        log "Usage: $0 {setup|start|debug}"
        log "  setup  - Set up development environment"
        log "  start  - Start services"
        log "  debug  - Start services with debugger"
        exit 1
        ;;
esac
