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
    
    # Check Python
    if ! command_exists python; then
        log "ERROR: Python not found"
        exit 1
    fi
    
    # Check Nuclei
    if ! command_exists nuclei; then
        log "ERROR: Nuclei not found"
        exit 1
    fi
    
    # Check pip
    if ! command_exists pip; then
        log "ERROR: pip not found"
        exit 1
    }

    log "All required tools are available"
}

# Setup development environment
setup_dev_env() {
    log "Setting up development environment..."
    
    # Create necessary directories
    mkdir -p /app/logs
    mkdir -p /app/data
    mkdir -p /app/tests/data/output
    
    # Install pre-commit hooks if git repo exists
    if [ -d ".git" ]; then
        log "Installing pre-commit hooks..."
        pre-commit install
    fi
    
    # Update pip tools
    log "Updating development dependencies..."
    pip-compile --upgrade requirements.txt
    pip-compile --upgrade tests/requirements-test.txt
    
    # Install dependencies
    log "Installing package in development mode..."
    pip install -e ".[dev]"
    pip install -r tests/requirements-test.txt
    
    # Create package symlinks
    log "Setting up package symlinks..."
    cd /usr/local/lib/python3.10/site-packages/
    ln -sf /app/recon_module .
    ln -sf /app/vuln_module .
    ln -sf /app/anarchy_copilot .
    cd /app
    
    # Update Nuclei templates
    log "Updating Nuclei templates..."
    nuclei -update-templates
}

# Setup debugger
setup_debugger() {
    log "Configuring debugger..."
    
    # Create VSCode debugging configuration if it doesn't exist
    if [ ! -f ".vscode/launch.json" ]; then
        mkdir -p .vscode
        cat > .vscode/launch.json <<EOF
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
                    "localRoot": "\${workspaceFolder}",
                    "remoteRoot": "/app"
                }
            ]
        }
    ]
}
EOF
    fi
}

# Start development services
start_services() {
    log "Starting development services..."
    
    # Start API server with debugger
    log "Starting API server with debugger on port 5678..."
    python -m debugpy --listen 0.0.0.0:5678 \
        -m uvicorn anarchy_copilot.api:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --log-level debug
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    # Add cleanup tasks here
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
