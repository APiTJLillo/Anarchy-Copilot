#!/bin/bash
set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Check if health endpoint is responding
# Use curl with a timeout and retry options
response=$(curl -s -f --max-time 5 --retry 3 --retry-delay 1 http://localhost:8000/api/health 2>&1)
status=$?

if [ $status -eq 0 ]; then
    if echo "$response" | grep -q "healthy"; then
        log "Health check passed"
        exit 0
    else
        log "Health check failed: unexpected response"
        log "Response: $response"
        exit 1
    fi
else
    log "Health check failed: service not responding"
    log "Error: $response"
    exit 1
fi
