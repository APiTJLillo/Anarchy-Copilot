#!/bin/bash

# Function to check if proxy is ready
check_proxy_health() {
    for i in {1..30}; do
        if nc -z localhost 8083; then
            echo "Proxy is up and listening on port 8083"
            return 0
        fi
        echo "Waiting for proxy to be ready... (attempt $i/30)"
        sleep 1
    done
    echo "Proxy failed to start within 30 seconds"
    return 1
}

# Wait for proxy to be healthy
check_proxy_health || exit 1

echo "Running test request to Google..."
# Run curl with verbose output and include timing
curl -v -x localhost:8083 https://www.google.com/ -w "\nTime taken: %{time_total}s\n" 