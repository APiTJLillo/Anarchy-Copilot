#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "Error: Docker is not running or you don't have permission to use Docker"
        echo "Please start Docker and ensure you have the necessary permissions"
        exit 1
    fi
}

# Function to display a spinner while waiting
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Function to ensure certificates directory exists
setup_certs_directory() {
    mkdir -p ./certs
    if [ ! -f ./certs/ca.crt ]; then
        echo "🔐 CA certificate will be generated on first proxy start"
    fi
}

# Function to wait for service readiness
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=30
    local attempt=1

    echo "⏳ Waiting for $service to be ready..."
    while ! nc -z $host $port >/dev/null 2>&1; do
        if [ $attempt -eq $max_attempts ]; then
            echo "❌ $service failed to start after $max_attempts attempts"
            return 1
        fi
        sleep 2
        let attempt++
        echo -n "."
    done
    echo "✅ $service is ready"
    return 0
}

# Main script
echo "🚀 Starting Anarchy Copilot..."

# Check if Docker is running
check_docker

# Create certificates directory
setup_certs_directory

# Build and start the containers
echo "🔨 Building and starting containers..."
docker-compose up --build -d

# Wait for core services to be ready
wait_for_service localhost 8000 "Backend API" || exit 1

# Verify all services are running
if ! docker-compose ps | grep -q "Up"; then
    echo "❌ Error: Some services failed to start"
    echo "Logs from containers:"
    docker-compose logs
    exit 1
fi

echo "✅ All services are up and running!"

# Start the proxy client after services are ready
echo "🔄 Starting proxy client..."
python3 scripts/proxy_client.py &

echo
echo "🌐 You can access the application at:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   Proxy Server: http://localhost:8080"
echo
echo "🔧 Proxy Setup Instructions:"
echo "   1. Import the CA certificate from ./certs/ca.crt into your browser"
echo "   2. Configure your browser's proxy settings:"
echo "      - Host: localhost"
echo "      - Port: 8080"
echo "   3. Visit the web interface and start the proxy from the dashboard"
echo
echo "📝 To view logs, use: docker-compose logs -f"
echo "⏹️  To stop the application, use: docker-compose down"
