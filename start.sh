#!/bin/bash

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

# Main script
echo "🚀 Starting Anarchy Copilot..."

# Check if Docker is running
check_docker

# Create certificates directory
setup_certs_directory

# Build and start the containers
echo "🔨 Building and starting containers..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10  # Give services some time to start

# Check if services are running
if ! docker-compose ps | grep -q "Up"; then
    echo "❌ Error: Services failed to start"
    echo "Logs from containers:"
    docker-compose logs
    exit 1
fi

echo "✅ Services are up and running!"
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
