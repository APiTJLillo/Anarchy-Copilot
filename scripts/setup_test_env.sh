#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up test environment...${NC}"

# Create certificates directory if it doesn't exist
mkdir -p certs

# Generate CA certificate if it doesn't exist
if [ ! -f certs/ca.crt ] || [ ! -f certs/ca.key ]; then
    echo -e "${YELLOW}Generating CA certificates...${NC}"
    openssl req -x509 -new -nodes \
        -keyout certs/ca.key \
        -out certs/ca.crt \
        -days 365 \
        -subj "/CN=Anarchy Copilot CA" \
        -addext "basicConstraints=critical,CA:true" \
        -addext "keyUsage=critical,keyCertSign,cRLSign"

    # Set permissions
    chmod 600 certs/ca.key
    chmod 644 certs/ca.crt
fi

# Check if we're running on Linux
if [ "$(uname)" == "Linux" ]; then
    echo -e "${YELLOW}Detected Linux system${NC}"
    
    # Check for common certificate directories
    CERT_DIRS=(
        "/usr/local/share/ca-certificates"
        "/etc/ca-certificates/trust-source/anchors"
        "/etc/pki/ca-trust/source/anchors"
    )
    
    CERT_INSTALLED=0
    for dir in "${CERT_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            echo -e "${YELLOW}Installing CA certificate to $dir${NC}"
            sudo cp certs/ca.crt "$dir/anarchy-copilot-ca.crt"
            CERT_INSTALLED=1
            break
        fi
    done
    
    if [ $CERT_INSTALLED -eq 1 ]; then
        # Update CA trust store
        if command -v update-ca-certificates &> /dev/null; then
            sudo update-ca-certificates
        elif command -v update-ca-trust &> /dev/null; then
            sudo update-ca-trust
        else
            echo -e "${RED}Could not find command to update CA trust store${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Could not find suitable directory to install CA certificate${NC}"
        exit 1
    fi

# Check if we're running on macOS
elif [ "$(uname)" == "Darwin" ]; then
    echo -e "${YELLOW}Detected macOS system${NC}"
    
    # Add to Keychain
    echo -e "${YELLOW}Installing CA certificate to Keychain...${NC}"
    sudo security add-trusted-cert -d -r trustRoot \
        -k "/Library/Keychains/System.keychain" \
        certs/ca.crt

else
    echo -e "${RED}Unsupported operating system: $(uname)${NC}"
    exit 1
fi

# Install Python dependencies
echo -e "${YELLOW}Installing Python test dependencies...${NC}"
pip install -r tests/requirements-test.txt
pip install -r tests/requirements-proxy-test.txt

# Set environment variables
export PROXY_HOST=0.0.0.0
export PROXY_PORT=8080
export CA_CERT_PATH=./certs/ca.crt
export CA_KEY_PATH=./certs/ca.key

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}To run tests:${NC}"
echo -e "  1. Start the proxy server in one terminal:"
echo -e "     ${GREEN}python -m proxy.core${NC}"
echo -e "  2. Run the tests in another terminal:"
echo -e "     ${GREEN}python scripts/run_proxy_tests.py${NC}"
echo -e "  Or run specific tests:"
echo -e "     ${GREEN}pytest tests/test_proxy_connections.py -v${NC}"
