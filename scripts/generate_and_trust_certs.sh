#!/bin/bash

CERT_DIR="./certs"
CERT_FILE="$CERT_DIR/ca.crt"
KEY_FILE="$CERT_DIR/ca.key"

# Create certs directory if it doesn't exist
mkdir -p $CERT_DIR

# Generate certificate and key if they don't exist
if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
    echo "Generating self-signed certificate and key..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout $KEY_FILE -out $CERT_FILE -subj "/CN=localhost"
fi

# Add certificate to trusted certificates
if [ -f "$CERT_FILE" ]; then
    echo "Adding certificate to trusted certificates..."
    sudo cp $CERT_FILE /usr/local/share/ca-certificates/
    sudo update-ca-certificates
fi

echo "Certificate setup complete."
