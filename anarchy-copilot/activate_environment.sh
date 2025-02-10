#!/bin/bash

# Source the virtual environment
source venv/bin/activate

# Ensure Go binaries are in PATH
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin
export GO111MODULE=on

# Display current environment status
echo -e "\033[0;32mEnvironment activated:\033[0m"
echo "- Python virtual environment: active"
echo "- Go tools path: $GOPATH/bin"
echo -e "\nYou can now run Anarchy-Copilot components"
