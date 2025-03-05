#!/bin/bash
cd "$(dirname "$0")"

# Make scripts executable
chmod +x install.sh install_deps.sh install-test-deps.sh

# Install main dependencies
./install.sh

# Install Redux and other dependencies
./install_deps.sh

# Install test dependencies
./install-test-deps.sh

# Run TypeScript checks
npm run typecheck

echo "Setup complete! You can now start the development server with 'npm start'"
