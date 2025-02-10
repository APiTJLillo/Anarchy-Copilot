#!/bin/bash

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Store the original directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}Starting Anarchy-Copilot Installation${NC}\n"

# Check if running as root for system package installation
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root or with sudo${NC}"
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get the real user who ran sudo
REAL_USER=${SUDO_USER:-$USER}
USER_HOME=$(eval echo ~$REAL_USER)

echo -e "${GREEN}Installing system dependencies...${NC}"
# Read system packages from dependencies.txt (only those with version numbers)
SYSTEM_DEPS=$(grep -E "^[a-z][^|]+:[0-9]" dependencies.txt | cut -d':' -f1 | tr -d ' ')
echo "System dependencies to install: $SYSTEM_DEPS"
apt update
for dep in $SYSTEM_DEPS; do
    echo -e "${YELLOW}Installing $dep...${NC}"
    apt install -y "$dep"
done

# Set up Go environment
echo -e "\n${GREEN}Setting up Go environment...${NC}"
if ! command_exists go; then
    echo -e "${RED}Go installation failed!${NC}"
    exit 1
fi

# Set up GOPATH for the real user
GOPATH="$USER_HOME/go"
mkdir -p "$GOPATH/bin"
chown -R $REAL_USER:$REAL_USER "$GOPATH"

# Add Go environment to user's bashrc if not already present
BASHRC="$USER_HOME/.bashrc"
if ! grep -q "GOPATH=" "$BASHRC"; then
    echo 'export GOPATH=$HOME/go' >> "$BASHRC"
    echo 'export PATH=$PATH:$GOPATH/bin' >> "$BASHRC"
    echo 'export GO111MODULE=on' >> "$BASHRC"
fi

# Set environment variables for the current session
export GOPATH="$GOPATH"
export PATH="$PATH:$GOPATH/bin"
export GO111MODULE=on

# Install Git repositories and build from source
echo -e "\n${GREEN}Installing tools from Git repositories...${NC}"
TEMP_DIR=$(mktemp -d)
chown $REAL_USER:$REAL_USER $TEMP_DIR

# Process Git repositories
while IFS='|' read -r line || [ -n "$line" ]; do
    # Skip comments, empty lines, and system packages
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[[:space:]]*[a-z][^|]+:[0-9] ]] && continue
    
    # Parse the line (trim whitespace and remove comments)
    name=$(echo "$line" | cut -d'|' -f1 | tr -d ' ')
    url=$(echo "$line" | cut -d'|' -f2 | tr -d ' ')
    type=$(echo "$line" | cut -d'|' -f3 | tr -d ' ')
    path=$(echo "$line" | cut -d'|' -f4 | cut -d'#' -f1 | tr -d ' ')
    
    # Skip if any required field is empty
    [[ -z "$name" || -z "$url" || -z "$type" ]] && continue
    [[ "$url" != "http"* ]] && continue
    
    echo -e "\n${YELLOW}Processing $name from $url (type: $type)${NC}"
    
    cd "$TEMP_DIR"
    echo -e "${YELLOW}Cloning repository...${NC}"
    sudo -u $REAL_USER git clone "$url" "$name"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to clone $url${NC}"
        continue
    fi
    
    cd "$name"
    
    case "$type" in
        "go")
            echo -e "${YELLOW}Building Go project $name...${NC}"
            if [ -f "go.mod" ]; then
                echo "Using existing go.mod"
                sudo -u $REAL_USER go mod tidy
            else
                echo "Initializing go module"
                sudo -u $REAL_USER go mod init "$name"
                sudo -u $REAL_USER go mod tidy
            fi

            # Change to the directory containing main package
            if [ -n "$path" ] && [ -d "$path" ]; then
                cd "$path"
            fi

            echo -e "${YELLOW}Installing from directory: $(pwd)${NC}"
            sudo -u $REAL_USER env "PATH=$PATH" "GOPATH=$GOPATH" "GO111MODULE=on" go install ./...
            ;;
        *)
            echo -e "${RED}Unknown type: $type${NC}"
            ;;
    esac
done < "$SCRIPT_DIR/dependencies.txt"

# Cleanup
cd "$SCRIPT_DIR"
rm -rf "$TEMP_DIR"

# Create and set up Python virtual environment
echo -e "\n${GREEN}Setting up Python virtual environment...${NC}"
sudo -u $REAL_USER python3 -m venv venv
VENV_ACTIVATE="./venv/bin/activate"
source $VENV_ACTIVATE

echo -e "${GREEN}Installing Python dependencies...${NC}"
sudo -u $REAL_USER ./venv/bin/pip install --upgrade pip
sudo -u $REAL_USER ./venv/bin/pip install -r requirements.txt

# Set up frontend
echo -e "\n${GREEN}Setting up frontend dependencies...${NC}"
cd frontend
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    sudo -u $REAL_USER npm install
fi

# Return to original directory
cd "$SCRIPT_DIR"

# Create initial database
echo -e "\n${GREEN}Setting up database...${NC}"
sudo -u $REAL_USER ./venv/bin/python3 -c "
from database import Base, engine
from models import User, Project, ReconResult
Base.metadata.create_all(bind=engine)
"

# Create the activation script
echo -e "\n${GREEN}Creating environment activation script...${NC}"
cat > activate_environment.sh << 'EOL'
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
EOL

chmod +x activate_environment.sh
chown $REAL_USER:$REAL_USER activate_environment.sh

echo -e "\n${GREEN}Installation complete!${NC}"
echo -e "${YELLOW}Please run the following commands to set up your environment:${NC}"
echo -e "1. ${GREEN}source ./activate_environment.sh${NC}"
echo -e "\n${YELLOW}You can verify the installation by running:${NC}"
echo -e "${GREEN}./check_installation.py${NC}"
