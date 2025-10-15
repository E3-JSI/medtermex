#!/bin/bash
# setup_env.sh - Setup the development environment

set -e  # Exit immediately if a command exits with a non-zero status

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up development environment...${NC}"

# Define Python version from params or use default
VENV_NAME=".venv"

# Create virtual environment
echo -e "${GREEN}Creating virtual environment...${NC}"
if [ ! -d "./$VENV_NAME" ]; then
    if command -v uv &> /dev/null; then
        echo -e "${GREEN}Using uv for virtual environment creation...${NC}"
        uv venv $VENV_NAME --python 3.12
    else
        echo -e "${YELLOW}uv not found, falling back to python...${NC}"
        if ! command -v python3.12 &> /dev/null; then
            echo -e "${RED}Python 3.12 is not installed. Please install Python 3.12 and try again.${RED}"
            exit 1
        fi
        python3.12 -m venv ./$VENV_NAME
    fi
    echo "Virtual environment created at $VENV_NAME"
else
    echo "Virtual environment already exists at $VENV_NAME"
fi

# Activate virtual environment depending on the OS
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    source ./$VENV_NAME/bin/activate
elif [[ "$OSTYPE" == "msys" ]]; then
    source ./$VENV_NAME/Scripts/activate
fi

# Check if uv is available
if command -v uv &> /dev/null; then
    echo -e "${GREEN}Using uv for package management...${NC}"
    # Install project in development mode using uv
    uv pip install -e .[dev]
else
    echo -e "${YELLOW}uv not found, falling back to pip...${NC}"
    # Upgrade pip
    echo -e "${GREEN}Upgrading pip...${NC}"
    pip install --upgrade pip
    # Install project in development mode using pip
    pip install -e .[dev]
fi

# Create necessary directories if they don't exist
echo -e "${GREEN}Data directories...${NC}"
echo -e "${YELLOW}!!! Ask the HPC staff for creating the project directories !!!${NC}"

# Create necessary directories if they don't exist
echo -e "${GREEN}Creating project directories...${NC}"
mkdir -p data/raw data/interim data/final data/external

# Success message
echo -e "${GREEN}Environment setup complete!${NC}"
echo "To activate the virtual environment, run:"
echo -e "  ${YELLOW}source ./$VENV_NAME/bin/activate${NC} (Linux/macOS)"
echo -e "  ${YELLOW}./$VENV_NAME/Scripts/activate${NC} (Windows)"

# Deactivate virtual environment
deactivate
