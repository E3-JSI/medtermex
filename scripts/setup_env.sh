#!/bin/bash
# setup_env.sh - Setup the development environment

set -e  # Exit immediately if a command exits with a non-zero status

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Read Python version from .python-version file
if [ -f "./.python-version" ]; then
    PYTHON_VERSION=$(cat ./.python-version 2>/dev/null | tr -d '\n\r' | xargs)
    # Validate format is X.Y (major.minor only)
    if [[ ! $PYTHON_VERSION =~ ^[0-9]+\.[0-9]+$ ]]; then
        echo -e "${YELLOW}Invalid Python version format in .python-version: '$PYTHON_VERSION'."
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
        echo -e "${YELLOW}Using default Python version $PYTHON_VERSION${NC}"
    fi
else
    echo -e "${YELLOW}.python-version file not found."
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    echo -e "${YELLOW}Using default Python version $PYTHON_VERSION${NC}"
fi

echo -e "${GREEN}Setting up development environment...${NC}"

# Define Python version from params or use default
VENV_NAME=".venv"

# Create virtual environment
echo -e "${GREEN}Creating virtual environment...${NC}"
if [ ! -d "./$VENV_NAME" ]; then
    VENV_NEEDS_CREATION=true
else
    # Check if existing venv matches the specified Python version
    if [ -f "./$VENV_NAME/bin/python" ]; then
        VENV_PYTHON_VERSION=$("./$VENV_NAME/bin/python" --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
        if [ "$VENV_PYTHON_VERSION" != "$PYTHON_VERSION" ]; then
            echo -e "${YELLOW}Virtual environment uses Python $VENV_PYTHON_VERSION but Python $PYTHON_VERSION is required. Reinitializing...${NC}"
            rm -rf "./$VENV_NAME"
            VENV_NEEDS_CREATION=true
        fi
    else
        echo -e "${YELLOW}Virtual environment is corrupted. Reinitializing...${NC}"
        rm -rf "./$VENV_NAME"
        VENV_NEEDS_CREATION=true
    fi
fi

if [ "$VENV_NEEDS_CREATION" = true ]; then
    if command -v uv &> /dev/null; then
        echo -e "${GREEN}Using uv for virtual environment creation...${NC}"
        uv venv $VENV_NAME --python $PYTHON_VERSION
    else
        echo -e "${YELLOW}uv not found, falling back to python...${NC}"
        if ! command -v python${PYTHON_VERSION} &> /dev/null; then
            echo -e "${RED}Python ${PYTHON_VERSION} is not installed. Please install Python ${PYTHON_VERSION} and try again.${RED}"
            exit 1
        fi
        python${PYTHON_VERSION} -m venv ./$VENV_NAME
    fi
    echo "Virtual environment created at $VENV_NAME"
else
    echo "Virtual environment already exists at $VENV_NAME with Python $PYTHON_VERSION"
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
    uv pip install -e .
else
    echo -e "${YELLOW}uv not found, falling back to pip...${NC}"
    # Upgrade pip
    echo -e "${GREEN}Upgrading pip...${NC}"
    pip install --upgrade pip
    # Install project in development mode using pip
    pip install -e .
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
