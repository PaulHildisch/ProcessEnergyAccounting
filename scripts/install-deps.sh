#!/bin/bash
set -e

# Use the user-setup script to get the user env on the remote where the experiments will run.

# Make sure docker and docker-compose are installed
if ! command -v docker &> /dev/null
then
    echo "Docker not found. Installing Docker..."
    # Install Docker (Ubuntu/Debian example)
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo systemctl enable --now docker
else
    echo "Docker is already installed."
fi

if ! command -v docker-compose &> /dev/null
then
    echo "docker-compose not found. Installing docker-compose..."
    sudo apt-get install -y docker-compose
else
    echo "docker-compose is already installed."
fi

# Install java & nextflow using sdkman
if ! command -v sdk &> /dev/null
then
    echo "SDKMAN not found. Installing SDKMAN..."
    curl -s "https://get.sdkman.io" | bash
    source "$HOME/.sdkman/bin/sdkman-init.sh"
else
    echo "SDKMAN is already installed."
    source "$HOME/.sdkman/bin/sdkman-init.sh"
fi

echo "Installing Java (OpenJDK 17) via SDKMAN..."
sdk install java 17.0.10-tem
sdk default java 17.0.10-tem

# Remove a directory named 'nextflow' if it exists, to allow the installer to create the executable
if [ -d "nextflow" ]; then
    echo "Removing existing 'nextflow' directory to allow installation."
    rm -rf nextflow
fi

echo "Installing Nextflow..."
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/
sudo chmod +x /usr/local/bin/nextflow

# Install python BCC and the necessary build tools to install bcc from scratch
sudo apt-get update

# Try to install python3-distutils if available, otherwise skip
if apt-cache show python3-distutils 2>/dev/null | grep -q 'Package:'; then
    PY_DISTUTILS="python3-distutils"
else
    PY_DISTUTILS=""
fi

sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libelf-dev \
    zlib1g-dev \
    libfl-dev \
    bison \
    flex \
    libclang-14-dev \
    clang-14 \
    llvm-14-dev \
    libpolly-14-dev \
    llvm-14-tools \
    libssl-dev \
    python3 \
    python3-pip \
    python3-venv \
    linux-headers-$(uname -r) \
    $PY_DISTUTILS

# Always check for active venv or Poetry and install BCC Python bindings there if needed

if [ -n "$VIRTUAL_ENV" ] || [ -n "$POETRY_ACTIVE" ]; then
    if [ -n "$VIRTUAL_ENV" ]; then
        VENV_PYTHON="python"
        VENV_PIP="pip"
        ENV_DESC="Python virtual environment at $VIRTUAL_ENV"
    elif [ -n "$POETRY_ACTIVE" ]; then
        VENV_PYTHON="python"
        VENV_PIP="pip"
        ENV_DESC="Poetry environment"
    fi

    echo "Detected active $ENV_DESC"
    if $VENV_PYTHON -c "import bcc" &> /dev/null; then
        echo "Python BCC is already installed and importable in the active environment. Skipping BCC build."
    else
        # Remove existing bcc directory with sudo to avoid permission issues
        if [ -d "bcc" ]; then
            echo "Removing existing bcc directory..."
            sudo rm -rf bcc
        fi

        git clone https://github.com/iovisor/bcc.git
        mkdir bcc/build; cd bcc/build
        cmake ..
        make
        sudo make install
        cmake -DPYTHON_CMD=python3 .. # build python3 binding

        if [ -d src/python/bcc-python3 ]; then
            pushd src/python/bcc-python3
            # Try pip install first
            $VENV_PIP install .
            if $VENV_PYTHON -c "import bcc; print('SUCCESS: bcc imported from', bcc.__file__)"; then
                echo "BCC Python bindings installed via pip."
            else
                echo "ERROR: Could not import bcc in the active environment after pip install."
                echo "Would you like to manually copy the BCC Python package to your Poetry venv site-packages? (y/n)"
                read -r CONFIRM_BCC_COPY
                if [ "$CONFIRM_BCC_COPY" = "y" ]; then
                    PROJECT_DIR="$(pwd)"
                    echo "Switching to project directory: $PROJECT_DIR"
                    VENV=$($VENV_PYTHON -c "import sys; print(sys.prefix)")
                    cp -r ~/bcc/build/src/python/bcc-python3/bcc "$VENV/lib/python3.12/site-packages/"
                    $VENV_PYTHON -c "import bcc; print('SUCCESS: bcc imported from', bcc.__file__)" \
                        || echo "ERROR: Manual copy failed. Please check permissions and paths."
                else
                    echo "Skipping manual copy of BCC Python package."
                fi
            fi
            popd
        else
            echo "src/python/bcc-python3 directory not found, skipping python BCC pip install."
        fi
    fi
else
    # No active venv or Poetry: check system install
    if python3 -c "import bcc" &> /dev/null; then
        echo "Python BCC is already installed and importable system-wide. Skipping BCC build."
    else
        # Remove existing bcc directory with sudo to avoid permission issues
        if [ -d "bcc" ]; then
            echo "Removing existing bcc directory..."
            sudo rm -rf bcc
        fi

        git clone https://github.com/iovisor/bcc.git
        mkdir bcc/build; cd bcc/build
        cmake ..
        make
        sudo make install
        cmake -DPYTHON_CMD=python3 .. # build python3 binding

        if [ -d src/python/bcc-python3 ]; then
            pushd src/python/bcc-python3
            sudo $VENV_PIP install .
            python3 -c "import bcc; print('SUCCESS: bcc imported from', bcc.__file__)" \
                || echo "ERROR: Could not import bcc system-wide."
            popd
        else
            echo "src/python/bcc-python3 directory not found, skipping python BCC pip install."
        fi
    fi
fi
