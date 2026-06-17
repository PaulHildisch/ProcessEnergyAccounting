#!/bin/bash
set -e

# Installs system dependencies and builds BCC from source.
# If a Python virtual environment is active, the Python BCC bindings are installed into that environment.

PROJECT_ROOT="$(pwd)"
BCC_BASE_DIR="$(mktemp -d /tmp/bcc-src.XXXXXX)"
BCC_SRC_DIR="${BCC_BASE_DIR}/bcc-src"
BCC_BUILD_DIR="${BCC_SRC_DIR}/build"
BCC_PY_BINDINGS_DIR="${BCC_BUILD_DIR}/src/python/bcc-python3"

log() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

error() {
  echo "[ERROR] $*" >&2
  exit 1
}

# Make sure docker and docker-compose are installed
if ! command -v docker >/dev/null 2>&1; then
  log "Docker not found. Installing Docker..."
  sudo apt-get update
  sudo apt-get install -y docker.io
  sudo systemctl enable --now docker
else
  log "Docker is already installed."
fi

if ! command -v docker-compose >/dev/null 2>&1; then
  log "docker-compose not found. Installing docker-compose..."
  sudo apt-get install -y docker-compose
else
  log "docker-compose is already installed."
fi

# Helpful guidance for non-root Docker usage
if getent group docker >/dev/null 2>&1; then
  if id -nG "$USER" | grep -qw docker; then
    log "User '$USER' is already in the docker group."
  else
    warn "User '$USER' is not in the docker group."
    warn "To avoid Docker permission errors, run: sudo usermod -aG docker $USER"
    warn "Then log out and back in before using Docker without sudo."
  fi
else
  warn "docker group not found. It should normally be created by the Docker package."
fi

build_bcc() {
  log "Using temporary BCC source directory at $BCC_BASE_DIR"
  log "Cloning BCC into $BCC_SRC_DIR"
  git clone https://github.com/iovisor/bcc.git "$BCC_SRC_DIR"

  mkdir -p "$BCC_BUILD_DIR"
  (
    cd "$BCC_BUILD_DIR"
    log "Configuring BCC core build"
    cmake ..
    log "Building BCC core"
    make
    log "Installing BCC core system components"
    sudo make install
    log "Restoring ownership of the BCC source tree after sudo install"
    sudo chown -R "$USER":"$(id -gn)" "$BCC_SRC_DIR"
    log "Configuring Python 3 bindings"
    cmake -DPYTHON_CMD=python3 ..
  )

  if [[ ! -d "$BCC_PY_BINDINGS_DIR" ]]; then
    error "Expected Python bindings directory not found: $BCC_PY_BINDINGS_DIR"
  fi
}

install_bcc_into_active_env() {
  local env_python=""
  local env_pip=""
  local env_desc=""

  if [[ -n "$VIRTUAL_ENV" ]]; then
    env_python="$VIRTUAL_ENV/bin/python"
    env_pip="$VIRTUAL_ENV/bin/pip"
    env_desc="virtual environment at $VIRTUAL_ENV"
  elif [[ -d "${PROJECT_ROOT}/.venv" ]]; then
    env_python="${PROJECT_ROOT}/.venv/bin/python"
    env_pip="${PROJECT_ROOT}/.venv/bin/pip"
    env_desc="project virtual environment at ${PROJECT_ROOT}/.venv"
  else
    warn "No active virtual environment detected and no project .venv found."
    warn "BCC core was built in $BCC_SRC_DIR, but Python bindings were not installed into a project environment."
    warn "Activate your venv and run: cd $BCC_PY_BINDINGS_DIR && pip install ."
    return 0
  fi

  if [[ ! -x "$env_python" || ! -x "$env_pip" ]]; then
    error "Could not find python/pip for $env_desc"
  fi

  log "Installing Python BCC bindings into $env_desc"
  (
    cd "$BCC_PY_BINDINGS_DIR"
    "$env_pip" install .
  )

  if "$env_python" -c "from bcc import BPF; import bcc; print('SUCCESS: bcc imported from', bcc.__file__)"; then
    log "BCC Python bindings installed successfully into $env_desc"
  else
    error "BCC Python bindings were installed, but 'from bcc import BPF' still failed in $env_desc"
  fi
}

if python3 -c "from bcc import BPF" >/dev/null 2>&1; then
  log "Python BCC is already importable system-wide."
else
  log "Python BCC is not importable system-wide. Building BCC from source."
fi

cleanup() {
  if [[ -d "$BCC_BASE_DIR" ]]; then
    log "Cleaning up temporary BCC source directory at $BCC_BASE_DIR"
    rm -rf "$BCC_BASE_DIR"
  fi
}

trap cleanup EXIT

build_bcc
install_bcc_into_active_env
