#!/bin/bash

# clean-py-env.sh
# Robust, extensible Python environment setup script
# Author: (your name)
# Usage: ./clean-py-env.sh [--project-path <path>] [--tool <venv|poetry|uv|pipx|pixi|auto>] [--system-python] [--force] [--clean] [--verbose] [--quiet] [--dry-run] [--help]

set -e

# --- Configurable defaults ---
DEFAULT_TOOL="auto"
DEFAULT_PROJECT_PATH="."
LOGFILE=""
VERBOSE=1
QUIET=0
DRYRUN=0
FORCE=0
CLEAN=0
SYSTEM_PYTHON=0
PYTHON_VERSION=""
TOOL="$DEFAULT_TOOL"
PROJECT_PATH="$DEFAULT_PROJECT_PATH"

# --- Color output ---
c_bold="\033[1m"
c_red="\033[31m"
c_green="\033[32m"
c_yellow="\033[33m"
c_blue="\033[34m"
c_reset="\033[0m"

log() {
  [[ $QUIET -eq 1 ]] && return
  echo -e "${c_bold}${c_blue}[INFO]${c_reset} $*"
}

warn() {
  [[ $QUIET -eq 1 ]] && return
  echo -e "${c_bold}${c_yellow}[WARN]${c_reset} $*" >&2
}

error() {
  echo -e "${c_bold}${c_red}[ERROR]${c_reset} $*" >&2
  exit 1
}

verbose() {
  [[ $VERBOSE -eq 1 ]] && echo -e "${c_green}$*${c_reset}"
}

dryrun() {
  [[ $DRYRUN -eq 1 ]] && echo -e "${c_yellow}[DRYRUN]${c_reset} $*"
}

# --- Usage ---
usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --project-path <path>   Path to project (default: current directory)
  --tool <venv|poetry|uv|pipx|pixi|auto>
                         Tool to use for environment (default: auto-detect)
  --python <version>     Specify Python version (e.g., 3.11)
  --system-python        Use system Python (e.g., for bcc)
  --force                Force re-creation of environment
  --clean                Remove existing environment before creating
  --verbose              Verbose output (default)
  --quiet                Quiet output
  --dry-run              Show what would be done, but make no changes
  --help                 Show this help message

Examples:
  $0 --project-path ~/myproj
  $0 --tool poetry --force
  $0 --system-python --clean

EOF
  exit 0
}

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-path) PROJECT_PATH="$2"; shift 2 ;;
    --tool) TOOL="$2"; shift 2 ;;
    --python) PYTHON_VERSION="$2"; shift 2 ;;
    --system-python) SYSTEM_PYTHON=1; shift ;;
    --force) FORCE=1; shift ;;
    --clean) CLEAN=1; shift ;;
    --verbose) VERBOSE=1; QUIET=0; shift ;;
    --quiet) QUIET=1; VERBOSE=0; shift ;;
    --dry-run) DRYRUN=1; shift ;;
    --help) usage ;;
    *) error "Unknown argument: $1" ;;
  esac
done

# --- Path traversal to find project root ---
find_project_root() {
  local dir
  dir="$(cd "$1" && pwd)"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/pyproject.toml" || -f "$dir/requirements.txt" ]]; then
      echo "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

# --- Tool detection ---
check_tool() {
  command -v "$1" >/dev/null 2>&1
}

ensure_tool() {
  if ! check_tool "$1"; then
    warn "Required tool '$1' not found in PATH."
    read -rp "Do you want to install '$1'? [Y/n] " yn
    yn=${yn:-Y}
    if [[ "$yn" =~ ^[Yy]$ ]]; then
      case "$1" in
        poetry)
          curl -sSL https://install.python-poetry.org | python3 - ;;
        uv)
          pip install --user uv || pipx install uv || error "Failed to install uv." ;;
        pipx)
          pip install --user pipx || error "Failed to install pipx." ;;
        pixi)
          curl -fsSL https://pixi.sh/install.sh | bash || error "Failed to install pixi." ;;
        python3)
          sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip || error "Failed to install python3." ;;
        *)
          error "No install instructions for tool '$1'. Please install it manually."
          ;;
      esac
      # Refresh shell environment so new tools are available immediately
      export PATH="$PATH:$HOME/.local/bin"
      if [ -f "$HOME/.profile" ]; then . "$HOME/.profile"; fi
      if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi
      hash -r
      if ! check_tool "$1"; then
        error "Installation of '$1' failed or it is not in PATH."
      fi
    else
      error "Cannot proceed without required tool '$1'."
    fi
  fi
}

detect_env_tool() {
  local root="$1"
  if [[ -f "$root/pyproject.toml" ]]; then
    if grep -q '\[tool.poetry\]' "$root/pyproject.toml"; then
      echo "poetry"
      return
    elif grep -q '\[tool.pixi\]' "$root/pyproject.toml"; then
      echo "pixi"
      return
    elif grep -q '\[project\]' "$root/pyproject.toml"; then
      # PEP 621, could use uv or pip
      if check_tool uv; then
        echo "uv"
        return
      fi
      echo "venv"
      return
    fi
  fi
  if [[ -f "$root/requirements.txt" ]]; then
    if check_tool uv; then
      echo "uv"
      return
    fi
    echo "venv"
    return
  fi
  error "No recognizable Python project file found in $root"
}

# --- Python interpreter info ---
print_python_info() {
  log "System Python: $(command -v python3 2>/dev/null || echo 'not found')"
  python3 --version 2>/dev/null || true
  if command -v python >/dev/null 2>&1; then
    log "Default 'python': $(command -v python)"
    python --version 2>/dev/null || true
  fi
  if [[ -d "$1/venv" ]]; then
    log "Venv Python: $1/venv/bin/python"
    "$1/venv/bin/python" --version 2>/dev/null || true
  fi
  if [[ -d "$1/.venv" ]]; then
    log "Project .venv Python: $1/.venv/bin/python"
    "$1/.venv/bin/python" --version 2>/dev/null || true
  fi
}

# --- Clean up environment ---
clean_env() {
  local root="$1"
  log "Cleaning up existing environment in $root"
  [[ -d "$root/venv" ]] && rm -rf "$root/venv"
  [[ -d "$root/.venv" ]] && rm -rf "$root/.venv"
  [[ -d "$root/.pixi" ]] && rm -rf "$root/.pixi"
  [[ -d "$root/.tox" ]] && rm -rf "$root/.tox"
  [[ -d "$root/.mypy_cache" ]] && rm -rf "$root/.mypy_cache"
  [[ -f "$root/poetry.lock" && $FORCE -eq 1 ]] && rm -f "$root/poetry.lock"
}

# --- Environment creation functions ---
create_venv() {
  local root="$1"
  local python="python3"
  [[ -n "$PYTHON_VERSION" ]] && python="python$PYTHON_VERSION"
  ensure_tool "$python"
  local venv_args=""
  [[ $SYSTEM_PYTHON -eq 1 ]] && venv_args="--system-site-packages"
  log "Creating venv with $python $venv_args in $root/venv"
  [[ $DRYRUN -eq 1 ]] && dryrun "$python -m venv $venv_args venv" && return
  (cd "$root" && $python -m venv $venv_args venv)
  log "Activating venv and installing requirements (if any)..."
  if [[ -f "$root/requirements.txt" ]]; then
    (cd "$root" && source venv/bin/activate && pip install -U pip && pip install -r requirements.txt)
  elif [[ -f "$root/pyproject.toml" ]]; then
    (cd "$root" && source venv/bin/activate && pip install -U pip && pip install .)
  fi
}

select_python_interpreter() {
  local candidate=""
  local version_spec="${PYTHON_VERSION}"

  if [[ -n "$version_spec" ]]; then
    candidate="python${version_spec}"
    if ! command -v "$candidate" >/dev/null 2>&1; then
      error "Requested interpreter '$candidate' was not found in PATH."
    fi
    echo "$candidate"
    return 0
  fi

  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' >/dev/null 2>&1; then
        echo "$candidate"
        return 0
      fi
    fi
  done

  error "Could not find a compatible Python interpreter (>=3.10,<3.14) in PATH."
}

create_poetry_env() {
  local root="$1"
  local selected_python
  ensure_tool poetry
  selected_python="$(select_python_interpreter)"
  log "Configuring poetry to create an in-project virtual environment in $root/.venv"
  log "Using interpreter for poetry env: $(command -v "$selected_python")"
  "$selected_python" --version 2>/dev/null || true
  [[ $DRYRUN -eq 1 ]] && dryrun "poetry config virtualenvs.in-project true --local" && dryrun "poetry env use $selected_python" && dryrun "poetry install" && return
  (
    cd "$root"
    poetry config virtualenvs.in-project true --local
    if [[ $SYSTEM_PYTHON -eq 1 ]]; then
      poetry config virtualenvs.options.system-site-packages true --local
    fi
    poetry env use "$selected_python"
    poetry install
  )
}

create_uv_env() {
  local root="$1"
  ensure_tool uv
  log "Creating uv venv and installing dependencies in $root"
  [[ $DRYRUN -eq 1 ]] && dryrun "uv venv && uv pip install -r requirements.txt" && return
  (cd "$root" && uv venv)
  if [[ -f "$root/requirements.txt" ]]; then
    (cd "$root" && uv pip install -r requirements.txt)
  elif [[ -f "$root/pyproject.toml" ]]; then
    (cd "$root" && uv pip install .)
  fi
}

create_pixi_env() {
  local root="$1"
  ensure_tool pixi
  log "Installing dependencies with pixi in $root"
  [[ $DRYRUN -eq 1 ]] && dryrun "pixi install" && return
  (cd "$root" && pixi install)
}

create_pipx_env() {
  local root="$1"
  ensure_tool pipx
  log "Installing project with pipx in $root"
  [[ $DRYRUN -eq 1 ]] && dryrun "pipx install ." && return
  (cd "$root" && pipx install .)
}

# --- Validation ---
validate_env() {
  local root="$1"
  local tool="$2"
  log "Validating environment..."
  case "$tool" in
    venv|uv)
      if [[ -x "$root/venv/bin/python" ]]; then
        "$root/venv/bin/python" -c "import sys; print('Python:', sys.version)"
        "$root/venv/bin/pip" list
      else
        warn "Venv python not found!"
      fi
      ;;
    poetry)
      (cd "$root" && poetry run python -c "import sys; print('Python:', sys.version)")
      (cd "$root" && poetry run pip list)
      ;;
    pixi)
      (cd "$root" && pixi run python -c "import sys; print('Python:', sys.version)")
      (cd "$root" && pixi run pip list)
      ;;
    pipx)
      log "pipx environments are isolated; check with 'pipx list'"
      pipx list
      ;;
  esac
}

# --- Main logic ---
main() {
  # 1. Find project root
  local root
  root=$(find_project_root "$PROJECT_PATH") || error "Could not find project root from $PROJECT_PATH"
  log "Project root: $root"

  # 2. Print Python info
  print_python_info "$root"

  # 3. Clean if requested
  [[ $CLEAN -eq 1 ]] && clean_env "$root"

  # 4. Tool selection
  local selected_tool="$TOOL"
  if [[ "$selected_tool" == "auto" ]]; then
    selected_tool=$(detect_env_tool "$root")
    log "Auto-detected tool: $selected_tool"
  else
    log "Using user-specified tool: $selected_tool"
  fi

  # 5. Check tool availability
  case "$selected_tool" in
    venv) ensure_tool python3 ;;
    poetry) ensure_tool poetry ;;
    uv) ensure_tool uv ;;
    pixi) ensure_tool pixi ;;
    pipx) ensure_tool pipx ;;
    *) error "Unknown tool: $selected_tool" ;;
  esac

  # 6. Create environment
  [[ $FORCE -eq 1 ]] && clean_env "$root"
  case "$selected_tool" in
    venv) create_venv "$root" ;;
    poetry) create_poetry_env "$root" ;;
    uv) create_uv_env "$root" ;;
    pixi) create_pixi_env "$root" ;;
    pipx) create_pipx_env "$root" ;;
  esac

  # 7. Validate
  validate_env "$root" "$selected_tool"

  log "Environment setup complete!"
  print_summary "$selected_tool" "$root"
}

print_summary() {
  local tool="$1"
  local root="$2"
  echo -e "\n${c_bold}${c_green}=== Python Environment Setup Summary ===${c_reset}"
  case "$tool" in
    venv|uv)
      echo "Environment type: venv"
      echo "Location: $root/venv"
      echo "Activate:   source \"$root/venv/bin/activate\""
      echo "Deactivate: deactivate"
      echo "Run script: source \"$root/venv/bin/activate\" && python your_script.py"
      echo "Installed packages:"
      "$root/venv/bin/pip" list --format=columns | tail -n +3
      ;;
    poetry)
      echo "Environment type: poetry"
      local poetry_env_path
      local in_project_env="$root/.venv"
      # Prefer the in-project venv we configure above. Fall back to poetry env info.
      if [[ -d "$in_project_env" ]]; then
        poetry_env_path="$in_project_env"
      else
        poetry_env_path=$(cd "$root" && poetry env info -p 2>/dev/null || true)
      fi
      if [[ -n "$poetry_env_path" && -d "$poetry_env_path" ]]; then
        echo "Location: $poetry_env_path"
        echo "Activate: source \"$poetry_env_path/bin/activate\""
        echo "Deactivate: deactivate"
        echo "Run script (recommended): poetry run python your_script.py"
        echo "  (or, after activation: python your_script.py)"
        echo "Installed packages:"
        if [[ -x "$poetry_env_path/bin/pip" ]]; then
          "$poetry_env_path/bin/pip" list --format=columns | tail -n +3
        else
          (cd "$root" && poetry run pip list)
        fi
      else
        echo "Location: (could not determine poetry venv path automatically)"
        echo "Activate: poetry run python your_script.py"
        echo "Alternative: run 'poetry env info -p' manually and activate that path"
        echo "Deactivate: deactivate"
        echo "Installed packages:"
        (cd "$root" && poetry show || poetry run pip list)
      fi
      ;;
    pixi)
      echo "Environment type: pixi"
      echo "Location: $root/.pixi"
      echo "Activate: pixi shell"
      echo "Deactivate: exit"
      echo "Run script: pixi run python your_script.py"
      echo "Installed packages:"
      (cd "$root" && pixi list)
      ;;
    pipx)
      echo "Environment type: pipx"
      echo "Location: (managed by pipx, see pipx list)"
      echo "Run script: pipx run python your_script.py"
      echo "Installed packages:"
      pipx list
      ;;
    *)
      echo "Unknown environment type: $tool"
      ;;
  esac
}

main "$@"

# --- End of script ---

: <<'DOC'
# How to use this script

1. Place this script in your PATH or run directly.
2. Make it executable: chmod +x clean-py-env.sh
3. Run with your project path:
   ./clean-py-env.sh --project-path /path/to/project
4. To force recreation: ./clean-py-env.sh --force
5. To use a specific tool: ./clean-py-env.sh --tool poetry
6. To use system Python (for e.g. bcc): ./clean-py-env.sh --system-python
7. To clean before setup: ./clean-py-env.sh --clean
8. For dry-run: ./clean-py-env.sh --dry-run

The script will:
- Traverse up from the given path to find a Python project root.
- Detect the best tool to use (venv, poetry, uv, pixi, pipx).
- Create and validate the environment.
- Print a summary and instructions.

Extensible: Add new tools by adding new functions and updating the case statements.

DOC
