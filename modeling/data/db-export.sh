#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CALLER_PWD="$(pwd)"
MODELING_DIR="${REPO_ROOT}/modeling"
DATA_LOADER_PATH="${MODELING_DIR}/data/data_loader.py"

usage() {
  cat <<'EOF'
Usage:
  modeling/data/db-export.sh --session-dir <session-dir> [--output <path>] [--aggregate-every <window>]

Examples:
  modeling/data/db-export.sh --session-dir workload/daw/nextflow/nfcore-20260312T120000Z
  modeling/data/db-export.sh --session-dir workload/daw/nextflow/nfcore-20260312T120000Z --output cpu10_data.parquet
EOF
}

SESSION_DIR=""
OUTPUT_PATH=""
AGGREGATE_EVERY="1s"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-dir)
      SESSION_DIR="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --aggregate-every)
      AGGREGATE_EVERY="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${SESSION_DIR}" ]]; then
  echo "--session-dir is required." >&2
  exit 1
fi

resolve_session_dir() {
  local candidate="$1"

  if [[ "${candidate}" == /* ]]; then
    [[ -d "${candidate}" ]] || return 1
    (cd -- "${candidate}" && pwd)
    return 0
  fi

  if [[ -d "${candidate}" ]]; then
    (cd -- "${candidate}" && pwd)
    return 0
  fi

  if [[ -d "${REPO_ROOT}/${candidate}" ]]; then
    (cd -- "${REPO_ROOT}/${candidate}" && pwd)
    return 0
  fi

  return 1
}

if ! SESSION_DIR="$(resolve_session_dir "${SESSION_DIR}")"; then
  echo "Session directory not found: ${SESSION_DIR}" >&2
  echo "Tried as absolute path, relative to current dir (${CALLER_PWD}), and relative to repo root (${REPO_ROOT})." >&2
  exit 1
fi

START_FILE="${SESSION_DIR}/session_start.txt"
STOP_FILE="${SESSION_DIR}/session_stop.txt"

if [[ ! -f "${START_FILE}" || ! -f "${STOP_FILE}" ]]; then
  echo "Session timestamps not found under ${SESSION_DIR}" >&2
  exit 1
fi

START_TIME="$(tr -d '\n' < "${START_FILE}")"
STOP_TIME="$(tr -d '\n' < "${STOP_FILE}")"
if [[ -n "${OUTPUT_PATH}" ]]; then
  if [[ "${OUTPUT_PATH}" != /* ]]; then
    OUTPUT_PATH="${CALLER_PWD}/${OUTPUT_PATH}"
  fi
else
  OUTPUT_PATH="${SESSION_DIR}/datasets/process_interval_data.parquet"
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

if [[ ! -f "${DATA_LOADER_PATH}" ]]; then
  echo "Data loader not found at ${DATA_LOADER_PATH}" >&2
  exit 1
fi

if command -v poetry >/dev/null 2>&1; then
  (
    cd "${MODELING_DIR}"
    PYTHONPATH="${REPO_ROOT}/monitor:${REPO_ROOT}:${PYTHONPATH:-}" \
      poetry run python "${DATA_LOADER_PATH}" \
      --level process \
      --start "${START_TIME}" \
      --stop "${STOP_TIME}" \
      --aggregate-every "${AGGREGATE_EVERY}" \
      --output "${OUTPUT_PATH}"
  )
else
  (
    cd "${MODELING_DIR}"
    PYTHONPATH="${REPO_ROOT}/monitor:${REPO_ROOT}:${PYTHONPATH:-}" \
      python3 "${DATA_LOADER_PATH}" \
      --level process \
      --start "${START_TIME}" \
      --stop "${STOP_TIME}" \
      --aggregate-every "${AGGREGATE_EVERY}" \
      --output "${OUTPUT_PATH}"
  )
fi

echo "db-export.sh exported dataset to ${OUTPUT_PATH}"
