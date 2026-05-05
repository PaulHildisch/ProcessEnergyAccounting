#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/export_process_dataset.sh --session-dir <runs/session-id> [--output <path>] [--aggregate-every <window>]

Examples:
  scripts/export_process_dataset.sh --session-dir runs/nfcore-20260312T120000Z
  scripts/export_process_dataset.sh --session-dir runs/nfcore-20260312T120000Z --output runs/nfcore-20260312T120000Z/datasets/process_interval_data.parquet
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

START_FILE="${SESSION_DIR}/session_start.txt"
STOP_FILE="${SESSION_DIR}/session_stop.txt"

if [[ ! -f "${START_FILE}" || ! -f "${STOP_FILE}" ]]; then
  echo "Session timestamps not found under ${SESSION_DIR}" >&2
  exit 1
fi

START_TIME="$(tr -d '\n' < "${START_FILE}")"
STOP_TIME="$(tr -d '\n' < "${STOP_FILE}")"
OUTPUT_PATH="${OUTPUT_PATH:-${SESSION_DIR}/datasets/process_interval_data.parquet}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

if command -v poetry >/dev/null 2>&1; then
  poetry run python -m estimation.data.data_loader \
    --level process \
    --start "${START_TIME}" \
    --stop "${STOP_TIME}" \
    --aggregate-every "${AGGREGATE_EVERY}" \
    --output "${OUTPUT_PATH}"
else
  python3 -m estimation.data.data_loader \
    --level process \
    --start "${START_TIME}" \
    --stop "${STOP_TIME}" \
    --aggregate-every "${AGGREGATE_EVERY}" \
    --output "${OUTPUT_PATH}"
fi

echo "Exported process dataset to ${OUTPUT_PATH}"
