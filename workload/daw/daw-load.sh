#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
NEXTFLOW_DIR="${SCRIPT_DIR}/nextflow"
TRACE_CONFIG="${NEXTFLOW_DIR}/trace.config"
CO2_CONFIG="${NEXTFLOW_DIR}/co2footprint.config"
DEFAULT_PIPELINE="nf-core/rnaseq"

PIPELINES_FILE=""
SESSION_ID="nfcore-$(date -u +"%Y%m%dT%H%M%SZ")"
BACKEND_PROFILE="docker"
PIPELINE_PROFILE="test"
IDLE_MIN=30
IDLE_MAX=180
INITIAL_IDLE=60
FINAL_IDLE=60
REPEAT=1
MAX_CPUS=""
STRESS_DURATION=0
STRESS_CPUS=""
PIPELINE_SPECS=()
EXTRA_CONFIG_ARGS=()
FAILED_PIPELINES=0

SESSIONS_DIR="${NEXTFLOW_DIR}"
SESSION_ROOT=""
SEGMENTS_ROOT=""
MANIFEST_PATH=""

usage() {
  cat <<'EOF'
Usage:
  workload/daw/daw-load.sh [options]

Options:
  --pipelines-file <path>     File with one pipeline spec per line.
  --pipeline "<spec>"         Single pipeline spec. Repeatable.
  --backend-profile <name>    Backend profile appended to the pipeline profile. Default: docker.
  --session-id <id>           Session directory under workload/daw/nextflow/. Default: nfcore-<timestamp>.
  --idle-min <seconds>        Minimum idle between pipeline runs. Default: 30.
  --idle-max <seconds>        Maximum idle between pipeline runs. Default: 180.
  --initial-idle <seconds>    Idle before the first pipeline. Default: 60.
  --final-idle <seconds>      Idle after the last pipeline. Default: 60.
  --repeat <n>                Repeat each pipeline N times. Default: 1.
  --max-cpus <n>              Override Nextflow process.cpus.
  --stress-duration <seconds> Stress burst duration. Default: 0 (disabled).
  --stress-cpus <n>           CPUs used by stress-ng. Default: all available.
  --pipeline-profile <name>   Nextflow pipeline profile. Default: test.

Examples:
  workload/daw/daw-load.sh --pipelines-file workload/daw/nextflow/nfcore_test_pipelines.txt

  workload/daw/daw-load.sh \
    --pipelines-file workload/daw/nextflow/nfcore_test_pipelines.txt \
    --repeat 3 \
    --stress-duration 60 \
    --max-cpus 4
EOF
}

fail() {
  echo "$*" >&2
  exit 1
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --pipelines-file) PIPELINES_FILE="$2"; shift 2 ;;
      --pipeline) PIPELINE_SPECS+=("$2"); shift 2 ;;
      --backend-profile) BACKEND_PROFILE="$2"; shift 2 ;;
      --session-id) SESSION_ID="$2"; shift 2 ;;
      --idle-min) IDLE_MIN="$2"; shift 2 ;;
      --idle-max) IDLE_MAX="$2"; shift 2 ;;
      --initial-idle) INITIAL_IDLE="$2"; shift 2 ;;
      --final-idle) FINAL_IDLE="$2"; shift 2 ;;
      --repeat) REPEAT="$2"; shift 2 ;;
      --max-cpus) MAX_CPUS="$2"; shift 2 ;;
      --stress-duration) STRESS_DURATION="$2"; shift 2 ;;
      --stress-cpus) STRESS_CPUS="$2"; shift 2 ;;
      --pipeline-profile) PIPELINE_PROFILE="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) usage >&2; fail "Unknown argument: $1" ;;
    esac
  done
}

load_pipelines() {
  if [[ -n "${PIPELINES_FILE}" ]]; then
    while IFS= read -r line; do
      [[ -z "${line}" ]] && continue
      [[ "${line}" =~ ^[[:space:]]*# ]] && continue
      PIPELINE_SPECS+=("${line}")
    done < "${PIPELINES_FILE}"
  fi

  if [[ ${#PIPELINE_SPECS[@]} -eq 0 ]]; then
    PIPELINE_SPECS=("${DEFAULT_PIPELINE}")
    echo "No pipelines specified. Defaulting to: ${DEFAULT_PIPELINE}"
  fi

  echo "Loaded ${#PIPELINE_SPECS[@]} pipeline spec(s)."
}

validate_config() {
  [[ -f "${TRACE_CONFIG}" ]] || fail "Missing required file: ${TRACE_CONFIG}"
  (( IDLE_MIN >= 0 && IDLE_MAX >= 0 && INITIAL_IDLE >= 0 && FINAL_IDLE >= 0 )) || fail "Idle durations must be non-negative."
  (( IDLE_MIN <= IDLE_MAX )) || fail "--idle-min must be <= --idle-max."
  (( REPEAT >= 1 )) || fail "--repeat must be >= 1."
  [[ -n "${BACKEND_PROFILE}" ]] || fail "--backend-profile must not be empty."

  if (( STRESS_DURATION > 0 )) && ! command -v stress-ng >/dev/null 2>&1; then
    fail "stress-ng not found but --stress-duration > 0"
  fi

  if [[ -z "${STRESS_CPUS}" ]]; then
    STRESS_CPUS="$(nproc)"
  fi
}

setup_session() {
  SESSION_ROOT="${SESSIONS_DIR}/${SESSION_ID}"
  SEGMENTS_ROOT="${SESSION_ROOT}/segments"
  MANIFEST_PATH="${SESSION_ROOT}/manifest.tsv"

  mkdir -p "${SEGMENTS_ROOT}"

  printf "segment_id\ttype\tpipeline\tstart\tstop\tstatus\tpath\tidle_seconds\n" > "${MANIFEST_PATH}"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${SESSION_ROOT}/session_start.txt"

  {
    echo "session_id=${SESSION_ID}"
    echo "repeat=${REPEAT}"
    echo "max_cpus=${MAX_CPUS:-unset}"
    echo "stress_duration=${STRESS_DURATION}"
    echo "stress_cpus=${STRESS_CPUS}"
    echo "idle_min=${IDLE_MIN}"
    echo "idle_max=${IDLE_MAX}"
    echo "initial_idle=${INITIAL_IDLE}"
    echo "final_idle=${FINAL_IDLE}"
    echo "backend_profile=${BACKEND_PROFILE}"
    echo "pipeline_profile=${PIPELINE_PROFILE}"
  } > "${SESSION_ROOT}/session_params.txt"
}

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

record_manifest() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "${MANIFEST_PATH}"
}

sanitize_name() {
  echo "$1" | tr '/ :' '---' | tr -cd '[:alnum:]_.-'
}

idle_phase() {
  local segment_id="$1"
  local seconds="$2"
  local segment_dir="${SEGMENTS_ROOT}/${segment_id}"
  local start_time
  local stop_time

  mkdir -p "${segment_dir}"
  start_time="$(timestamp)"
  sleep "${seconds}"
  stop_time="$(timestamp)"

  record_manifest "${segment_id}" "idle" "__idle__" "${start_time}" "${stop_time}" "ok" "${segment_dir}" "${seconds}"
}

stress_phase() {
  local segment_id="$1"
  local segment_dir="${SEGMENTS_ROOT}/${segment_id}"
  local start_time
  local stop_time

  mkdir -p "${segment_dir}"
  echo "  [stress] ${STRESS_CPUS} CPUs for ${STRESS_DURATION}s"
  start_time="$(timestamp)"
  stress-ng --cpu "${STRESS_CPUS}" --timeout "${STRESS_DURATION}s" --metrics-brief > "${segment_dir}/stress.log" 2>&1 || true
  stop_time="$(timestamp)"

  record_manifest "${segment_id}" "stress" "__stress__" "${start_time}" "${stop_time}" "ok" "${segment_dir}" ""
}

build_extra_configs() {
  local segment_dir="$1"
  EXTRA_CONFIG_ARGS=()

  if [[ -n "${MAX_CPUS}" ]]; then
    local cpu_config="${segment_dir}/cpu_override.config"
    printf 'process { cpus = %s }\n' "${MAX_CPUS}" > "${cpu_config}"
    EXTRA_CONFIG_ARGS+=(-c "${cpu_config}")
  fi

  if [[ -f "${CO2_CONFIG}" ]]; then
    EXTRA_CONFIG_ARGS+=(-c "${CO2_CONFIG}")
  fi
}

cleanup_after_pipeline() {
  local outdir="$1"
  local segment_dir="$2"
  local co2file

  [[ -d "${outdir}" ]] && rm -rf "${outdir}"
  [[ -d "${REPO_ROOT}/work" ]] && rm -rf "${REPO_ROOT}/work"

  for co2file in co2footprint_trace.txt co2footprint_summary.txt co2footprint_report.html; do
    [[ -f "${REPO_ROOT}/${co2file}" ]] && mv "${REPO_ROOT}/${co2file}" "${segment_dir}/${co2file}"
  done
}

run_nextflow() {
  (
    cd "${REPO_ROOT}"
    nextflow run "$@"
  )
}

run_pipeline_once() {
  local spec="$1"
  local repeat_idx="$2"
  local pipeline_parts=()
  local pipeline
  local profile="${PIPELINE_PROFILE},${BACKEND_PROFILE}"
  local pipeline_slug
  local segment_id
  local segment_dir
  local trace_file
  local report_file
  local timeline_file
  local outdir
  local start_time
  local stop_time
  local status="ok"

  read -r -a pipeline_parts <<< "${spec}"
  pipeline="${pipeline_parts[0]}"

  if [[ "${spec}" == *" -profile "* ]] || [[ "${spec}" == -profile* ]] || [[ "${spec}" == *" --outdir "* ]]; then
    fail "Pipeline spec must not contain -profile or --outdir: ${spec}"
  fi

  pipeline_slug="$(sanitize_name "${pipeline##*/}")"
  segment_id="${pipeline_slug}-r${repeat_idx}-$(date -u +"%Y%m%dT%H%M%SZ")"
  segment_dir="${SEGMENTS_ROOT}/${segment_id}"
  trace_file="${segment_dir}/trace.txt"
  report_file="${segment_dir}/report.html"
  timeline_file="${segment_dir}/timeline.html"
  outdir="${segment_dir}/outdir"

  mkdir -p "${segment_dir}" "${outdir}"
  build_extra_configs "${segment_dir}"
  start_time="$(timestamp)"

  if [[ "${PIPELINE_PROFILE}" == "test_full" ]]; then
    echo "  Trying profile: ${profile}"
    if ! run_nextflow "${pipeline_parts[@]}" -c "${TRACE_CONFIG}" "${EXTRA_CONFIG_ARGS[@]}" -profile "${profile}" --outdir "${outdir}" -with-trace "${trace_file}" -with-report "${report_file}" -with-timeline "${timeline_file}"; then
      profile="test,${BACKEND_PROFILE}"
      echo "  Falling back to: ${profile}"
      if ! run_nextflow "${pipeline_parts[@]}" -c "${TRACE_CONFIG}" "${EXTRA_CONFIG_ARGS[@]}" -profile "${profile}" --outdir "${outdir}" -with-trace "${trace_file}" -with-report "${report_file}" -with-timeline "${timeline_file}"; then
        status="failed"
      fi
    fi
  else
    if ! run_nextflow "${pipeline_parts[@]}" -c "${TRACE_CONFIG}" "${EXTRA_CONFIG_ARGS[@]}" -profile "${profile}" --outdir "${outdir}" -with-trace "${trace_file}" -with-report "${report_file}" -with-timeline "${timeline_file}"; then
      status="failed"
    fi
  fi

  stop_time="$(timestamp)"
  cleanup_after_pipeline "${outdir}" "${segment_dir}"
  record_manifest "${segment_id}" "pipeline" "${pipeline}" "${start_time}" "${stop_time}" "${status}" "${segment_dir}" ""

  if [[ "${status}" != "ok" ]]; then
    echo "Pipeline failed (continuing): ${spec}" >&2
    ((FAILED_PIPELINES += 1))
  fi
}

random_idle() {
  if (( IDLE_MIN == IDLE_MAX )); then
    echo "${IDLE_MIN}"
  else
    echo $(( RANDOM % (IDLE_MAX - IDLE_MIN + 1) + IDLE_MIN ))
  fi
}

run_between_runs_phase() {
  local idx="$1"
  local rep="$2"
  local is_last=0

  if (( idx == ${#PIPELINE_SPECS[@]} - 1 && rep == REPEAT )); then
    is_last=1
  fi
  (( is_last == 1 )) && return

  if (( STRESS_DURATION > 0 )); then
    stress_phase "stress-after-${idx}-rep${rep}"
    idle_phase "idle-after-stress-${idx}-rep${rep}" "10"
  else
    local idle_seconds
    idle_seconds="$(random_idle)"
    (( idle_seconds > 0 )) && idle_phase "idle-between-${idx}-rep${rep}" "${idle_seconds}"
  fi
}

main() {
  parse_args "$@"
  load_pipelines
  validate_config
  setup_session

  if (( INITIAL_IDLE > 0 )); then
    echo "Initial idle: ${INITIAL_IDLE}s"
    idle_phase "idle-before" "${INITIAL_IDLE}"
  fi

  if (( STRESS_DURATION > 0 )); then
    stress_phase "stress-before"
    idle_phase "idle-after-stress-before" "10"
  fi

  local idx
  local rep
  for idx in "${!PIPELINE_SPECS[@]}"; do
    for rep in $(seq 1 "${REPEAT}"); do
      echo "Running pipeline $((idx + 1))/${#PIPELINE_SPECS[@]}, repeat ${rep}/${REPEAT}: ${PIPELINE_SPECS[$idx]}"
      run_pipeline_once "${PIPELINE_SPECS[$idx]}" "${rep}"
      run_between_runs_phase "${idx}" "${rep}"
    done
  done

  if (( STRESS_DURATION > 0 )); then
    idle_phase "idle-before-final-stress" "10"
    stress_phase "stress-after"
  fi

  if (( FINAL_IDLE > 0 )); then
    echo "Final idle: ${FINAL_IDLE}s"
    idle_phase "idle-after" "${FINAL_IDLE}"
  fi

  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${SESSION_ROOT}/session_stop.txt"
  echo "daw-load.sh generated nf-core load under ${SESSION_ROOT}"
  if (( FAILED_PIPELINES > 0 )); then
    echo "Completed with ${FAILED_PIPELINES} failed pipeline run(s). See ${MANIFEST_PATH}." >&2
  fi
}

main "$@"
