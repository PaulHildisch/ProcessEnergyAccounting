#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_daw_load_generation.sh [options]

Options:
  --pipelines-file <path>     File with one nf-core pipeline spec per line.
  --pipeline "<spec>"         Single pipeline spec. Repeatable.
  --backend-profile <name>    Additional Nextflow profile appended to test (for example: docker, singularity). Default: docker.
  --session-id <id>           Session directory name under runs/. Defaults to nfcore-<timestamp>.
  --idle-min <seconds>        Minimum random sleep between pipeline runs. Default: 30.
  --idle-max <seconds>        Maximum random sleep between pipeline runs. Default: 180.
  --initial-idle <seconds>    Fixed idle time before the first pipeline. Default: 60.
  --final-idle <seconds>      Fixed idle time after the last pipeline. Default: 60.
  --repeat <n>                Run each pipeline N times sequentially. Default: 1.
  --max-cpus <n>              Limit Nextflow process CPUs. Passed as process.cpus override. Default: unset.
  --stress-duration <seconds> Duration of each stress burst (uses stress-ng). Default: 0 (disabled).
  --stress-cpus <n>           Number of CPUs to stress. Default: all available.
  --pipeline-profile <name>   Nextflow test profile to use. Default: test.
                              Use "test_full" for larger datasets and longer runs.
                              Not all pipelines have test_full — those will fall back to test.

Examples:
  # Basic run
  scripts/run_daw_load_generation.sh --pipelines-file scripts/nfcore_test_pipelines.txt

  # More variance: repeat pipelines, add stress bursts, limit CPUs
  scripts/run_daw_load_generation.sh \
    --pipelines-file scripts/nfcore_test_pipelines.txt \
    --repeat 3 \
    --stress-duration 60 \
    --max-cpus 4

  # CPU sweep (run manually at different --max-cpus values for a gradient):
  #   --max-cpus 1, then --max-cpus 4, then --max-cpus 8
EOF
}

PIPELINES_FILE=""
SESSION_ID="nfcore-$(date -u +"%Y%m%dT%H%M%SZ")"
BACKEND_PROFILE="docker"
IDLE_MIN=30
IDLE_MAX=180
INITIAL_IDLE=60
FINAL_IDLE=60
REPEAT=1
MAX_CPUS=""
STRESS_DURATION=0
STRESS_CPUS=""
PIPELINE_PROFILE="test"
PIPELINE_SPECS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipelines-file)
      PIPELINES_FILE="$2"
      shift 2
      ;;
    --pipeline)
      PIPELINE_SPECS+=("$2")
      shift 2
      ;;
    --backend-profile)
      BACKEND_PROFILE="$2"
      shift 2
      ;;
    --session-id)
      SESSION_ID="$2"
      shift 2
      ;;
    --idle-min)
      IDLE_MIN="$2"
      shift 2
      ;;
    --idle-max)
      IDLE_MAX="$2"
      shift 2
      ;;
    --initial-idle)
      INITIAL_IDLE="$2"
      shift 2
      ;;
    --final-idle)
      FINAL_IDLE="$2"
      shift 2
      ;;
    --repeat)
      REPEAT="$2"
      shift 2
      ;;
    --max-cpus)
      MAX_CPUS="$2"
      shift 2
      ;;
    --stress-duration)
      STRESS_DURATION="$2"
      shift 2
      ;;
    --stress-cpus)
      STRESS_CPUS="$2"
      shift 2
      ;;
    --pipeline-profile)
      PIPELINE_PROFILE="$2"
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

if [[ -n "${PIPELINES_FILE}" ]]; then
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    PIPELINE_SPECS+=("${line}")
  done < "${PIPELINES_FILE}"
fi

if [[ ${#PIPELINE_SPECS[@]} -eq 0 ]]; then
  echo "No pipelines specified. Use --pipeline or --pipelines-file." >&2
  exit 1
fi

if (( IDLE_MIN < 0 || IDLE_MAX < 0 || INITIAL_IDLE < 0 || FINAL_IDLE < 0 )); then
  echo "Idle durations must be non-negative." >&2
  exit 1
fi

if (( IDLE_MIN > IDLE_MAX )); then
  echo "--idle-min must be less than or equal to --idle-max." >&2
  exit 1
fi

if [[ -z "${BACKEND_PROFILE}" ]]; then
  echo "--backend-profile must not be empty." >&2
  exit 1
fi

if (( REPEAT < 1 )); then
  echo "--repeat must be >= 1." >&2
  exit 1
fi

if (( STRESS_DURATION > 0 )) && ! command -v stress-ng &>/dev/null; then
  echo "stress-ng not found but --stress-duration > 0. Install with: sudo apt-get install stress-ng" >&2
  exit 1
fi

# Resolve stress CPU count (default: all available)
if [[ -z "${STRESS_CPUS}" ]]; then
  STRESS_CPUS="$(nproc)"
fi

SESSION_ROOT="runs/${SESSION_ID}"
SEGMENTS_ROOT="${SESSION_ROOT}/segments"
NEXTFLOW_ROOT="${SESSION_ROOT}/nextflow"
LOG_ROOT="${SESSION_ROOT}/logs"
MANIFEST_PATH="${SESSION_ROOT}/manifest.tsv"
SESSION_TRACE_PATH="${NEXTFLOW_ROOT}/session_trace.tsv"

mkdir -p "${SEGMENTS_ROOT}" "${NEXTFLOW_ROOT}" "${LOG_ROOT}"

printf "segment_id\ttype\tpipeline\tstart\tstop\tstatus\tpath\tidle_seconds\n" > "${MANIFEST_PATH}"
printf "workflow_run_id\tpipeline_name\ttask_id\tnative_id\tprocess\ttag\tstatus\tsubmit\tstart\tcomplete\tduration\trealtime\t%%cpu\tpeak_rss\tworkdir\n" > "${SESSION_TRACE_PATH}"

date -u +"%Y-%m-%dT%H:%M:%SZ" > "${SESSION_ROOT}/session_start.txt"

# Write runtime parameters for reproducibility
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

sanitize_name() {
  echo "$1" | tr '/ :' '---' | tr -cd '[:alnum:]_.-'
}

record_manifest() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "${MANIFEST_PATH}"
}

sleep_with_manifest() {
  local segment_id="$1"
  local duration="$2"
  local segment_dir="${SEGMENTS_ROOT}/${segment_id}"
  local start_time=""
  local stop_time=""

  mkdir -p "${segment_dir}"
  start_time="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  sleep "${duration}"
  stop_time="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  record_manifest \
    "${segment_id}" \
    "idle" \
    "__idle__" \
    "${start_time}" \
    "${stop_time}" \
    "ok" \
    "${segment_dir}" \
    "${duration}"
}

# Run stress-ng at full CPU for STRESS_DURATION seconds and record as a manifest segment.
# This creates high-energy, high-CPU-utilization intervals that are critical for model training.
stress_with_manifest() {
  local segment_id="$1"
  local segment_dir="${SEGMENTS_ROOT}/${segment_id}"
  local start_time=""
  local stop_time=""

  mkdir -p "${segment_dir}"
  echo "  [stress] Running stress-ng: ${STRESS_CPUS} CPUs for ${STRESS_DURATION}s"
  start_time="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  stress-ng --cpu "${STRESS_CPUS}" --timeout "${STRESS_DURATION}s" --metrics-brief \
    > "${segment_dir}/stress.log" 2>&1 || true
  stop_time="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  record_manifest \
    "${segment_id}" \
    "stress" \
    "__stress__" \
    "${start_time}" \
    "${stop_time}" \
    "ok" \
    "${segment_dir}" \
    ""
}

append_trace_rows() {
  local trace_file="$1"
  local workflow_run_id="$2"
  local pipeline_name="$3"

  if [[ -f "${trace_file}" ]]; then
    tail -n +2 "${trace_file}" | awk -F'\t' -v OFS='\t' -v workflow_run_id="${workflow_run_id}" -v pipeline_name="${pipeline_name}" \
      '{print workflow_run_id, pipeline_name, $0}' >> "${SESSION_TRACE_PATH}"
  fi
}

run_pipeline_segment() {
  local spec="$1"
  local repeat_idx="$2"
  local pipeline_parts=()
  local pipeline=""
  local pipeline_slug=""
  local segment_id=""
  local segment_dir=""
  local trace_file=""
  local report_file=""
  local timeline_file=""
  local outdir=""
  local start_time=""
  local stop_time=""
  local status="ok"

  read -r -a pipeline_parts <<< "${spec}"
  pipeline="${pipeline_parts[0]}"

  if [[ "${spec}" == *" -profile "* ]] || [[ "${spec}" == -profile* ]] || [[ "${spec}" == *" --outdir "* ]]; then
    echo "Pipeline spec must not contain -profile or --outdir: ${spec}" >&2
    exit 1
  fi

  pipeline_slug="$(sanitize_name "${pipeline##*/}")"
  segment_id="${pipeline_slug}-r${repeat_idx}-$(date -u +"%Y%m%dT%H%M%SZ")"
  segment_dir="${SEGMENTS_ROOT}/${segment_id}"
  trace_file="${segment_dir}/trace.txt"
  report_file="${segment_dir}/report.html"
  timeline_file="${segment_dir}/timeline.html"
  outdir="${segment_dir}/outdir"

  mkdir -p "${segment_dir}" "${outdir}"
  start_time="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  # Build optional CPU override config
  local extra_config_args=()
  if [[ -n "${MAX_CPUS}" ]]; then
    local cpu_config_file="${segment_dir}/cpu_override.config"
    printf 'process { cpus = %s }\n' "${MAX_CPUS}" > "${cpu_config_file}"
    extra_config_args+=(-c "${cpu_config_file}")
  fi

  local profile="${PIPELINE_PROFILE},${BACKEND_PROFILE}"

  # If test_full requested, try it first and fall back to test if it fails
  if [[ "${PIPELINE_PROFILE}" == "test_full" ]]; then
    echo "  Trying profile: ${profile}"
    if ! nextflow run "${pipeline_parts[@]}" \
      -c scripts/trace.config \
      "${extra_config_args[@]}" \
      -profile "${profile}" \
      --outdir "${outdir}" \
      -with-trace "${trace_file}" \
      -with-report "${report_file}" \
      -with-timeline "${timeline_file}"; then
      echo "  test_full failed or not available — falling back to test,${BACKEND_PROFILE}"
      profile="test,${BACKEND_PROFILE}"
      if ! nextflow run "${pipeline_parts[@]}" \
        -c scripts/trace.config \
        "${extra_config_args[@]}" \
        -profile "${profile}" \
        --outdir "${outdir}" \
        -with-trace "${trace_file}" \
        -with-report "${report_file}" \
        -with-timeline "${timeline_file}"; then
        status="failed"
      fi
    fi
  else
    if ! nextflow run "${pipeline_parts[@]}" \
      -c scripts/trace.config \
      "${extra_config_args[@]}" \
      -profile "${profile}" \
      --outdir "${outdir}" \
      -with-trace "${trace_file}" \
      -with-report "${report_file}" \
      -with-timeline "${timeline_file}"; then
      status="failed"
    fi
  fi

  stop_time="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  # Remove pipeline outputs and work directory immediately after the run —
  # they can be hundreds of GBs and are not needed for energy monitoring.
  # Trace, report, and timeline files are kept.
  if [[ -d "${outdir}" ]]; then
    echo "  Cleaning up outdir: ${outdir}"
    rm -rf "${outdir}"
  fi
  # Nextflow writes its work directory to the project root by default
  if [[ -d "work" ]]; then
    echo "  Cleaning up Nextflow work dir"
    rm -rf work/
  fi

  append_trace_rows "${trace_file}" "${segment_id}" "${pipeline}"

  record_manifest \
    "${segment_id}" \
    "pipeline" \
    "${pipeline}" \
    "${start_time}" \
    "${stop_time}" \
    "${status}" \
    "${segment_dir}" \
    ""

  if [[ "${status}" != "ok" ]]; then
    echo "Pipeline failed: ${spec}" >&2
    exit 1
  fi
}

random_idle_duration() {
  if (( IDLE_MIN == IDLE_MAX )); then
    echo "${IDLE_MIN}"
    return
  fi

  echo $(( RANDOM % (IDLE_MAX - IDLE_MIN + 1) + IDLE_MIN ))
}

# ── Main sequence ──────────────────────────────────────────────────────────────

if (( INITIAL_IDLE > 0 )); then
  echo "Initial idle: ${INITIAL_IDLE}s"
  sleep_with_manifest "idle-before" "${INITIAL_IDLE}"
fi

# Optional stress burst before pipelines to establish a high-energy baseline
if (( STRESS_DURATION > 0 )); then
  stress_with_manifest "stress-before"
  sleep_with_manifest "idle-after-stress-before" "10"
fi

for idx in "${!PIPELINE_SPECS[@]}"; do
  for rep in $(seq 1 "${REPEAT}"); do
    echo "Running pipeline $((idx + 1))/${#PIPELINE_SPECS[@]}, repeat ${rep}/${REPEAT}: ${PIPELINE_SPECS[$idx]}"
    run_pipeline_segment "${PIPELINE_SPECS[$idx]}" "${rep}"

    # Stress burst after each pipeline run (except the very last one)
    if (( STRESS_DURATION > 0 )); then
      if (( idx < ${#PIPELINE_SPECS[@]} - 1 || rep < REPEAT )); then
        stress_with_manifest "stress-after-${idx}-rep${rep}"
        sleep_with_manifest "idle-after-stress-${idx}-rep${rep}" "10"
      fi
    else
      # No stress: insert idle between runs (except after the last)
      if (( idx < ${#PIPELINE_SPECS[@]} - 1 || rep < REPEAT )); then
        idle_seconds="$(random_idle_duration)"
        if (( idle_seconds > 0 )); then
          sleep_with_manifest "idle-between-${idx}-rep${rep}" "${idle_seconds}"
        fi
      fi
    fi
  done
done

# Optional stress burst after all pipelines
if (( STRESS_DURATION > 0 )); then
  sleep_with_manifest "idle-before-final-stress" "10"
  stress_with_manifest "stress-after"
fi

if (( FINAL_IDLE > 0 )); then
  echo "Final idle: ${FINAL_IDLE}s"
  sleep_with_manifest "idle-after" "${FINAL_IDLE}"
fi

date -u +"%Y-%m-%dT%H:%M:%SZ" > "${SESSION_ROOT}/session_stop.txt"
echo "Generated nf-core load under ${SESSION_ROOT}"
