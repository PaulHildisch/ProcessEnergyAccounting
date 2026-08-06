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
WORKLOAD_PHASES=()
NEXTFLOW_CONFIG_FILES=()
EXTRA_CONFIG_ARGS=()
FAILED_PIPELINES=0

SESSIONS_DIR="${NEXTFLOW_DIR}"
SESSION_ROOT=""
SEGMENTS_ROOT=""
MANIFEST_PATH=""

HAS_STRESS_NG=$(command -v stress-ng >/dev/null 2>&1 && echo 1 || echo 0)
HAS_GCC=$(command -v gcc >/dev/null 2>&1 && echo 1 || echo 0)
HAS_PYTHON3=$(command -v python3 >/dev/null 2>&1 && echo 1 || echo 0)
MAX_CORES=$(nproc)
MAX_MEM_GB=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
IO_DIR=""
IO_DIRECT=0
NET_HOST="localhost"
NET_PROBE_DURATION=5
NET_CONNECT_TIMEOUT=5
NET_MAX_MBPS=""
IPERF3_SERVER_PID=""

usage() {
  cat <<'EOF'
Usage:
  workload/daw/daw-load.sh [options]

Options:
  --pipelines-file <path>     File with one pipeline spec per line.
  --pipeline "<spec>"         Single pipeline spec. Repeatable.
  --backend-profile <name>    Backend profile appended to the pipeline profile. Default: docker.
  --nextflow-config <path>    Extra Nextflow config file passed via -c. Repeatable.
  --session-id <id>           Session directory under workload/daw/nextflow/. Default: nfcore-<timestamp>.
  --idle-min <seconds>        Minimum idle between pipeline runs. Default: 30.
  --idle-max <seconds>        Maximum idle between pipeline runs. Default: 180.
  --initial-idle <seconds>    Idle before the first pipeline. Default: 60.
  --final-idle <seconds>      Idle after the last pipeline. Default: 60.
  --repeat <n>                Repeat each pipeline N times. Default: 1.
  --max-cpus <n>              Override Nextflow process.cpus.
  --pipeline-profile <name>   Nextflow pipeline profile. Default: test.

Workload phases between pipeline runs:
  --phase "<spec>"            Repeatable workload phase.
                              Sequential format: type:mode:duration[:param=value,...]
                              Concurrent format:  parallel:<spec1>;<spec2>[;<spec3>...]
                              Examples:
                                cpu:scalar:30:cores=4
                                cpu:avx:30:cores=8
                                mem:bandwidth:45:size=16
                                io:seq-write:30:io=2048
                                net:bandwidth:30:net=50%
                                idle:10
                                parallel:cpu:avx:300:cores=8;mem:bandwidth:240:size=32
                                parallel:io:random:320:io=2048;net:bursty:260:net=50%
  --io-dir <path>             Scratch directory for I/O phases. Default: nextflow/<session>/io-scratch.
  --net-host <host>           iperf3 target host for network phases. Default: localhost.
  --net-probe-duration <sec>  Probe duration for net=max/net=%. Default: 5.
  --net-connect-timeout <sec> iperf3 connectivity timeout. Default: 5.

Legacy CPU-only stress options:
  --stress-duration <seconds> Stress burst duration. Default: 0 (disabled).
  --stress-cpus <n>           CPUs used by legacy stress-ng phase. Default: all available.
                              If --phase is provided, these legacy options are ignored.

Supported phase modes:
  CPU: cpu:scalar, cpu:avx, cpu:mixed, cpu:burst, cpu:sweep
  MEM: mem:bandwidth, mem:random, mem:sparse, mem:allocation, mem:sweep
  I/O: io:seq-read, io:seq-write, io:random
  NET: net:bandwidth, net:bursty

Examples:
  workload/daw/daw-load.sh --pipelines-file workload/daw/nextflow/nfcore_test_pipelines.txt

  workload/daw/daw-load.sh \
    --pipeline nf-core/rnaseq \
    --pipeline nf-core/sarek \
    --phase "cpu:avx:60:cores=8" \
    --phase "mem:bandwidth:60:size=16" \
    --phase "io:random:60:io=2048" \
    --phase "net:bandwidth:60:net=50%" \
    --net-host 130.149.248.105
EOF
}

fail() {
  echo "$*" >&2
  exit 1
}

log_info() {
  echo "[$(date +'%H:%M:%S')] $*"
}

resolve_path() {
  local input_path="$1"
  local dir base
  dir=$(cd -- "$(dirname -- "$input_path")" && pwd) || return 1
  base=$(basename -- "$input_path")
  printf '%s/%s\n' "$dir" "$base"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --pipelines-file) PIPELINES_FILE="$2"; shift 2 ;;
      --pipeline) PIPELINE_SPECS+=("$2"); shift 2 ;;
      --backend-profile) BACKEND_PROFILE="$2"; shift 2 ;;
      --nextflow-config) NEXTFLOW_CONFIG_FILES+=("$2"); shift 2 ;;
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
      --phase) WORKLOAD_PHASES+=("$2"); shift 2 ;;
      --io-dir) IO_DIR="$2"; shift 2 ;;
      --net-host) NET_HOST="$2"; shift 2 ;;
      --net-probe-duration) NET_PROBE_DURATION="$2"; shift 2 ;;
      --net-connect-timeout) NET_CONNECT_TIMEOUT="$2"; shift 2 ;;
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

phase_type() {
  local spec="$1"
  echo "${spec%%:*}"
}

split_parallel_phase() {
  local spec="$1"
  local payload
  [[ "$(phase_type "${spec}")" == "parallel" ]] || return 1
  payload="${spec#parallel:}"
  IFS=';' read -r -a SPLIT_PARALLEL_PHASE_RESULT <<< "${payload}"
}

phase_contains_type() {
  local spec="$1"
  local wanted="$2"
  local subphase

  if [[ "$(phase_type "${spec}")" == "parallel" ]]; then
    split_parallel_phase "${spec}" || return 1
    for subphase in "${SPLIT_PARALLEL_PHASE_RESULT[@]}"; do
      [[ -z "${subphase}" ]] && continue
      [[ "$(phase_type "${subphase}")" == "${wanted}" ]] && return 0
    done
    return 1
  fi

  [[ "$(phase_type "${spec}")" == "${wanted}" ]]
}

has_phase_type() {
  local wanted="$1"
  local phase
  for phase in "${WORKLOAD_PHASES[@]}"; do
    phase_contains_type "${phase}" "${wanted}" && return 0
  done
  return 1
}

validate_config() {
  local idx resolved

  [[ -f "${TRACE_CONFIG}" ]] || fail "Missing required file: ${TRACE_CONFIG}"
  (( IDLE_MIN >= 0 && IDLE_MAX >= 0 && INITIAL_IDLE >= 0 && FINAL_IDLE >= 0 )) || fail "Idle durations must be non-negative."
  (( IDLE_MIN <= IDLE_MAX )) || fail "--idle-min must be <= --idle-max."
  (( REPEAT >= 1 )) || fail "--repeat must be >= 1."
  [[ -n "${BACKEND_PROFILE}" ]] || fail "--backend-profile must not be empty."

  for idx in "${!NEXTFLOW_CONFIG_FILES[@]}"; do
    [[ -f "${NEXTFLOW_CONFIG_FILES[$idx]}" ]] || fail "Missing Nextflow config: ${NEXTFLOW_CONFIG_FILES[$idx]}"
    resolved=$(resolve_path "${NEXTFLOW_CONFIG_FILES[$idx]}") || fail "Could not resolve path: ${NEXTFLOW_CONFIG_FILES[$idx]}"
    NEXTFLOW_CONFIG_FILES[$idx]="$resolved"
  done

  if [[ -z "${STRESS_CPUS}" ]]; then
    STRESS_CPUS="$(nproc)"
  fi

  if has_phase_type "cpu" || has_phase_type "mem"; then
    [[ ${HAS_STRESS_NG} -eq 1 || ${HAS_GCC} -eq 1 || ${HAS_PYTHON3} -eq 1 ]] || \
      fail "CPU/memory phases need stress-ng, gcc, or python3."
  fi

  if has_phase_type "io"; then
    setup_io_dir
  fi

  if has_phase_type "net"; then
    prepare_network_phases
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
    echo "workload_phases=${WORKLOAD_PHASES[*]:-none}"
    echo "idle_min=${IDLE_MIN}"
    echo "idle_max=${IDLE_MAX}"
    echo "initial_idle=${INITIAL_IDLE}"
    echo "final_idle=${FINAL_IDLE}"
    echo "backend_profile=${BACKEND_PROFILE}"
    echo "pipeline_profile=${PIPELINE_PROFILE}"
    echo "nextflow_config_files=${NEXTFLOW_CONFIG_FILES[*]:-none}"
    echo "io_dir=${IO_DIR:-unset}"
    echo "net_host=${NET_HOST}"
    echo "net_probe_duration=${NET_PROBE_DURATION}"
    echo "net_connect_timeout=${NET_CONNECT_TIMEOUT}"
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

parse_phase_params() {
  local params_str="$1"
  local -n result=$2
  local pair key value

  [[ -z "${params_str}" ]] && return
  IFS=',' read -ra param_pairs <<< "${params_str}"
  for pair in "${param_pairs[@]}"; do
    IFS='=' read -r key value <<< "${pair}"
    result["${key}"]="${value}"
  done
}

setup_io_dir() {
  [[ -z "${IO_DIR}" ]] && IO_DIR="${SESSION_ROOT:-${NEXTFLOW_DIR}}/io-scratch"
  mkdir -p "${IO_DIR}"

  if dd if=/dev/zero of="${IO_DIR}/.directprobe" bs=4k count=1 oflag=direct >/dev/null 2>&1; then
    IO_DIRECT=1
  else
    IO_DIRECT=0
  fi
  rm -f "${IO_DIR}/.directprobe"
  log_info "I/O scratch dir: ${IO_DIR} (direct=${IO_DIRECT})"
}

start_iperf3_server() {
  if [[ "${NET_HOST}" != "localhost" && "${NET_HOST}" != "127.0.0.1" && "${NET_HOST}" != "::1" ]]; then
    return 0
  fi
  command -v iperf3 >/dev/null 2>&1 || return 1
  if timeout 1s iperf3 -c "${NET_HOST}" -t 1 >/dev/null 2>&1; then
    return 0
  fi
  iperf3 -s >/dev/null 2>&1 &
  IPERF3_SERVER_PID=$!
  sleep 0.5
}

cleanup() {
  if [[ -n "${IPERF3_SERVER_PID}" ]]; then
    kill -TERM "${IPERF3_SERVER_PID}" 2>/dev/null || true
    wait "${IPERF3_SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

check_net_reachable() {
  command -v iperf3 >/dev/null 2>&1 || fail "iperf3 not found; install iperf3 before running network phases."
  log_info "Checking iperf3 connectivity to ${NET_HOST}"
  timeout "${NET_CONNECT_TIMEOUT}s" iperf3 -c "${NET_HOST}" -t 1 >/dev/null 2>&1 || \
    fail "Cannot reach iperf3 server at ${NET_HOST}:5201. Start 'iperf3 -s' on the target host."
}

probe_net_max_mbps() {
  local probe_timeout=$((NET_PROBE_DURATION + NET_CONNECT_TIMEOUT + 2))
  local probe_output

  log_info "Probing max network rate against ${NET_HOST} for ${NET_PROBE_DURATION}s"
  probe_output=$(timeout "${probe_timeout}s" iperf3 -c "${NET_HOST}" -t "${NET_PROBE_DURATION}" -J 2>/dev/null || true)
  [[ -n "${probe_output}" ]] || fail "iperf3 network probe failed for ${NET_HOST}."

  NET_MAX_MBPS=$(python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
    end = data.get("end", {})
    bps = end.get("sum_received", {}).get("bits_per_second") or end.get("sum_sent", {}).get("bits_per_second") or 0
    print(max(0, int(bps / 1_000_000)))
except Exception:
    print(0)
' <<< "${probe_output}")

  [[ "${NET_MAX_MBPS}" =~ ^[0-9]+$ && "${NET_MAX_MBPS}" -gt 0 ]] || fail "Could not detect max network rate for ${NET_HOST}."
  log_info "Detected max network rate: ${NET_MAX_MBPS} Mbps"
}

prepare_network_phases() {
  start_iperf3_server || true
  check_net_reachable
  probe_net_max_mbps
}

resolve_net_mbps() {
  local requested=${1:-max}
  if [[ "${requested}" == "max" || "${requested}" == "0" ]]; then
    echo "${NET_MAX_MBPS:-0}"
  elif [[ "${requested}" =~ ^([0-9]+)%$ ]]; then
    echo $((NET_MAX_MBPS * BASH_REMATCH[1] / 100))
  else
    echo "${requested}"
  fi
}

cpu_stress_scalar() {
  local duration="$1"
  local cores="$2"
  if [[ ${HAS_STRESS_NG} -eq 1 ]]; then
    stress-ng --cpu "${cores}" --cpu-method int64 --timeout "${duration}s" --metrics-brief
  else
    local pids=()
    for _ in $(seq 1 "${cores}"); do (x=0; while true; do x=$((x + 1)); done) & pids+=("$!"); done
    sleep "${duration}"
    kill "${pids[@]}" 2>/dev/null || true
    wait 2>/dev/null || true
  fi
}

cpu_stress_avx() {
  local duration="$1"
  local cores="$2"
  if [[ ${HAS_STRESS_NG} -eq 1 ]]; then
    stress-ng --cpu "${cores}" --cpu-method ackermann --timeout "${duration}s" --metrics-brief
  else
    cpu_stress_scalar "${duration}" "${cores}"
  fi
}

cpu_stress_mixed() {
  local duration="$1"
  local cores="$2"
  local end_time=$(($(date +%s) + duration))
  local remaining
  while [[ $(date +%s) -lt ${end_time} ]]; do
    remaining=$((end_time - $(date +%s)))
    [[ ${remaining} -le 0 ]] && break
    cpu_stress_scalar "$((remaining < 5 ? remaining : 5))" "${cores}"
    remaining=$((end_time - $(date +%s)))
    [[ ${remaining} -le 0 ]] && break
    cpu_stress_avx "$((remaining < 5 ? remaining : 5))" "${cores}"
  done
}

cpu_stress_burst() {
  local duration="$1"
  local cores="$2"
  if [[ ${HAS_STRESS_NG} -eq 1 ]]; then
    stress-ng --cpu "${cores}" --cpu-load 50 --timeout "${duration}s" --metrics-brief
  else
    cpu_stress_mixed "${duration}" "${cores}"
  fi
}

cpu_stress_sweep() {
  local duration="$1"
  local max_cores="$2"
  local cores=1
  while [[ ${cores} -le ${max_cores} ]]; do
    cpu_stress_avx "${duration}" "${cores}"
    [[ ${cores} -ge ${max_cores} ]] && break
    if [[ ${cores} -eq 1 ]]; then cores=2; else cores=$((cores * 2)); fi
    [[ ${cores} -gt ${max_cores} ]] && cores=${max_cores}
  done
}

mem_stress_bandwidth() {
  local duration="$1"
  local size_gb="$2"
  local mem_mb=$((size_gb * 1024))
  if [[ ${HAS_STRESS_NG} -eq 1 ]]; then
    stress-ng --vm 1 --vm-bytes "${mem_mb}M" --vm-method write64 --vm-keep --timeout "${duration}s" --metrics-brief
  else
    python3 -c "
import time
a = bytearray(${mem_mb} * 1024 * 1024)
end = time.time() + ${duration}
while time.time() < end:
    a[:4096] = b'x' * 4096
" || sleep "${duration}"
  fi
}

mem_stress_random() {
  local duration="$1"
  local size_gb="$2"
  python3 -c "
import mmap, random, time
size = ${size_gb} * 1024 * 1024 * 1024
m = mmap.mmap(-1, size)
end = time.time() + ${duration}
while time.time() < end:
    pos = random.randint(0, max(0, size - 1000))
    m[pos:pos+1000] = b'x' * 1000
" || sleep "${duration}"
}

mem_stress_sparse() {
  local duration="$1"
  local size_gb="$2"
  if [[ ${HAS_STRESS_NG} -eq 1 ]]; then
    stress-ng --vm 1 --vm-bytes "$((size_gb * 1024))M" --vm-method flip --timeout "${duration}s" --metrics-brief
  else
    python3 -c "
import time
end = time.time() + ${duration}
while time.time() < end:
    time.sleep(0.1)
" || sleep "${duration}"
  fi
}

mem_stress_allocation() {
  local duration="$1"
  local size_gb="$2"
  python3 -c "
import time
chunk_size = max(1, int((${size_gb} * 1024 * 1024 * 1024) / 100))
end = time.time() + ${duration}
while time.time() < end:
    chunks = [bytearray(chunk_size) for _ in range(100)]
    del chunks
" || sleep "${duration}"
}

mem_stress_sweep() {
  local duration="$1"
  local max_size_gb="$2"
  local size_gb=1
  while [[ ${size_gb} -le ${max_size_gb} ]]; do
    mem_stress_bandwidth "${duration}" "${size_gb}"
    [[ ${size_gb} -ge ${max_size_gb} ]] && break
    size_gb=$((size_gb * 2))
    [[ ${size_gb} -gt ${max_size_gb} ]] && size_gb=${max_size_gb}
  done
}

io_stress_seq_read() {
  local duration="$1"
  local size_mb="${2:-1000}"
  command -v fio >/dev/null 2>&1 || { sleep "${duration}"; return; }
  local fio_file="${IO_DIR}/fio_seq_read_$$"
  fio --name=seq_read --ioengine=libaio --iodepth=32 --rw=read --bs=128k --direct="${IO_DIRECT}" --size="${size_mb}M" --runtime="${duration}s" --time_based --group_reporting --filename="${fio_file}"
  rm -f "${fio_file}"
}

io_stress_seq_write() {
  local duration="$1"
  local size_mb="${2:-1000}"
  command -v fio >/dev/null 2>&1 || { sleep "${duration}"; return; }
  local fio_file="${IO_DIR}/fio_seq_write_$$"
  fio --name=seq_write --ioengine=libaio --iodepth=32 --rw=write --bs=128k --direct="${IO_DIRECT}" --size="${size_mb}M" --runtime="${duration}s" --time_based --group_reporting --filename="${fio_file}"
  rm -f "${fio_file}"
}

io_stress_random() {
  local duration="$1"
  local size_mb="${2:-1000}"
  command -v fio >/dev/null 2>&1 || { sleep "${duration}"; return; }
  local fio_file="${IO_DIR}/fio_random_$$"
  fio --name=random_io --ioengine=libaio --iodepth=16 --rw=randrw --rwmixread=50 --bs=4k --direct="${IO_DIRECT}" --size="${size_mb}M" --runtime="${duration}s" --time_based --group_reporting --filename="${fio_file}"
  rm -f "${fio_file}"
}

net_stress_bandwidth() {
  local duration="$1"
  local requested_mbps="${2:-max}"
  local target_mbps
  target_mbps=$(resolve_net_mbps "${requested_mbps}")
  [[ "${target_mbps}" =~ ^[0-9]+$ && "${target_mbps}" -gt 0 ]] || return 1
  iperf3 -c "${NET_HOST}" -t "${duration}" -b "${target_mbps}M"
}

net_stress_bursty() {
  local duration="$1"
  local requested_mbps="${2:-max}"
  local target_mbps
  local end_time now remaining burst_duration
  target_mbps=$(resolve_net_mbps "${requested_mbps}")
  [[ "${target_mbps}" =~ ^[0-9]+$ && "${target_mbps}" -gt 0 ]] || return 1
  end_time=$(($(date +%s) + duration))
  while [[ $(date +%s) -lt ${end_time} ]]; do
    now=$(date +%s)
    remaining=$((end_time - now))
    [[ ${remaining} -le 0 ]] && break
    burst_duration=3
    [[ ${remaining} -lt ${burst_duration} ]] && burst_duration=${remaining}
    iperf3 -c "${NET_HOST}" -t "${burst_duration}" -b "${target_mbps}M" || return 1
    now=$(date +%s)
    remaining=$((end_time - now))
    [[ ${remaining} -le 0 ]] && break
    sleep $((remaining < 2 ? remaining : 2))
  done
}

execute_workload_phase() {
  local phase_spec="$1"
  IFS=':' read -ra parts <<< "${phase_spec}"
  local type="${parts[0]}"
  local mode="${parts[1]:-idle}"
  local duration="${parts[2]:-${parts[1]:-0}}"
  local params_str="${parts[3]:-}"

  if [[ "${type}" == "idle" && ${#parts[@]} -eq 2 ]]; then
    mode="idle"
    duration="${parts[1]}"
  fi

  declare -A params=()
  parse_phase_params "${params_str}" params
  local cores="${params[cores]:-${STRESS_CPUS:-${MAX_CORES}}}"
  local mem_size="${params[size]:-8}"
  local io_mbs="${params[io]:-1000}"
  local net_mbps="${params[net]:-max}"

  if [[ "${cores}" =~ ^[0-9]+$ && "${cores}" -gt "${MAX_CORES}" ]]; then
    cores="${MAX_CORES}"
  fi
  if [[ "${mem_size}" =~ ^[0-9]+$ && "${mem_size}" -gt "${MAX_MEM_GB}" ]]; then
    mem_size="${MAX_MEM_GB}"
  fi

  case "${type}" in
    cpu)
      case "${mode}" in
        scalar) cpu_stress_scalar "${duration}" "${cores}" ;;
        avx) cpu_stress_avx "${duration}" "${cores}" ;;
        mixed) cpu_stress_mixed "${duration}" "${cores}" ;;
        burst) cpu_stress_burst "${duration}" "${cores}" ;;
        sweep) cpu_stress_sweep "${duration}" "${cores}" ;;
        *) echo "Unknown CPU mode: ${mode}" >&2; return 1 ;;
      esac
      ;;
    mem)
      case "${mode}" in
        bandwidth) mem_stress_bandwidth "${duration}" "${mem_size}" ;;
        random) mem_stress_random "${duration}" "${mem_size}" ;;
        sparse) mem_stress_sparse "${duration}" "${mem_size}" ;;
        allocation) mem_stress_allocation "${duration}" "${mem_size}" ;;
        sweep) mem_stress_sweep "${duration}" "${mem_size}" ;;
        *) echo "Unknown memory mode: ${mode}" >&2; return 1 ;;
      esac
      ;;
    io)
      case "${mode}" in
        seq-read) io_stress_seq_read "${duration}" "${io_mbs}" ;;
        seq-write) io_stress_seq_write "${duration}" "${io_mbs}" ;;
        random) io_stress_random "${duration}" "${io_mbs}" ;;
        *) echo "Unknown I/O mode: ${mode}" >&2; return 1 ;;
      esac
      ;;
    net)
      case "${mode}" in
        bandwidth) net_stress_bandwidth "${duration}" "${net_mbps}" ;;
        bursty) net_stress_bursty "${duration}" "${net_mbps}" ;;
        *) echo "Unknown network mode: ${mode}" >&2; return 1 ;;
      esac
      ;;
    idle)
      sleep "${duration}"
      ;;
    *)
      echo "Unknown phase type: ${type}" >&2
      return 1
      ;;
  esac
}

parallel_workload_phase() {
  local segment_id="$1"
  local phase_spec="$2"
  local segment_dir="${SEGMENTS_ROOT}/${segment_id}"
  local start_time
  local stop_time
  local status="ok"
  local subphase
  local idx=0
  local -a pids=()

  split_parallel_phase "${phase_spec}" || return 1
  mkdir -p "${segment_dir}"
  echo "  [workload:parallel] ${phase_spec}"
  start_time="$(timestamp)"

  for subphase in "${SPLIT_PARALLEL_PHASE_RESULT[@]}"; do
    [[ -z "${subphase}" ]] && continue
    idx=$((idx + 1))
    (
      execute_workload_phase "${subphase}"
    ) > "${segment_dir}/subphase${idx}.log" 2>&1 &
    pids+=("$!")
  done

  if [[ ${#pids[@]} -eq 0 ]]; then
    status="failed"
  else
    local pid
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        status="failed"
      fi
    done
  fi

  stop_time="$(timestamp)"
  record_manifest "${segment_id}" "parallel" "${phase_spec}" "${start_time}" "${stop_time}" "${status}" "${segment_dir}" ""
  [[ "${status}" == "ok" ]]
}

workload_phase() {
  local segment_id="$1"
  local phase_spec="$2"
  local type
  local segment_dir="${SEGMENTS_ROOT}/${segment_id}"
  local start_time
  local stop_time
  local status="ok"

  type="$(phase_type "${phase_spec}")"
  if [[ "${type}" == "parallel" ]]; then
    parallel_workload_phase "${segment_id}" "${phase_spec}"
    return $?
  fi

  mkdir -p "${segment_dir}"
  echo "  [workload] ${phase_spec}"
  start_time="$(timestamp)"
  if ! execute_workload_phase "${phase_spec}" > "${segment_dir}/workload.log" 2>&1; then
    status="failed"
  fi
  stop_time="$(timestamp)"

  record_manifest "${segment_id}" "${type}" "${phase_spec}" "${start_time}" "${stop_time}" "${status}" "${segment_dir}" ""
  [[ "${status}" == "ok" ]]
}

run_workload_sequence() {
  local label="$1"
  local idx
  [[ ${#WORKLOAD_PHASES[@]} -eq 0 ]] && return
  for idx in "${!WORKLOAD_PHASES[@]}"; do
    workload_phase "workload-${label}-phase${idx}" "${WORKLOAD_PHASES[$idx]}"
  done
}

build_extra_configs() {
  local segment_dir="$1"
  local cfg
  EXTRA_CONFIG_ARGS=()

  if [[ -n "${MAX_CPUS}" ]]; then
    local cpu_config="${segment_dir}/cpu_override.config"
    printf 'process { cpus = %s }\n' "${MAX_CPUS}" > "${cpu_config}"
    EXTRA_CONFIG_ARGS+=(-c "${cpu_config}")
  fi

  if [[ -f "${CO2_CONFIG}" ]]; then
    EXTRA_CONFIG_ARGS+=(-c "${CO2_CONFIG}")
  fi

  for cfg in "${NEXTFLOW_CONFIG_FILES[@]}"; do
    EXTRA_CONFIG_ARGS+=(-c "${cfg}")
  done
}

cleanup_after_pipeline() {
  local outdir="$1"
  local segment_dir="$2"
  local co2file

  [[ -d "${outdir}" ]] && rm -rf "${outdir}"
  if [[ -d "${REPO_ROOT}/work" ]]; then
    log_info "Cleaning Nextflow work dir: ${REPO_ROOT}/work"
    rm -rf "${REPO_ROOT}/work"
  fi

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

  if [[ "${spec}" == *" -profile "* ]] || [[ "${spec}" == -profile* ]] || [[ "${spec}" == *" --outdir "* ]] || [[ "${spec}" == *" -c "* ]] || [[ "${spec}" == -c* ]] || [[ "${spec}" == *" --config "* ]]; then
    fail "Pipeline spec must not contain -profile, --outdir, -c, or --config: ${spec}"
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

  if [[ ${#WORKLOAD_PHASES[@]} -gt 0 ]]; then
    run_workload_sequence "after-${idx}-rep${rep}"
  else
    local idle_seconds
    idle_seconds="$(random_idle)"
    (( idle_seconds > 0 )) && idle_phase "idle-between-${idx}-rep${rep}" "${idle_seconds}"
  fi
}

main() {
  parse_args "$@"
  load_pipelines
  [[ -z "${STRESS_CPUS}" ]] && STRESS_CPUS="$(nproc)"
  if [[ ${#WORKLOAD_PHASES[@]} -eq 0 && ${STRESS_DURATION} -gt 0 ]]; then
    WORKLOAD_PHASES=("cpu:scalar:${STRESS_DURATION}:cores=${STRESS_CPUS}")
  fi
  setup_session
  validate_config

  if (( INITIAL_IDLE > 0 )); then
    echo "Initial idle: ${INITIAL_IDLE}s"
    idle_phase "idle-before" "${INITIAL_IDLE}"
  fi

  if [[ ${#WORKLOAD_PHASES[@]} -gt 0 ]]; then
    run_workload_sequence "before"
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

  if [[ ${#WORKLOAD_PHASES[@]} -gt 0 ]]; then
    run_workload_sequence "after"
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
