#!/bin/bash
#
# rapl_adversarial_workload.sh
#
# Adversarial workload generator designed to expose RAPL energy accounting weaknesses
# by stressing platform components outside the RAPL perimeter, creating burst aliasing,
# thermal hysteresis, and mixed multi-domain loads.
#
# Usage:
#   ./rapl_adversarial_workload.sh [OPTIONS]
#
# Options:
#   --duration SECONDS        Total runtime (default: 300)
#   --phase-time SECONDS      Duration per phase (default: 20)
#   --threads N               Thread count (default: all CPUs)
#   --mem-gb N                Memory size for allocation (default: 8)
#   --nvme-path PATH          Directory for I/O stress (default: /tmp)
#   --network-host HOST       iperf3 target host (optional)
#   --enable-gpu              Enable GPU stress if available
#   --burst-on-us N           Burst on duration microseconds (default: 500)
#   --burst-off-us N          Burst off duration microseconds (default: 500)
#   --dry-run                 Print phases without execution
#   --help                    Show this help
#

set -euo pipefail

# ========================= Configuration & Defaults =========================

DURATION=300
PHASE_TIME=20
THREADS=$(nproc)
MEM_GB=8
NVME_PATH="/tmp"
NETWORK_HOST=""
ENABLE_GPU=0
BURST_ON_US=500
BURST_OFF_US=500
DRY_RUN=0

# ========================= Argument Parsing =========================

show_help() {
    head -n 20 "$0" | grep "^#" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --duration) DURATION="$2"; shift 2 ;;
        --phase-time) PHASE_TIME="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --mem-gb) MEM_GB="$2"; shift 2 ;;
        --nvme-path) NVME_PATH="$2"; shift 2 ;;
        --network-host) NETWORK_HOST="$2"; shift 2 ;;
        --enable-gpu) ENABLE_GPU=1; shift ;;
        --burst-on-us) BURST_ON_US="$2"; shift 2 ;;
        --burst-off-us) BURST_OFF_US="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help) show_help ;;
        *) echo "Unknown option: $1"; show_help ;;
    esac
done

# ========================= Tool Detection =========================

HAS_STRESS_NG=$(command -v stress-ng >/dev/null 2>&1 && echo 1 || echo 0)
HAS_FIO=$(command -v fio >/dev/null 2>&1 && echo 1 || echo 0)
HAS_IPERF3=$(command -v iperf3 >/dev/null 2>&1 && echo 1 || echo 0)
HAS_NVIDIA_SMI=$(command -v nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0)
HAS_TASKSET=$(command -v taskset >/dev/null 2>&1 && echo 1 || echo 0)
HAS_GCC=$(command -v gcc >/dev/null 2>&1 && echo 1 || echo 0)

# Detect physical core count (for SMT sweep)
PHYSICAL_CORES=$THREADS
if [ -f /sys/devices/system/cpu/cpu0/topology/thread_siblings_list ]; then
    PHYSICAL_CORES=$(lscpu -p=Core | grep -v "^#" | sort -u | wc -l)
fi

echo "=== RAPL Adversarial Workload Generator ==="
echo "Duration: ${DURATION}s | Phase time: ${PHASE_TIME}s | Threads: ${THREADS} | Physical cores: ${PHYSICAL_CORES}"
echo "Tools: stress-ng=$HAS_STRESS_NG fio=$HAS_FIO iperf3=$HAS_IPERF3 nvidia-smi=$HAS_NVIDIA_SMI gcc=$HAS_GCC"
echo "NVMe path: $NVME_PATH | Network: ${NETWORK_HOST:-none} | GPU: $ENABLE_GPU"
echo "============================================"

# ========================= Utility Functions =========================

timestamp_ns() {
    date +%s%N
}

phase_marker() {
    local event_type=$1
    local phase_name=$2
    local ts=$(timestamp_ns)
    echo "PHASE_${event_type} ${phase_name} ${ts}"
}

run_phase() {
    local phase_name=$1
    local phase_func=$2
    local duration=${3:-$PHASE_TIME}

    phase_marker "START" "$phase_name"

    if [ $DRY_RUN -eq 1 ]; then
        echo "  [DRY RUN] Would execute: $phase_name for ${duration}s"
        sleep 1
    else
        $phase_func "$duration"
    fi

    phase_marker "END" "$phase_name"
}

cleanup_background_jobs() {
    jobs -p | xargs -r kill -9 2>/dev/null || true
    wait 2>/dev/null || true
}

# ========================= Phase Implementations =========================

# Phase 1: Idle baseline
# Purpose: Expose static platform power (fans, BMC, NICs) invisible to RAPL
phase_idle() {
    local duration=$1
    sleep "$duration"
}

# Phase 2: CPU scalar workload (integer ops, no AVX)
# Purpose: Baseline CPU load without AVX frequency throttling
phase_cpu_scalar() {
    local duration=$1
    if [ $HAS_STRESS_NG -eq 1 ]; then
        timeout "$duration" stress-ng --cpu "$THREADS" --cpu-method int64 --cpu-ops 0 >/dev/null 2>&1 || true
    else
        # Fallback: bash busy loop
        for i in $(seq 1 "$THREADS"); do
            (x=0; while true; do x=$((x + 1)); done) &
        done
        sleep "$duration"
        cleanup_background_jobs
    fi
}

# Phase 3: CPU AVX/FMA thermal heater
# Purpose: Max CPU power, AVX frequency transitions, fan ramp
phase_cpu_avx() {
    local duration=$1
    if [ $HAS_STRESS_NG -eq 1 ]; then
        # ackermann is AVX-heavy
        timeout "$duration" stress-ng --cpu "$THREADS" --cpu-method ackermann --cpu-ops 0 >/dev/null 2>&1 || true
    elif [ $HAS_GCC -eq 1 ]; then
        # Compile inline AVX burner
        local src="/tmp/avx_burner_$$.c"
        local bin="/tmp/avx_burner_$$"
        cat > "$src" <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
void* burn(void* arg) {
    double a[1024], b[1024], c[1024];
    for(int i=0; i<1024; i++) { a[i]=i; b[i]=i*2; c[i]=0; }
    time_t end = time(NULL) + *(int*)arg;
    while(time(NULL) < end) {
        for(int i=0; i<1024; i++) c[i] = a[i]*b[i] + c[i];
    }
    return NULL;
}
int main(int argc, char** argv) {
    int dur = atoi(argv[1]);
    int threads = atoi(argv[2]);
    pthread_t t[threads];
    for(int i=0; i<threads; i++) pthread_create(&t[i], NULL, burn, &dur);
    for(int i=0; i<threads; i++) pthread_join(t[i], NULL);
}
EOF
        gcc -O3 -march=native -pthread -o "$bin" "$src" 2>/dev/null
        "$bin" "$duration" "$THREADS" 2>/dev/null || true
        rm -f "$src" "$bin"
    else
        phase_cpu_scalar "$duration"
    fi
}

# Phase 4: Memory bandwidth (STREAM-like)
# Purpose: DRAM traffic, uncore power, DRAM RAPL domain stress
phase_memory_bw() {
    local duration=$1
    if [ $HAS_STRESS_NG -eq 1 ]; then
        timeout "$duration" stress-ng --stream "$THREADS" --stream-l3-size 0 >/dev/null 2>&1 || true
    else
        # Fallback: allocate and copy memory
        local mem_mb=$((MEM_GB * 1024))
        timeout "$duration" stress-ng --vm "$THREADS" --vm-bytes "${mem_mb}M" --vm-method all >/dev/null 2>&1 || true
    fi
}

# Phase 5: Memory sparse touch
# Purpose: Low CPU load, DRAM retention, expose DRAM RAPL bias at low activity
phase_memory_sparse() {
    local duration=$1
    # Allocate memory but touch rarely
    if [ $HAS_STRESS_NG -eq 1 ]; then
        local mem_mb=$((MEM_GB * 1024))
        timeout "$duration" stress-ng --vm 1 --vm-bytes "${mem_mb}M" --vm-method flip --vm-ops 1000 >/dev/null 2>&1 || true
    else
        python3 -c "
import time, mmap
size = $MEM_GB * 1024 * 1024 * 1024
m = mmap.mmap(-1, size)
end = time.time() + $duration
while time.time() < end:
    m[0] = 1
    time.sleep(0.1)
" 2>/dev/null || sleep "$duration"
    fi
}

# Phase 6: Sequential I/O (NVMe writes)
# Purpose: NVMe controller power, PCIe lanes, off-die platform stress
phase_io_sequential() {
    local duration=$1
    local test_file="$NVME_PATH/rapl_iotest_$$"

    if [ $HAS_FIO -eq 1 ]; then
        timeout "$duration" fio --name=seqwrite \
            --rw=write --bs=1M --size=10G \
            --filename="$test_file" \
            --direct=1 --numjobs=4 --group_reporting \
            --time_based --runtime="$duration" >/dev/null 2>&1 || true
    else
        # Fallback: dd
        for i in $(seq 1 4); do
            timeout "$duration" dd if=/dev/zero of="${test_file}_${i}" bs=1M count=10000 oflag=direct 2>/dev/null &
        done
        wait
    fi

    rm -f "$test_file"* 2>/dev/null || true
}

# Phase 7: Random I/O (IOPS)
# Purpose: NVMe random access, controller overhead, different power profile than sequential
phase_io_random() {
    local duration=$1
    local test_file="$NVME_PATH/rapl_iotest_$$"

    if [ $HAS_FIO -eq 1 ]; then
        timeout "$duration" fio --name=randwrite \
            --rw=randwrite --bs=4k --size=2G \
            --filename="$test_file" \
            --direct=1 --numjobs=8 --group_reporting \
            --time_based --runtime="$duration" --iodepth=32 >/dev/null 2>&1 || true
    else
        # Fallback: sequential as proxy
        phase_io_sequential "$duration"
    fi

    rm -f "$test_file"* 2>/dev/null || true
}

# Phase 8: Network stress
# Purpose: NIC power, PCIe, chipset - completely outside RAPL
phase_network() {
    local duration=$1

    if [ -z "$NETWORK_HOST" ]; then
        echo "  [SKIP] No network host specified"
        sleep "$duration"
        return
    fi

    if [ $HAS_IPERF3 -eq 1 ]; then
        timeout "$duration" iperf3 -c "$NETWORK_HOST" -t "$duration" -P 4 >/dev/null 2>&1 || true
    else
        echo "  [SKIP] iperf3 not available"
        sleep "$duration"
    fi
}

# Phase 9: GPU stress
# Purpose: GPU power outside RAPL perimeter (if discrete GPU)
phase_gpu() {
    local duration=$1

    if [ $ENABLE_GPU -eq 0 ]; then
        echo "  [SKIP] GPU stress disabled"
        sleep "$duration"
        return
    fi

    if [ $HAS_NVIDIA_SMI -eq 0 ]; then
        echo "  [SKIP] nvidia-smi not available"
        sleep "$duration"
        return
    fi

    # Try to run GPU stress if available
    if command -v gpu-burn >/dev/null 2>&1; then
        timeout "$duration" gpu-burn "$duration" >/dev/null 2>&1 || true
    elif [ $HAS_STRESS_NG -eq 1 ]; then
        timeout "$duration" stress-ng --cuda 0 --cuda-ops 0 >/dev/null 2>&1 || true
    else
        echo "  [SKIP] No GPU stress tool available"
        sleep "$duration"
    fi
}

# ========================= Mixed Workload Phases =========================

# Phase 10: Mixed CPU + Memory
# Purpose: Combined CPU and DRAM power, stress both RAPL domains simultaneously
phase_mixed_compute_mem() {
    local duration=$1
    local half=$((THREADS / 2))

    if [ $HAS_STRESS_NG -eq 1 ]; then
        timeout "$duration" stress-ng \
            --cpu "$half" --cpu-method ackermann \
            --stream "$half" --stream-l3-size 0 \
            >/dev/null 2>&1 || true
    else
        phase_cpu_avx "$duration" &
        phase_memory_bw "$duration" &
        wait
    fi
}

# Phase 11: Mixed CPU + I/O
# Purpose: AVX heat + NVMe = max PSU load and VRM losses
phase_mixed_cpu_io() {
    local duration=$1

    phase_cpu_avx "$duration" &
    phase_io_sequential "$duration" &
    wait
}

# Phase 12: Mixed all domains
# Purpose: CPU + memory + I/O + network simultaneously - absolute max platform power
phase_mixed_all_domains() {
    local duration=$1
    local quarter=$((THREADS / 4))

    if [ $HAS_STRESS_NG -eq 1 ]; then
        timeout "$duration" stress-ng \
            --cpu "$quarter" --cpu-method ackermann \
            --stream "$quarter" \
            >/dev/null 2>&1 &
    else
        phase_cpu_avx "$duration" &
    fi

    phase_io_random "$duration" &
    [ -n "$NETWORK_HOST" ] && phase_network "$duration" &

    wait
}

# Phase 13: Mixed unbalanced (asymmetric core loading)
# Purpose: Half cores AVX, half idle - test per-core RAPL modeling
phase_mixed_unbalanced() {
    local duration=$1
    local half=$((THREADS / 2))

    if [ $HAS_STRESS_NG -eq 1 ] && [ $HAS_TASKSET -eq 1 ]; then
        # Pin to first half of cores
        local core_list="0-$((half - 1))"
        timeout "$duration" taskset -c "$core_list" stress-ng --cpu "$half" --cpu-method ackermann >/dev/null 2>&1 || true
    else
        # Fallback: just run half threads
        if [ $HAS_STRESS_NG -eq 1 ]; then
            timeout "$duration" stress-ng --cpu "$half" --cpu-method ackermann >/dev/null 2>&1 || true
        else
            phase_cpu_avx "$duration"
        fi
    fi
}

# Phase 14: Platform max (everything except CPU)
# Purpose: Maximize non-RAPL platform components: I/O + network + GPU
phase_mixed_platform_max() {
    local duration=$1

    phase_io_random "$duration" &
    [ -n "$NETWORK_HOST" ] && phase_network "$duration" &
    [ $ENABLE_GPU -eq 1 ] && phase_gpu "$duration" &

    wait
}

# Phase 15: Thermal ramp (gradual increase)
# Purpose: Gradual power increase to expose thermal lag and RAPL tracking delay
phase_mixed_thermal_ramp() {
    local duration=$1
    local step=$((duration / 4))

    [ $step -lt 1 ] && step=1

    # Gradual ramp: idle → scalar → AVX → AVX+mem+I/O
    timeout "$step" stress-ng --cpu 1 --cpu-method int64 >/dev/null 2>&1 || true
    timeout "$step" stress-ng --cpu "$((THREADS / 2))" --cpu-method int64 >/dev/null 2>&1 || true
    timeout "$step" stress-ng --cpu "$THREADS" --cpu-method ackermann >/dev/null 2>&1 || true

    # Final spike
    if [ $HAS_STRESS_NG -eq 1 ]; then
        timeout "$step" stress-ng \
            --cpu "$THREADS" --cpu-method ackermann \
            --stream "$THREADS" \
            >/dev/null 2>&1 &
    fi
    phase_io_sequential "$step" &
    wait
}

# ========================= Adversarial/Edge Case Phases =========================

# Phase 16: Burst micro (sub-millisecond oscillation)
# Purpose: RAPL sampling aliasing - bursts faster than RAPL update rate (~1ms)
phase_burst_micro() {
    local duration=$1

    if [ $HAS_GCC -eq 1 ]; then
        # Compile burst oscillator with precise timing
        local src="/tmp/burst_micro_$$.c"
        local bin="/tmp/burst_micro_$$"
        cat > "$src" <<EOF
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#define NSEC_PER_SEC 1000000000L

void* burst_worker(void* arg) {
    int dur = *(int*)arg;
    long on_ns = ${BURST_ON_US} * 1000L;
    long off_ns = ${BURST_OFF_US} * 1000L;

    struct timespec start, now, sleep_time;
    clock_gettime(CLOCK_MONOTONIC, &start);

    while(1) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        long elapsed = (now.tv_sec - start.tv_sec) * NSEC_PER_SEC + (now.tv_nsec - start.tv_nsec);
        if(elapsed / NSEC_PER_SEC >= dur) break;

        // Busy spin for on_ns
        struct timespec burst_start;
        clock_gettime(CLOCK_MONOTONIC, &burst_start);
        volatile double x = 1.0;
        while(1) {
            clock_gettime(CLOCK_MONOTONIC, &now);
            long burst_elapsed = (now.tv_sec - burst_start.tv_sec) * NSEC_PER_SEC + (now.tv_nsec - burst_start.tv_nsec);
            if(burst_elapsed >= on_ns) break;
            for(int i=0; i<100; i++) x = x * 1.1 + 0.9;
        }

        // Sleep for off_ns
        sleep_time.tv_sec = 0;
        sleep_time.tv_nsec = off_ns;
        nanosleep(&sleep_time, NULL);
    }
    return NULL;
}

int main(int argc, char** argv) {
    int dur = atoi(argv[1]);
    int threads = atoi(argv[2]);
    pthread_t t[threads];
    for(int i=0; i<threads; i++) pthread_create(&t[i], NULL, burst_worker, &dur);
    for(int i=0; i<threads; i++) pthread_join(t[i], NULL);
}
EOF
        gcc -O3 -pthread -o "$bin" "$src" 2>/dev/null
        "$bin" "$duration" "$THREADS" 2>/dev/null || true
        rm -f "$src" "$bin"
    else
        # Fallback: Python burst
        python3 -c "
import time, multiprocessing, os
def burst_worker(dur):
    end = time.time() + dur
    on_s = ${BURST_ON_US} / 1e6
    off_s = ${BURST_OFF_US} / 1e6
    while time.time() < end:
        start = time.time()
        x = 1.0
        while time.time() - start < on_s:
            x = x * 1.1 + 0.9
        time.sleep(off_s)

if __name__ == '__main__':
    procs = [multiprocessing.Process(target=burst_worker, args=($duration,)) for _ in range($THREADS)]
    for p in procs: p.start()
    for p in procs: p.join()
" 2>/dev/null || sleep "$duration"
    fi
}

# Phase 17: Burst milli (10ms oscillation)
# Purpose: Frequency scaling thrashing, turbo boost cycling
phase_burst_milli() {
    local duration=$1
    local on_ms=10
    local off_ms=10

    # Override burst timings for this phase
    BURST_ON_US=$((on_ms * 1000))
    BURST_OFF_US=$((off_ms * 1000))

    phase_burst_micro "$duration"
}

# Phase 18: SMT sweep
# Purpose: Compare 1 thread vs N/2 (physical) vs N (hyperthreaded) - expose SMT power modeling issues
phase_smt_sweep() {
    local duration=$1
    local step=$((duration / 3))

    [ $step -lt 1 ] && step=1

    # 1 thread
    if [ $HAS_STRESS_NG -eq 1 ]; then
        timeout "$step" stress-ng --cpu 1 --cpu-method ackermann >/dev/null 2>&1 || true
        # Physical cores only
        timeout "$step" stress-ng --cpu "$PHYSICAL_CORES" --cpu-method ackermann >/dev/null 2>&1 || true
        # All logical cores (hyperthreading)
        timeout "$step" stress-ng --cpu "$THREADS" --cpu-method ackermann >/dev/null 2>&1 || true
    else
        sleep "$duration"
    fi
}

# Phase 19: NUMA imbalance (if multi-socket)
# Purpose: Pin all work to socket 0, leaving socket 1 idle
phase_numa_imbalance() {
    local duration=$1

    # Check if NUMA is available
    if [ ! -d /sys/devices/system/node/node1 ]; then
        echo "  [SKIP] Single-socket system, no NUMA imbalance possible"
        sleep "$duration"
        return
    fi

    if command -v numactl >/dev/null 2>&1 && [ $HAS_STRESS_NG -eq 1 ]; then
        timeout "$duration" numactl --cpunodebind=0 --membind=0 \
            stress-ng --cpu "$THREADS" --cpu-method ackermann >/dev/null 2>&1 || true
    else
        echo "  [SKIP] numactl not available"
        sleep "$duration"
    fi
}

# Phase 20: Thermal hysteresis
# Purpose: AVX blast followed by immediate idle - fans keep spinning, wall meter stays high, RAPL drops instantly
phase_thermal_hysteresis() {
    local duration=$1
    local heat_time=$((duration / 2))
    local cool_time=$((duration - heat_time))

    # Heat up
    phase_cpu_avx "$heat_time"

    # Immediate idle while fans spin down
    sleep "$cool_time"
}

# Phase 21: Power virus
# Purpose: Absolute maximum platform power - AVX512 + memory + I/O simultaneously
phase_power_virus() {
    local duration=$1

    if [ $HAS_STRESS_NG -eq 1 ]; then
        timeout "$duration" stress-ng \
            --cpu "$THREADS" --cpu-method ackermann \
            --stream "$THREADS" --stream-l3-size 0 \
            --vm "$THREADS" --vm-bytes 128M \
            >/dev/null 2>&1 &
    else
        phase_cpu_avx "$duration" &
    fi

    phase_io_random "$duration" &
    wait
}

# ========================= Main Phase Schedule =========================

PHASES=(
    # Core single-domain phases
    "idle:phase_idle"
    "cpu_scalar:phase_cpu_scalar"
    "cpu_avx:phase_cpu_avx"
    "memory_bw:phase_memory_bw"
    "memory_sparse:phase_memory_sparse"
    "io_sequential:phase_io_sequential"
    "io_random:phase_io_random"

    # Mixed workload phases
    "mixed_compute_mem:phase_mixed_compute_mem"
    "mixed_cpu_io:phase_mixed_cpu_io"
    "mixed_all_domains:phase_mixed_all_domains"
    "mixed_unbalanced:phase_mixed_unbalanced"
    "mixed_platform_max:phase_mixed_platform_max"
    "mixed_thermal_ramp:phase_mixed_thermal_ramp"

    # Adversarial/edge case phases
    "burst_micro:phase_burst_micro"
    "burst_milli:phase_burst_milli"
    "smt_sweep:phase_smt_sweep"
    "numa_imbalance:phase_numa_imbalance"
    "thermal_hysteresis:phase_thermal_hysteresis"
    "power_virus:phase_power_virus"
)

# Optional phases (only if enabled)
[ -n "$NETWORK_HOST" ] && PHASES+=("network:phase_network")
[ $ENABLE_GPU -eq 1 ] && PHASES+=("gpu:phase_gpu")

# ========================= Signal Handling & Cleanup =========================

trap cleanup_background_jobs EXIT INT TERM

# ========================= Main Execution Loop =========================

START_TIME=$(date +%s)
PHASE_INDEX=0

echo ""
echo "=== Starting workload execution at $(date) ==="
echo ""

while true; do
    ELAPSED=$(($(date +%s) - START_TIME))

    if [ $ELAPSED -ge $DURATION ]; then
        echo ""
        echo "=== Duration reached, stopping ==="
        break
    fi

    # Get current phase
    PHASE_ENTRY="${PHASES[$((PHASE_INDEX % ${#PHASES[@]}))]}"
    PHASE_NAME="${PHASE_ENTRY%%:*}"
    PHASE_FUNC="${PHASE_ENTRY##*:}"

    # Calculate remaining time
    REMAINING=$((DURATION - ELAPSED))
    THIS_PHASE_TIME=$PHASE_TIME
    [ $REMAINING -lt $PHASE_TIME ] && THIS_PHASE_TIME=$REMAINING

    # Execute phase
    run_phase "$PHASE_NAME" "$PHASE_FUNC" "$THIS_PHASE_TIME"

    PHASE_INDEX=$((PHASE_INDEX + 1))
done

echo ""
echo "=== Workload complete at $(date) ==="
echo "Total phases executed: $PHASE_INDEX"
echo ""
