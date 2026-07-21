#!/bin/bash

INFLUX_URL="${INFLUX_URL:-http://localhost:8086}"
INFLUX_TOKEN="${INFLUX_TOKEN:-my-super-secret-auth-token}"
INFLUX_ORG="${INFLUX_ORG:-myorg}"
INFLUX_BUCKET="${INFLUX_BUCKET:-mybucket}"

write_marker() {
  local event="$1"
  local ts
  ts=$(date +%s%N)
  curl -s -o /dev/null -X POST "${INFLUX_URL}/api/v2/write?org=${INFLUX_ORG}&bucket=${INFLUX_BUCKET}&precision=ns" \
    -H "Authorization: Token ${INFLUX_TOKEN}" \
    -H "Content-Type: text/plain" \
    --data-binary "benchmark_marker event=\"${event}\" ${ts}"
}

# Benchmark groups
cpu_benchmarks=(
  "pts/build-linux-kernel" "pts/compress-7zip" "pts/x264"
  "pts/blake2" "pts/coremark" "pts/c-ray" "pts/gmpbench" "pts/primesieve"
)

mem_benchmarks=(
  "pts/stream" "pts/mbw" "pts/tinymembench" "pts/pmbench"
)

io_benchmarks=(
  "pts/dbench" "pts/compilebench" "pts/fs-mark" "pts/postmark"
)

server_benchmarks=(
  "pts/apache" "pts/memcached" "pts/redis" "pts/mysqlslap" "pts/cassandra" "pts/nginx"
)

mixed_benchmarks=(
  "pts/sysbench" "pts/stress-ng" "pts/byte" "pts/hackbench"
)

all_benchmarks=("${cpu_benchmarks[@]}" "${mem_benchmarks[@]}" "${io_benchmarks[@]}" "${server_benchmarks[@]}" "${mixed_benchmarks[@]}")

iterations=10
bench_timeout_min=4   # Longer sustained load for better signal
bench_timeout_max=6
idle_min=1.0          # Longer idle so model can learn true baseline
idle_max=2.0

run_benchmark() {
  bench="$1"
  safe_name=$(echo "$bench" | tr '/' '_')
  echo "[$(date)] ▶ Running: $bench"

  timeout_seconds=$(awk "BEGIN {print int(($RANDOM/32767)*($bench_timeout_max-$bench_timeout_min)*60 + ($bench_timeout_min*60))}")
  timeout "$timeout_seconds" phoronix-test-suite batch-run "$bench" \
    -y \
    > "/tmp/pts_${safe_name}.log" 2>&1

  result=$?
  if [[ $result -eq 124 ]]; then
    echo "[$(date)] ⏰ Timed out: $bench"
  else
    echo "[$(date)] ✔ Finished: $bench"
  fi
}

write_marker "start"
echo "[$(date)] ✅ Benchmark run started (marker written to InfluxDB)"

for ((i=1; i<=iterations; i++)); do
  echo "[$(date)] 🌀 Iteration $i"

  # Vary parallelism across full range (1 to 8) for wider load coverage
  count=$(( RANDOM % 8 + 1 ))

  # Every 5th iteration: run a single benchmark in isolation
  if (( i % 5 == 0 )); then
    count=1
    echo "[$(date)] 🔬 Single-benchmark isolation run"
  fi

  selected=()
  for ((j=0; j<count; j++)); do
    selected+=("${all_benchmarks[$RANDOM % ${#all_benchmarks[@]}]}")
  done

  echo "[$(date)] Parallelism: $count — Selected: ${selected[*]}"

  for bench in "${selected[@]}"; do
    run_benchmark "$bench" &
  done

  wait

  # Longer idle periods so the model learns true idle baseline
  idle_seconds=$(awk "BEGIN {print int(($RANDOM/32767)*($idle_max-$idle_min)*60 + ($idle_min*60))}")
  echo "[$(date)] 💤 Idle for $(awk "BEGIN {printf \"%.1f\", $idle_seconds/60}") minutes..."
  sleep "$idle_seconds"
done

write_marker "end"
echo "[$(date)] ✅ Benchmark run ended (marker written to InfluxDB)"
