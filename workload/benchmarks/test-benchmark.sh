#!/usr/bin/env bash

TEST_BENCHMARKS=(
  # CPU‑Intensive
  "pts/coremark"
  "pts/primesieve"

  # Memory‑Intensive
  "pts/stream"
  "pts/pmbench"

  # I/O‑Intensive
  "pts/dbench"
  "pts/postmark"

  # Server/DB
  "pts/memcached"
  "pts/nginx"

  # Mixed/Stress
  "pts/sysbench"
  "pts/stress-ng"
)

SLEEP_BETWEEN=60  # seconds

echo "Starting test sequence at $(date)"
for bm in "${TEST_BENCHMARKS[@]}"; do
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: $bm"
  phoronix-test-suite batch-run "$bm" >/dev/null 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $bm"
  echo "→ Sleeping ${SLEEP_BETWEEN}s before next run"
  sleep "$SLEEP_BETWEEN"
done

echo
echo "All tests finished at $(date)"
