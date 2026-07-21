#!/bin/bash

benchmarks=(
  # CPU-Intensive
  "pts/build-linux-kernel"
  "pts/compress-7zip"
  "pts/x264"
  "pts/blake2"
  #"pts/coremark"
  "pts/c-ray"
  "pts/gmpbench"
  #"pts/primesieve"

  # Memory-Intensive
  #"pts/stream"
  "pts/mbw"
  "pts/tinymembench"
  #"pts/pmbench"

  # IO-Intensive
  #"pts/dbench"
  "pts/compilebench"
  "pts/fs-mark"
  #"pts/postmark"

  # Server/Database
  "pts/apache"
  #"pts/memcached"
  "pts/redis"
  "pts/mysqlslap"
  "pts/cassandra"
  #"pts/nginx"

  # Mixed/Stress
  #"pts/sysbench"
  #"pts/stress-ng"
  "pts/byte"
  "pts/hackbench"
)

iterations=20


for ((i=1; i<=iterations; i++)); do
  benchmark=${benchmarks[$RANDOM % ${#benchmarks[@]}]}
  echo "[$(date)] Running benchmark: $benchmark"

  phoronix-test-suite batch-run $benchmark

  wait_minutes=$((RANDOM % 10 + 1))
  echo "[$(date)] Sleeping for $wait_minutes minutes..."
  sleep "${wait_minutes}m"
done
