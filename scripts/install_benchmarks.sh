#!/bin/bash

echo "[INFO] Updating APT and installing common system dependencies..."
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  p7zip-full \
  cmake \
  make \
  gcc \
  g++ \
  unzip \
  python3 \
  git \
  libtool \
  autoconf \
  automake \
  pkg-config \
  libssl-dev \
  zlib1g-dev \
  libjpeg-dev \
  libpng-dev \
  bzip2 \
  xz-utils \
  curl \
  wget \
  yasm \
  nasm \
  libgmp3-dev \
  libaio-dev \
  libpopt-dev \
  python2-minimal \
  libpcre3-dev \
  libevent-dev \
  default-jdk \
  libapparmor-dev

echo "[INFO] Installing Phoronix benchmarks..."

benchmarks=(
  # CPU-Intensive
  "pts/build-linux-kernel"
  "pts/compress-7zip"
  "pts/x264"
  "pts/blake2"
  "pts/coremark"
  "pts/c-ray"
  "pts/gmpbench"
  "pts/primesieve"

  # Memory-Intensive
  "pts/stream"
  "pts/mbw"
  "pts/tinymembench"
  "pts/pmbench"

  # IO-Intensive
  "pts/dbench"
  "pts/compilebench"
  "pts/fs-mark"
  "pts/postmark"

  # Server/Database
  "pts/apache"
  "pts/memcached"
  "pts/redis"
  "pts/mysqlslap"
  "pts/cassandra"
  "pts/nginx"

  # Mixed/Stress
  "pts/sysbench"
  "pts/stress-ng"
  "pts/byte"
  "pts/hackbench"
)

for benchmark in "${benchmarks[@]}"; do
  echo "[INSTALLING] $benchmark"
  phoronix-test-suite install "$benchmark"
done

echo "[DONE] All dependencies and benchmarks installed."
