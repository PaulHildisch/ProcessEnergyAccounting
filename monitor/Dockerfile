FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    build-essential \
    gcc \
    curl \
    git \
    kmod \
    cmake \
    flex \
    bison \
    libelf-dev \
    zlib1g-dev \
    libedit-dev \
    llvm \
    zip \
    llvm-dev \
    libpolly-18-dev \
    clang \
    libclang-dev \
    libfl-dev \
    binutils-dev \
    liblzma-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV" && \
    pip install --upgrade pip setuptools wheel poetry

COPY pyproject.toml poetry.lock ./

ENV PIP_NO_BINARY=numpy

RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi && \
    pip uninstall -y numpy pandas && \
    pip install --no-binary=numpy --no-cache-dir "numpy<2.0" && \
    pip install --no-cache-dir pandas

RUN git clone https://github.com/iovisor/bcc.git /tmp/bcc-src && \
    mkdir -p /tmp/bcc-src/build && \
    cd /tmp/bcc-src/build && \
    cmake .. && \
    make -j1 && \
    make install && \
    cmake -DPYTHON_CMD=python3 .. && \
    cd /tmp/bcc-src/build/src/python/bcc-python3 && \
    pip install . && \
    python -c "from bcc import BPF; import bcc; print('SUCCESS: bcc imported from', bcc.__file__)" && \
    rm -rf /tmp/bcc-src

COPY . .

# Pretrained energy models, baked into the image so the DaemonSet needs no
# host-mounted files. Referenced via --model-pkl /app/models/<name>.pkl.
COPY models/ ./models/

CMD ["python", "delta_aggregator.py"]
