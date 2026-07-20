#!/usr/bin/env bash
set -u

# Usage:
# ./docker-test.sh [sleep_between_containers] [num_parallel_containers] [container_lifetime] [mode]
#
# mode: cpu | file | mem | net | mixed
# example:
# ./docker-test.sh 5 4 60 mixed

SLEEP_BETWEEN_CONTAINERS="${1:-35}"
NUM_PARALLEL_CONTAINERS="${2:-1}"
CONTAINER_LIFETIME="${3:-60}"
MODE="${4:-mixed}"

IMAGE="ghcr.io/colinianking/stress-ng"

cleanup() {
    echo "cleanup..."
    docker ps -aq --filter "name=randctr_" | xargs -r docker rm -f
    pkill -P $$ 2>/dev/null || true
}
trap cleanup EXIT INT TERM

bench_args() {
    case "$MODE" in
        cpu)
            echo "--cpu 0 --cpu-method all"
            ;;
        file)
            echo "--hdd 2 --hdd-bytes 2G --hdd-opts dsync"
            ;;
        mem)
            echo "--vm 0 --vm-bytes 80% --vm-method all --verify"
            ;;
        net)
            echo "--sock 4 --sock-domain ipv4 --sock-type stream"
            ;;
        mixed)
            echo "--cpu 2 --cpu-method all --hdd 1 --hdd-bytes 1G --hdd-opts dsync --vm 1 --vm-bytes 1G --vm-method all --sock 2"
            ;;
        *)
            echo "invalid mode: $MODE" >&2
            exit 1
            ;;
    esac
}

run_worker() {
    local worker_id="$1"

    while true; do
        local cname="randctr_${worker_id}_$(tr -dc a-z0-9 </dev/urandom | head -c 8)"
        echo "starting $cname mode=$MODE lifetime=${CONTAINER_LIFETIME}s"

        docker run -d --name "$cname" \
            -v /tmp:/data \
            "$IMAGE" \
            $(bench_args) \
            --temp-path /data \
            --timeout 0 >/dev/null

        sleep "$CONTAINER_LIFETIME"

        echo "stopping $cname"
        docker rm -f "$cname" >/dev/null 2>&1 || true

        sleep "$SLEEP_BETWEEN_CONTAINERS"
    done
}

for ((i=0; i<NUM_PARALLEL_CONTAINERS; i++)); do
    run_worker "$i" &
done

wait
