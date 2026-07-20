#!/usr/bin/env bash
set -u

# Usage:
# ./k8s-test.sh [sleep_between_pods] [num_parallel_pods] [pod_lifetime] [mode]
#
# mode: cpu | file | mem | net | mixed
# example:
# ./k8s-test.sh 5 4 60 mixed

SLEEP_BETWEEN_PODS="${1:-35}"
NUM_PARALLEL_PODS="${2:-1}"
POD_LIFETIME="${3:-60}"
MODE="${4:-mixed}"

IMAGE="busybox"
NAMESPACE="default"

cleanup() {
    echo "cleanup..."
    kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '/randpod_/ {print $1}' | xargs -r kubectl delete pod -n "$NAMESPACE" --force --grace-period=0 >/dev/null 2>&1 || true
    pkill -P $$ 2>/dev/null || true
}
trap cleanup EXIT INT TERM

run_worker() {
    local worker_id="$1"

    while true; do
        local pname="randpod-${worker_id}-$(tr -dc a-z0-9 </dev/urandom | head -c 8)"
        local launcher="end=\$(( \$(date +%s) + ${POD_LIFETIME} )); while [ \$(date +%s) -lt \$end ]; do sh -c 'sleep 8' & sh -c 'sleep 12' & sh -c 'sleep 16' & sleep 6; done; wait"
        echo "starting $pname lifetime=${POD_LIFETIME}s"

        kubectl run "$pname" \
            -n "$NAMESPACE" \
            --image="$IMAGE" \
            --restart=Never \
            --command -- /bin/sh -c "$launcher" >/dev/null

        kubectl wait --for=condition=Ready "pod/$pname" -n "$NAMESPACE" --timeout=30s >/dev/null 2>&1 || true
        kubectl wait --for=jsonpath='{.status.phase}'=Succeeded "pod/$pname" -n "$NAMESPACE" --timeout=$((POD_LIFETIME + 60))s >/dev/null 2>&1 || true

        sleep "$SLEEP_BETWEEN_PODS"
    done
}

for ((i=0; i<NUM_PARALLEL_PODS; i++)); do
    run_worker "$i" &
done

wait
