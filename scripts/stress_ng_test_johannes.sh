# Custom stress script
# Just for initial testing while nextflow isn't installed
# run this from /ProcessEnergyAccounting


SESSION_DIR="runs/stressng-custom-$(date +%s)"
mkdir -p "$SESSION_DIR"

#log start time
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$SESSION_DIR/session_start.txt"
echo "[INFO] Started recording data under $SESSION_DIR"

for round in {1..3}; do
    echo "=== Round $round: Starting BUSY phase (60 seconds) ==="

    stress-ng --cpu 0 --timeout 60s --metrics-brief

    echo "=== Round $round: Starting IDLE phase (45 seconds) ==="
    
    sleep 45
done
# creat end timestamp for the export script
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$SESSION_DIR/session_stop.txt"
echo "Workload generation complete!"
