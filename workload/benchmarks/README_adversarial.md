# RAPL Adversarial Workload Generator

A lightweight bash-based workload generator designed to expose weaknesses in Intel RAPL energy accounting by stressing components outside the RAPL perimeter and creating edge cases where RAPL diverges from wall-outlet power measurements.

## Purpose

This script intentionally creates workloads where **RAPL will fail** to accurately track wall power:
- **Platform components**: NVMe, NIC, fans, PSU losses, VRMs, chipset
- **Burst aliasing**: Sub-millisecond power spikes that RAPL sampling misses
- **Thermal hysteresis**: Fan/thermal lag after workload ends
- **Mixed domains**: Complex multi-component loads

## Requirements

### Mandatory
- `bash` (modern version with `set -euo pipefail` support)
- `stress-ng` (recommended) - provides CPU, memory, and I/O stress
  - Install: `sudo apt install stress-ng` (Debian/Ubuntu) or `sudo yum install stress-ng` (RHEL/CentOS)

### Optional (graceful degradation)
- `fio` - for more aggressive I/O stress
- `iperf3` - for network stress (requires remote iperf3 server)
- `gcc` - for compiling optimized AVX burners and burst oscillators
- `nvidia-smi` + GPU stress tool - for GPU workloads
- `taskset`, `numactl` - for CPU pinning and NUMA control

### Without stress-ng
The script will fall back to Python/bash loops and basic tools like `dd`, but won't be as effective.

## Installation

```bash
# Install recommended tools
sudo apt install stress-ng fio iperf3 gcc

# Make script executable
chmod +x rapl_adversarial_workload.sh
```

## Usage

### Basic Run (5 minutes, default settings)
```bash
./rapl_adversarial_workload.sh
```

### Typical Research Run (10 minutes with I/O stress)
```bash
./rapl_adversarial_workload.sh \
    --duration 600 \
    --phase-time 30 \
    --nvme-path /scratch \
    --mem-gb 16
```

### Full Adversarial Run (all stressors)
```bash
./rapl_adversarial_workload.sh \
    --duration 1200 \
    --phase-time 40 \
    --threads $(nproc) \
    --mem-gb 32 \
    --nvme-path /nvme/fast \
    --network-host 192.168.1.100 \
    --enable-gpu \
    --burst-on-us 500 \
    --burst-off-us 500
```

### Dry Run (see what would execute)
```bash
./rapl_adversarial_workload.sh --duration 300 --dry-run
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--duration SECONDS` | 300 | Total runtime |
| `--phase-time SECONDS` | 20 | Duration per phase |
| `--threads N` | all CPUs | Thread count for parallel workloads |
| `--mem-gb N` | 8 | Memory allocation size |
| `--nvme-path PATH` | /tmp | Directory for I/O stress tests |
| `--network-host HOST` | none | iperf3 server for network stress |
| `--enable-gpu` | disabled | Enable GPU stress if tools available |
| `--burst-on-us N` | 500 | Burst phase: on duration (microseconds) |
| `--burst-off-us N` | 500 | Burst phase: off duration (microseconds) |
| `--dry-run` | disabled | Print phases without execution |

## Workload Phases

### Single-Domain Phases
1. **idle** - Sleep, exposes static platform power invisible to RAPL
2. **cpu_scalar** - Integer operations without AVX frequency throttling
3. **cpu_avx** - AVX/FMA thermal heater, maximum CPU power
4. **memory_bw** - STREAM-like bandwidth, DRAM domain stress
5. **memory_sparse** - Low-activity DRAM access, exposes RAPL bias
6. **io_sequential** - Sequential NVMe writes, PCIe/controller power
7. **io_random** - Random IOPS, different power profile

### Mixed Workload Phases
8. **mixed_compute_mem** - CPU + memory simultaneously
9. **mixed_cpu_io** - AVX + disk I/O, maximizes PSU load and VRM losses
10. **mixed_all_domains** - CPU + memory + I/O + network, absolute max platform power
11. **mixed_unbalanced** - Half cores loaded, half idle (per-core modeling test)
12. **mixed_platform_max** - I/O + network + GPU only (non-RAPL components)
13. **mixed_thermal_ramp** - Gradual ramp: idle → scalar → AVX → full load

### Adversarial/Edge Cases
14. **burst_micro** - 500μs on/off cycles, RAPL sampling aliasing
15. **burst_milli** - 10ms on/off cycles, frequency scaling thrashing
16. **smt_sweep** - 1 thread → N/2 → N threads, SMT power modeling
17. **numa_imbalance** - Pin all work to socket 0 (multi-socket only)
18. **thermal_hysteresis** - AVX blast → immediate idle (fan lag)
19. **power_virus** - Maximum everything: AVX + memory + I/O

### Optional Phases (if enabled)
20. **network** - iperf3 blast (if `--network-host` specified)
21. **gpu** - GPU stress (if `--enable-gpu` enabled)

## Phase Markers

The script outputs timestamped phase markers to stdout for synchronization with external power measurements:

```
PHASE_START idle 1738195200123456789
PHASE_END idle 1738195220123456789
PHASE_START cpu_avx 1738195220123456790
...
```

These markers allow you to:
1. Parse the output to extract phase boundaries
2. Align with RAPL energy counter readings
3. Align with wall power meter timestamps
4. Compare RAPL vs wall power per-phase

### Example: Extracting Phase Timestamps
```bash
./rapl_adversarial_workload.sh --duration 300 2>&1 | grep "PHASE_" > phase_markers.txt
```

## Expected RAPL Failures

Based on the workload design, you should observe:

### High Divergence (20-50% error)
- **mixed_platform_max**: Wall shows 50-150W, RAPL shows ~5-10W
- **thermal_hysteresis**: Wall stays elevated 20-30s after RAPL drops
- **io_random/io_sequential**: NVMe controller power completely missed
- **network**: NIC power invisible to RAPL

### Medium Divergence (10-30% error)
- **burst_micro**: Random aliasing, spikes/gaps in RAPL
- **mixed_cpu_io**: PSU efficiency losses (10-20% overhead)
- **cpu_avx**: VRM losses (5-15% overhead)

### Low Divergence (<10% error)
- **cpu_scalar**: Steady-state, RAPL should track well
- **memory_bw**: DRAM domain usually accurate for bandwidth

### Platform-Constant Offset
- **idle**: Wall meter shows static ~20-50W (fans, BMC, NICs), RAPL shows ~0-2W

## Typical Experimental Setup

1. **Connect wall power meter** (e.g., WattsUp, Yokogawa, or smart PDU)
2. **Start external logging** at high frequency (≥1Hz recommended)
3. **Start RAPL logging** (separate script, sampling at ≥100Hz)
4. **Run workload generator**:
   ```bash
   ./rapl_adversarial_workload.sh --duration 600 --phase-time 30 \
       --nvme-path /nvme/scratch --mem-gb 32 2>&1 | tee workload.log
   ```
5. **Parse phase markers** from `workload.log`
6. **Align timeseries**: RAPL, wall meter, phase boundaries
7. **Compute per-phase energy**: RAPL package+DRAM vs wall meter
8. **Generate comparison plots**

## Script Size Comparison

- **Original Python script**: 1,347 lines
- **New bash script**: 663 lines (51% reduction)
- **Lines of actual logic**: ~400 (rest is comments/structure)

## Why Bash?

1. **Direct tool access**: Native stress-ng, fio, iperf3 invocation
2. **Simpler dependencies**: No Python multiprocessing/numpy complexity
3. **HPC-friendly**: Shell scripts are standard in HPC environments
4. **Graceful fallbacks**: Easy to detect and skip missing tools
5. **Readable**: Sequential phase execution is clearer than class hierarchies

## Troubleshooting

### "stress-ng: command not found"
Install stress-ng or the script will use basic fallbacks (less effective).

### I/O phases fail with "Permission denied"
Change `--nvme-path` to a writable location: `/tmp`, `/scratch`, or your home directory.

### Network phase skips
Requires `--network-host` pointing to a running iperf3 server:
```bash
# On remote machine:
iperf3 -s

# On test machine:
./rapl_adversarial_workload.sh --network-host remote.server.ip
```

### Burst phases don't compile
Requires `gcc`. Install it or the script falls back to Python (less precise timing).

### Phases complete instantly
Check `--phase-time` and `--duration` values. Also verify stress-ng is actually running (not erroring silently).

## Contributing

This is a minimal, focused tool. If you need additional phases:
1. Copy an existing phase function
2. Modify the workload logic
3. Add to the `PHASES` array
4. Test with `--dry-run` first

## License

Same as parent project.

## Citation

If you use this in research, please cite the associated paper on RAPL energy accounting validation.
