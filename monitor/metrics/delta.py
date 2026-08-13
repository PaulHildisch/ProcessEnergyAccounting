import os
from math import isfinite

from monitoring.proc_monitoring_client import FP_ARITH_METRIC_NAMES

HZ = os.sysconf("SC_CLK_TCK")


def compute_delta(snapshots, hw_profiler):
    if len(snapshots) < 2:
        return None, {}

    (t1, d1), (t2, d2) = snapshots[0], snapshots[-1]
    dict1 = {proc["pid"]: proc for proc in d1}
    dict2 = {proc["pid"]: proc for proc in d2}
    interval = t2 - t1
    deltas = {}
    interval_hw_features = (
        hw_profiler.get_interval_features() if hw_profiler is not None else {}
    )

    for pid in set(dict1) & set(dict2):
        prev = dict1[pid]
        curr = dict2[pid]

        delta_cpu_ns = _delta(curr, prev, "cpu_time_ns", clamp_monotonic=True)
        delta_io_bytes = _delta(curr, prev, "disk_io_bytes", clamp_monotonic=True)
        delta_net_send_bytes = _delta(
            curr, prev, "net_send_bytes", clamp_monotonic=True
        )
        delta_syscalls = _delta(curr, prev, "syscall_count", clamp_monotonic=True)
        delta_ctx_switches = _delta(
            curr, prev, "context_switches", clamp_monotonic=True
        )
        delta_cpu_time_psutil = _delta(
            curr, prev, "psutil_cpu_time_ns", clamp_monotonic=True
        )
        delta_cpu_time_ticks = _delta(
            curr, prev, "cpu_time_ticks", clamp_monotonic=True
        )
        delta_instruction = _delta(curr, prev, "instructions", clamp_monotonic=True)
        delta_branch_instr = _delta(
            curr, prev, "branch_instructions", clamp_monotonic=True
        )
        delta_cycles = _delta(curr, prev, "cycles", clamp_monotonic=True)
        delta_cache_references = _delta(
            curr, prev, "cache_references", clamp_monotonic=True
        )
        delta_cache_misses = _delta(curr, prev, "cache_misses", clamp_monotonic=True)
        delta_stalled_cycles_backend = _delta(
            curr, prev, "stalled_cycles_backend", clamp_monotonic=True
        )
        delta_llc_load_misses = _delta(
            curr, prev, "llc_load_misses", clamp_monotonic=True
        )
        delta_llc_store_misses = _delta(
            curr, prev, "llc_store_misses", clamp_monotonic=True
        )
        delta_cpu_migrations = _delta(
            curr, prev, "cpu_migrations", clamp_monotonic=True
        )
        delta_page_faults_min = _delta(
            curr, prev, "page_faults_min", clamp_monotonic=True
        )
        delta_page_faults_maj = _delta(
            curr, prev, "page_faults_maj", clamp_monotonic=True
        )
        delta_rss_memory = _delta(curr, prev, "memory_rss_bytes", clamp_monotonic=False)
        # ticks in ns
        delta_cpu_time_proc_ns = delta_cpu_time_ticks * (1e9 / HZ)

        # New perf counters (correctly named after constant fix)
        delta_stalled_cycles_frontend = _delta(
            curr, prev, "stalled_cycles_frontend", clamp_monotonic=True
        )
        delta_branch_misses = _delta(curr, prev, "branch_misses", clamp_monotonic=True)
        delta_ref_cpu_cycles = _delta(
            curr, prev, "ref_cpu_cycles", clamp_monotonic=True
        )
        delta_l1d_load_misses = _delta(
            curr, prev, "l1d_load_misses", clamp_monotonic=True
        )
        delta_dtlb_load_misses = _delta(
            curr, prev, "dtlb_load_misses", clamp_monotonic=True
        )
        delta_dtlb_store_misses = _delta(
            curr, prev, "dtlb_store_misses", clamp_monotonic=True
        )
        delta_node_load_misses = _delta(
            curr, prev, "node_load_misses", clamp_monotonic=True
        )

        # New BPF counters: directional IO + network recv + packet counts
        delta_disk_read_bytes = _delta(
            curr, prev, "disk_read_bytes", clamp_monotonic=True
        )
        delta_disk_write_bytes = _delta(
            curr, prev, "disk_write_bytes", clamp_monotonic=True
        )
        delta_net_recv_bytes = _delta(
            curr, prev, "net_recv_bytes", clamp_monotonic=True
        )
        delta_net_send_packets = _delta(
            curr, prev, "net_send_packets", clamp_monotonic=True
        )
        delta_net_recv_packets = _delta(
            curr, prev, "net_recv_packets", clamp_monotonic=True
        )

        ppid = curr.get("ppid")
        if ppid is None:
            ppid = -1

        deltas[pid] = {
            "pid": pid,
            "ppid": int(ppid),
            "name": curr.get("name") or "",
            "delta_cpu_ns": int(delta_cpu_ns),
            "delta_io_bytes": int(delta_io_bytes),
            "delta_net_send_bytes": int(delta_net_send_bytes),
            "context_switches": int(delta_ctx_switches),
            "syscall_count": int(delta_syscalls),
            "delta_rss_memory": int(delta_rss_memory),
            "delta_cpu_time_psutil": int(delta_cpu_time_psutil),
            "delta_cpu_time_proc": int(delta_cpu_time_proc_ns),
            # NOTE: branch_instructions and cache_misses now collect the
            # correct counters after fixing PERF_COUNT_HW_BRANCH_INSTRUCTIONS
            # (4) and PERF_COUNT_HW_CACHE_MISSES (3). Models trained before
            # this fix used BRANCH_MISSES and REF_CPU_CYCLES respectively.
            "delta_instructions": int(delta_instruction),
            "delta_cycles": int(delta_cycles),
            "delta_branch_instructions": int(delta_branch_instr),
            "delta_cache_references": int(delta_cache_references),
            "delta_cache_misses": int(delta_cache_misses),
            "delta_stalled_cycles_backend": int(delta_stalled_cycles_backend),
            "delta_llc_load_misses": int(delta_llc_load_misses),
            "delta_llc_store_misses": int(delta_llc_store_misses),
            "delta_cpu_migrations": int(delta_cpu_migrations),
            "delta_page_faults_min": int(delta_page_faults_min),
            "delta_page_faults_maj": int(delta_page_faults_maj),
            # New perf counters
            "delta_stalled_cycles_frontend": int(delta_stalled_cycles_frontend),
            "delta_branch_misses": int(delta_branch_misses),
            "delta_ref_cpu_cycles": int(delta_ref_cpu_cycles),
            "delta_l1d_load_misses": int(delta_l1d_load_misses),
            "delta_dtlb_load_misses": int(delta_dtlb_load_misses),
            "delta_dtlb_store_misses": int(delta_dtlb_store_misses),
            "delta_node_load_misses": int(delta_node_load_misses),
            # New BPF counters
            "delta_disk_read_bytes": int(delta_disk_read_bytes),
            "delta_disk_write_bytes": int(delta_disk_write_bytes),
            "delta_net_recv_bytes": int(delta_net_recv_bytes),
            "delta_net_send_packets": int(delta_net_send_packets),
            "delta_net_recv_packets": int(delta_net_recv_packets),
        }

        # Attach one host-level hardware snapshot to every process record so
        # the DB layer can store it per PID without re-reading sysfs for every
        # PID in the interval.
        if interval_hw_features:
            deltas[pid].update(interval_hw_features)

        prev_classes = prev.get("syscall_classes") or {}
        curr_classes = curr.get("syscall_classes") or {}
        all_classes = set(prev_classes) | set(curr_classes)
        deltas[pid]["syscall_class_deltas"] = {
            cls: int(_num(curr_classes.get(cls)) - _num(prev_classes.get(cls)))
            for cls in all_classes
        }

        # FP/SIMD counters are vendor-specific (Intel by width, AMD by op
        # type), so write whichever set was detected, mirroring the dynamic
        # syscall-class handling above.
        deltas[pid]["fp_op_deltas"] = {
            name: int(_delta(curr, prev, name, clamp_monotonic=True))
            for name in FP_ARITH_METRIC_NAMES
        }

    return interval, deltas


def _num(v):
    """Coerce any value to a finite float; None/NaN/invalid -> 0."""
    if v is None:
        return 0.0
    try:
        x = float(v)
        return x if isfinite(x) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _delta(curr, prev, key, clamp_monotonic=True):
    """
    Safe delta for (mostly) monotonic counters.
    If clamp_monotonic is True, negative deltas (reset/rollover) are clamped to 0.
    """
    d = _num(curr.get(key)) - _num(prev.get(key))
    if clamp_monotonic and d < 0:
        d = 0.0
    return d
