import ctypes
import os
import struct

PERF_TYPE_HARDWARE = 0
PERF_COUNT_HW_CPU_CYCLES = 0
PERF_COUNT_HW_INSTRUCTIONS = 1
PERF_COUNT_HW_CACHE_MISSES = 3
PERF_COUNT_HW_BRANCH_INSTRUCTIONS = 5

PMC_COUNTERS = {
    "cycles": PERF_COUNT_HW_CPU_CYCLES,
    "instructions": PERF_COUNT_HW_INSTRUCTIONS,
    "cache_misses": PERF_COUNT_HW_CACHE_MISSES,
    "branch_instructions": PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
}


class perf_event_attr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint),
        ("size", ctypes.c_uint),
        ("config", ctypes.c_ulonglong),
        ("sample_period", ctypes.c_ulonglong),
        ("sample_type", ctypes.c_ulonglong),
        ("read_format", ctypes.c_ulonglong),
        ("flags", ctypes.c_ulonglong * 3),
    ]


def perf_event_open(attr, pid, cpu, group_fd, flags):
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    return libc.syscall(298, ctypes.byref(attr), pid, cpu, group_fd, flags)


def read_counter(fd):
    return struct.unpack("Q", os.read(fd, 8))[0]


def open_pmc_fds(pids):
    """
    Open perf event fds for all given PIDs. Returns a dict:
        {pid: {counter_name: fd}}
    Counters start accumulating from the moment they are opened.
    PIDs for which perf_event_open fails (e.g. process already gone) are skipped.
    """
    fds_by_pid = {}
    attr = perf_event_attr()
    attr.type = PERF_TYPE_HARDWARE
    attr.size = ctypes.sizeof(perf_event_attr)

    for pid in pids:
        pid_fds = {}
        for name, config in PMC_COUNTERS.items():
            attr.config = config
            fd = perf_event_open(attr, pid, -1, -1, 0)
            if fd != -1:
                pid_fds[name] = fd
        if pid_fds:
            fds_by_pid[pid] = pid_fds

    return fds_by_pid


def read_and_close_pmc_fds(fds_by_pid):
    """
    Read accumulated counter values from all open fds, close them, and return:
        {pid: {counter_name: value}}
    """
    results = {}
    for pid, pid_fds in fds_by_pid.items():
        pid_results = {}
        for name, fd in pid_fds.items():
            try:
                pid_results[name] = read_counter(fd)
            except OSError:
                pid_results[name] = 0
            finally:
                try:
                    os.close(fd)
                except OSError:
                    pass
        results[pid] = pid_results
    return results


def get_memory_usage(pid):
    try:
        with open(f"/proc/{pid}/statm") as f:
            parts = f.read().split()
            rss_pages = int(parts[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            rss_bytes = rss_pages * page_size
            return rss_bytes
    except Exception:
        return None


def get_cpu_usage(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
            utime = int(parts[13])
            stime = int(parts[14])
            return utime + stime
    except Exception:
        return None


def get_all_metrics(pid):
    """Returns memory and cpu_time_ticks for a single PID (no PMC — handled separately)."""
    metrics = {}
    mem = get_memory_usage(pid)
    if mem is not None:
        metrics["memory_rss_bytes"] = mem
    cpu = get_cpu_usage(pid)
    if cpu is not None:
        metrics["cpu_time_ticks"] = cpu
    return metrics
