import ctypes
import logging
import os
import struct

PERF_TYPE_HARDWARE = 0
PERF_COUNT_HW_INSTRUCTIONS = 1
PERF_COUNT_HW_CPU_CYCLES = 0
# BUG FIX: the previous values were wrong against the Linux kernel headers.
#   Old BRANCH_INSTRUCTIONS = 5  → was actually BRANCH_MISSES
#   Old CACHE_MISSES = 9         → was actually REF_CPU_CYCLES
# Any model trained before this fix used the wrong counters for those two
# features. Correct values per include/uapi/linux/perf_event.h:
PERF_COUNT_HW_BRANCH_INSTRUCTIONS = 4  # was 5 (BRANCH_MISSES)
PERF_COUNT_HW_CACHE_MISSES = 3  # was 9 (REF_CPU_CYCLES)

# Additional generalized hardware counters (added).
PERF_COUNT_HW_STALLED_CYCLES_BACKEND = 8

# Newly added generalized hardware counters.
PERF_COUNT_HW_STALLED_CYCLES_FRONTEND = 7
PERF_COUNT_HW_BRANCH_MISSES = (
    5  # properly named; was previously mislabelled as BRANCH_INSTRUCTIONS
)
PERF_COUNT_HW_REF_CPU_CYCLES = (
    9  # properly named; was previously mislabelled as CACHE_MISSES
)

# Hardware cache events use PERF_TYPE_HW_CACHE with a composed config:
#   config = cache_id | (op_id << 8) | (result_id << 16)
PERF_TYPE_HW_CACHE = 3
PERF_COUNT_HW_CACHE_LL = 2
PERF_COUNT_HW_CACHE_L1D = 0  # L1 data cache
PERF_COUNT_HW_CACHE_DTLB = 3  # Data TLB
PERF_COUNT_HW_CACHE_NODE = 6  # NUMA node (remote memory accesses)
PERF_COUNT_HW_CACHE_OP_READ = 0
PERF_COUNT_HW_CACHE_OP_WRITE = 1
PERF_COUNT_HW_CACHE_RESULT_MISS = 1

# Software events are counted by the kernel and do NOT consume a hardware PMU
# counter, so they add no multiplexing pressure.
PERF_TYPE_SOFTWARE = 1
PERF_COUNT_SW_CPU_MIGRATIONS = 4
PERF_COUNT_SW_PAGE_FAULTS_MIN = 5
PERF_COUNT_SW_PAGE_FAULTS_MAJ = 6


def hw_cache_config(cache_id, op_id, result_id):
    return cache_id | (op_id << 8) | (result_id << 16)


# Raw (model-specific) FP/SIMD events. Retired FP uops counted by vector width:
# wider vectors (especially AVX-512) draw disproportionately more power, so the
# width breakdown is a strong energy signal that the generalized counters miss.
PERF_TYPE_RAW = 4

# Intel FP_ARITH_INST_RETIRED (event 0xC7), umasks combine single+double per width.
_INTEL_FP_ARITH_EVENT = 0xC7
_INTEL_FP_ARITH_UMASKS = {
    "fp_scalar": 0x03,  # scalar single + double
    "fp_128b_packed": 0x0C,  # 128-bit packed single + double
    "fp_256b_packed": 0x30,  # 256-bit packed single + double
    "fp_512b_packed": 0xC0,  # 512-bit packed single + double (AVX-512)
}

# AMD FpRetSseAvxOps (event 0x003): retired SSE/AVX FLOPS by operation type.
# AMD breaks FP ops down by operation (add/mul/div/mac), not by vector width, so
# the axis differs from Intel; the PMU counts each MAC as 2 FLOPS.
_AMD_FP_RET_EVENT = 0x003
_AMD_FP_RET_UMASKS = {
    "fp_add_sub": 0x01,
    "fp_mult": 0x02,
    "fp_div": 0x04,
    "fp_mac": 0x08,
}


def _read_cpuinfo():
    """Return (vendor_id, set(flags)) from /proc/cpuinfo; empty on failure."""
    vendor = ""
    flags: set[str] = set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("vendor_id") and not vendor:
                    vendor = line.split(":", 1)[1].strip()
                elif line.startswith("flags") and not flags:
                    flags = set(line.split(":", 1)[1].split())
                if vendor and flags:
                    break
    except OSError:
        pass
    return vendor, flags


def _raw_perf_config(event, umask):
    return (umask << 8) | event


def _build_fp_arith_events():
    """Map FP/SIMD-width metrics to raw perf configs for the detected CPU.

    Returns ``{metric_name: raw_config}``. Empty for unsupported CPUs; any
    individual event the hardware rejects is dropped at open time
    (perf_event_open returns -1), so a partial match degrades gracefully.
    """
    vendor, flags = _read_cpuinfo()
    events: dict[str, int] = {}
    if vendor == "GenuineIntel":
        for name, umask in _INTEL_FP_ARITH_UMASKS.items():
            # Only attempt the 512-bit width on AVX-512-capable parts.
            if name == "fp_512b_packed" and "avx512f" not in flags:
                continue
            events[name] = _raw_perf_config(_INTEL_FP_ARITH_EVENT, umask)
    elif vendor == "AuthenticAMD":
        for name, umask in _AMD_FP_RET_UMASKS.items():
            events[name] = _raw_perf_config(_AMD_FP_RET_EVENT, umask)
    return events


# Detected once at import; reused for every process/sample.
FP_ARITH_EVENTS = _build_fp_arith_events()

# Stable ordering of the detected FP metric names for the delta/export layers.
FP_ARITH_METRIC_NAMES = tuple(FP_ARITH_EVENTS.keys())


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


ALL_PERF_EVENT_SPECS = [
    ("instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS),
    ("cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES),
    ("branch_instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS),
    ("cache_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES),
    (
        "stalled_cycles_backend",
        PERF_TYPE_HARDWARE,
        PERF_COUNT_HW_STALLED_CYCLES_BACKEND,
    ),
    (
        "llc_load_misses",
        PERF_TYPE_HW_CACHE,
        hw_cache_config(
            PERF_COUNT_HW_CACHE_LL,
            PERF_COUNT_HW_CACHE_OP_READ,
            PERF_COUNT_HW_CACHE_RESULT_MISS,
        ),
    ),
    (
        "llc_store_misses",
        PERF_TYPE_HW_CACHE,
        hw_cache_config(
            PERF_COUNT_HW_CACHE_LL,
            PERF_COUNT_HW_CACHE_OP_WRITE,
            PERF_COUNT_HW_CACHE_RESULT_MISS,
        ),
    ),
    (
        "stalled_cycles_frontend",
        PERF_TYPE_HARDWARE,
        PERF_COUNT_HW_STALLED_CYCLES_FRONTEND,
    ),
    ("branch_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES),
    ("ref_cpu_cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES),
    (
        "l1d_load_misses",
        PERF_TYPE_HW_CACHE,
        hw_cache_config(
            PERF_COUNT_HW_CACHE_L1D,
            PERF_COUNT_HW_CACHE_OP_READ,
            PERF_COUNT_HW_CACHE_RESULT_MISS,
        ),
    ),
    (
        "dtlb_load_misses",
        PERF_TYPE_HW_CACHE,
        hw_cache_config(
            PERF_COUNT_HW_CACHE_DTLB,
            PERF_COUNT_HW_CACHE_OP_READ,
            PERF_COUNT_HW_CACHE_RESULT_MISS,
        ),
    ),
    (
        "dtlb_store_misses",
        PERF_TYPE_HW_CACHE,
        hw_cache_config(
            PERF_COUNT_HW_CACHE_DTLB,
            PERF_COUNT_HW_CACHE_OP_WRITE,
            PERF_COUNT_HW_CACHE_RESULT_MISS,
        ),
    ),
    (
        "node_load_misses",
        PERF_TYPE_HW_CACHE,
        hw_cache_config(
            PERF_COUNT_HW_CACHE_NODE,
            PERF_COUNT_HW_CACHE_OP_READ,
            PERF_COUNT_HW_CACHE_RESULT_MISS,
        ),
    ),
    ("cpu_migrations", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS),
    ("page_faults_min", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MIN),
    ("page_faults_maj", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MAJ),
]

ALL_PERF_EVENT_SPECS += [
    (name, PERF_TYPE_RAW, config) for name, config in FP_ARITH_EVENTS.items()
]

PERF_FEATURE_ALIASES = {
    "delta_instructions": "instructions",
    "delta_cycles": "cycles",
    "delta_branch_instructions": "branch_instructions",
    "delta_cache_misses": "cache_misses",
    "delta_stalled_cycles_backend": "stalled_cycles_backend",
    "delta_llc_load_misses": "llc_load_misses",
    "delta_llc_store_misses": "llc_store_misses",
    "delta_cpu_migrations": "cpu_migrations",
    "delta_page_faults_min": "page_faults_min",
    "delta_page_faults_maj": "page_faults_maj",
    "delta_stalled_cycles_frontend": "stalled_cycles_frontend",
    "delta_branch_misses": "branch_misses",
    "delta_ref_cpu_cycles": "ref_cpu_cycles",
    "delta_l1d_load_misses": "l1d_load_misses",
    "delta_dtlb_load_misses": "dtlb_load_misses",
    "delta_dtlb_store_misses": "dtlb_store_misses",
    "delta_node_load_misses": "node_load_misses",
}
PERF_FEATURE_ALIASES.update({f"delta_{name}": name for name in FP_ARITH_EVENTS})

DEFAULT_PERF_EVENT_NAMES = {
    "instructions",
    "cycles",
    "branch_instructions",
    "cache_misses",
}
MAX_OPEN_PERF_FDS = int(os.getenv("MAX_OPEN_PERF_FDS", "512"))


class PersistentPerfCounters:
    def __init__(self):
        self.pid_to_fds = {}
        self.failed_events = set()
        self.event_specs = [
            spec for spec in ALL_PERF_EVENT_SPECS if spec[0] in DEFAULT_PERF_EVENT_NAMES
        ]

    def configure(self, model_features=None):
        if model_features:
            wanted_names = {
                PERF_FEATURE_ALIASES[feature]
                for feature in model_features
                if feature in PERF_FEATURE_ALIASES
            }
            self.event_specs = [
                spec for spec in ALL_PERF_EVENT_SPECS if spec[0] in wanted_names
            ]
        else:
            self.event_specs = [
                spec
                for spec in ALL_PERF_EVENT_SPECS
                if spec[0] in DEFAULT_PERF_EVENT_NAMES
            ]
        self.close()

    def read_pid(self, pid):
        results = {}
        pid_fds = self.pid_to_fds.setdefault(pid, {})

        for name, event_type, config in self.event_specs:
            fd = pid_fds.get(name)
            if fd is None and (pid, name) not in self.failed_events:
                if self._open_fd_count() >= MAX_OPEN_PERF_FDS:
                    results[name] = None
                    continue
                fd = self._open(pid, event_type, config)
                if fd == -1:
                    self.failed_events.add((pid, name))
                    results[name] = None
                    continue
                pid_fds[name] = fd

            if fd is None:
                results[name] = None
                continue

            try:
                results[name] = read_counter(fd)
            except OSError as exc:
                logging.debug(
                    "Failed reading perf counter pid=%s event=%s: %s", pid, name, exc
                )
                self._close_fd(fd)
                pid_fds.pop(name, None)
                results[name] = None

        if not pid_fds:
            self.pid_to_fds.pop(pid, None)
        return results

    def cleanup(self, active_pids):
        active_pids = set(active_pids)
        for pid in list(self.pid_to_fds):
            if pid not in active_pids:
                for fd in self.pid_to_fds.pop(pid).values():
                    self._close_fd(fd)
        self.failed_events = {
            (pid, name) for pid, name in self.failed_events if pid in active_pids
        }

    def close(self):
        for pid_fds in self.pid_to_fds.values():
            for fd in pid_fds.values():
                self._close_fd(fd)
        self.pid_to_fds.clear()
        self.failed_events.clear()

    def _open_fd_count(self):
        return sum(len(pid_fds) for pid_fds in self.pid_to_fds.values())

    @staticmethod
    def _open(pid, event_type, config):
        attr = perf_event_attr()
        attr.type = event_type
        attr.size = ctypes.sizeof(perf_event_attr)
        attr.config = config
        return perf_event_open(attr, pid, -1, -1, 0)

    @staticmethod
    def _close_fd(fd):
        try:
            os.close(fd)
        except OSError:
            pass


_PERF_COUNTERS = PersistentPerfCounters()


def configure_pmu_metrics(model_features=None):
    _PERF_COUNTERS.configure(model_features)


def get_pmu_metrics(pid):
    return _PERF_COUNTERS.read_pid(pid)


def cleanup_pmu_metrics(active_pids):
    _PERF_COUNTERS.cleanup(active_pids)


def close_pmu_metrics():
    _PERF_COUNTERS.close()


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
    metrics = {}
    pmu = get_pmu_metrics(pid)
    if pmu:
        metrics.update(pmu)
    mem = get_memory_usage(pid)
    if mem is not None:
        metrics["memory_rss_bytes"] = mem
    cpu = get_cpu_usage(pid)
    if cpu is not None:
        metrics["cpu_time_ticks"] = cpu
    return metrics
