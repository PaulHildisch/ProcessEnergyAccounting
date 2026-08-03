import ctypes
import errno
import logging
import os
import re
import resource
import struct
import subprocess
from dataclasses import dataclass

PERF_TYPE_HARDWARE = 0
PERF_COUNT_HW_INSTRUCTIONS = 1
PERF_COUNT_HW_CPU_CYCLES = 0
# BUG FIX: the previous values were wrong against the Linux kernel headers.
#   Old BRANCH_INSTRUCTIONS = 5  → was actually BRANCH_MISSES
#   Old CACHE_MISSES = 9         → was actually REF_CPU_CYCLES
# Any model trained before this fix used the wrong counters for those two
# features. Correct values per include/uapi/linux/perf_event.h:
PERF_COUNT_HW_BRANCH_INSTRUCTIONS = 4  # was 5 (BRANCH_MISSES)
PERF_COUNT_HW_CACHE_REFERENCES = 2
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


def _raw_perf_config(event: int, umask: int) -> int:
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


@dataclass(frozen=True)
class PerfEventSpec:
    metric_name: str
    event_type: int
    config: int
    perf_names: tuple[str, ...]


def _spec(metric_name, event_type, config, *perf_names):
    return PerfEventSpec(
        metric_name, event_type, config, tuple(perf_names or (metric_name,))
    )


ALL_PERF_EVENT_SPECS = [
    _spec(
        "instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instructions"
    ),
    _spec("cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles"),
    _spec(
        "cache_references",
        PERF_TYPE_HARDWARE,
        PERF_COUNT_HW_CACHE_REFERENCES,
        "cache-references",
    ),
    _spec(
        "branch_instructions",
        PERF_TYPE_HARDWARE,
        PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
        "branches",
    ),
    _spec(
        "cache_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, "cache-misses"
    ),
    _spec(
        "stalled_cycles_backend",
        PERF_TYPE_HARDWARE,
        PERF_COUNT_HW_STALLED_CYCLES_BACKEND,
        "stalled-cycles-backend",
    ),
    _spec(
        "llc_load_misses",
        PERF_TYPE_HW_CACHE,
        hw_cache_config(
            PERF_COUNT_HW_CACHE_LL,
            PERF_COUNT_HW_CACHE_OP_READ,
            PERF_COUNT_HW_CACHE_RESULT_MISS,
        ),
        "LLC-load-misses",
    ),
    _spec(
        "llc_store_misses",
        PERF_TYPE_HW_CACHE,
        hw_cache_config(
            PERF_COUNT_HW_CACHE_LL,
            PERF_COUNT_HW_CACHE_OP_WRITE,
            PERF_COUNT_HW_CACHE_RESULT_MISS,
        ),
        "LLC-store-misses",
    ),
    _spec(
        "stalled_cycles_frontend",
        PERF_TYPE_HARDWARE,
        PERF_COUNT_HW_STALLED_CYCLES_FRONTEND,
        "stalled-cycles-frontend",
    ),
    _spec(
        "branch_misses",
        PERF_TYPE_HARDWARE,
        PERF_COUNT_HW_BRANCH_MISSES,
        "branch-misses",
    ),
    _spec(
        "ref_cpu_cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES, "ref-cycles"
    ),
    _spec(
        "l1d_load_misses",
        PERF_TYPE_HW_CACHE,
        hw_cache_config(
            PERF_COUNT_HW_CACHE_L1D,
            PERF_COUNT_HW_CACHE_OP_READ,
            PERF_COUNT_HW_CACHE_RESULT_MISS,
        ),
        "L1-dcache-load-misses",
    ),
    _spec(
        "dtlb_load_misses",
        PERF_TYPE_HW_CACHE,
        hw_cache_config(
            PERF_COUNT_HW_CACHE_DTLB,
            PERF_COUNT_HW_CACHE_OP_READ,
            PERF_COUNT_HW_CACHE_RESULT_MISS,
        ),
        "dTLB-load-misses",
    ),
    _spec(
        "dtlb_store_misses",
        PERF_TYPE_HW_CACHE,
        hw_cache_config(
            PERF_COUNT_HW_CACHE_DTLB,
            PERF_COUNT_HW_CACHE_OP_WRITE,
            PERF_COUNT_HW_CACHE_RESULT_MISS,
        ),
        "dTLB-store-misses",
    ),
    _spec(
        "node_load_misses",
        PERF_TYPE_HW_CACHE,
        hw_cache_config(
            PERF_COUNT_HW_CACHE_NODE,
            PERF_COUNT_HW_CACHE_OP_READ,
            PERF_COUNT_HW_CACHE_RESULT_MISS,
        ),
        "node-load-misses",
    ),
    _spec(
        "cpu_migrations",
        PERF_TYPE_SOFTWARE,
        PERF_COUNT_SW_CPU_MIGRATIONS,
        "cpu-migrations",
    ),
    _spec(
        "page_faults_min",
        PERF_TYPE_SOFTWARE,
        PERF_COUNT_SW_PAGE_FAULTS_MIN,
        "minor-faults",
    ),
    _spec(
        "page_faults_maj",
        PERF_TYPE_SOFTWARE,
        PERF_COUNT_SW_PAGE_FAULTS_MAJ,
        "major-faults",
    ),
]

ALL_PERF_EVENT_SPECS += [
    _spec(name, PERF_TYPE_RAW, config, name) for name, config in FP_ARITH_EVENTS.items()
]

PERF_FEATURE_ALIASES = {
    "delta_instructions": "instructions",
    "delta_cycles": "cycles",
    "delta_branch_instructions": "branch_instructions",
    "delta_cache_references": "cache_references",
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
    "cache_references",
    "cache_misses",
    "branch_instructions",
    "branch_misses",
    "dtlb_load_misses",
}
COMMON_SAFE_PERF_EVENT_NAMES = DEFAULT_PERF_EVENT_NAMES | {
    "dtlb_store_misses",
    "stalled_cycles_frontend",
    "stalled_cycles_backend",
}
ALL_PERF_EVENT_NAMES = tuple(spec.metric_name for spec in ALL_PERF_EVENT_SPECS)
EVENT_SPEC_BY_NAME = {spec.metric_name: spec for spec in ALL_PERF_EVENT_SPECS}
MAX_OPEN_PERF_FDS = int(os.getenv("MAX_OPEN_PERF_FDS", "8192"))
PERF_FD_RLIMIT_RESERVE = int(os.getenv("PERF_FD_RLIMIT_RESERVE", "256"))

PERF_EVENT_FALLBACKS = {
    "llc_load_misses": ("cache_misses",),
    "dtlb_store_misses": ("dtlb_load_misses",),
    "node_load_misses": (),
    "llc_store_misses": (),
}
PERF_NATIVE_NAME_ALIASES = {
    perf_name.lower(): spec.metric_name
    for spec in ALL_PERF_EVENT_SPECS
    for perf_name in spec.perf_names
}
PERF_NATIVE_NAME_ALIASES.update(
    {
        "stalled_cycles_backend": "stalled_cycles_backend",
        "stalled-cycles-backend": "stalled_cycles_backend",
        "stalled_cycles_frontend": "stalled_cycles_frontend",
        "stalled-cycles-frontend": "stalled_cycles_frontend",
        "dtlb_load_misses": "dtlb_load_misses",
        "dtlb-load-misses": "dtlb_load_misses",
        "dtlb_store_misses": "dtlb_store_misses",
        "dtlb-store-misses": "dtlb_store_misses",
        "llc_load_misses": "llc_load_misses",
        "llc-load-misses": "llc_load_misses",
        "llc_store_misses": "llc_store_misses",
        "llc-store-misses": "llc_store_misses",
        "node_load_misses": "node_load_misses",
        "node-load-misses": "node_load_misses",
        "branches": "branch_instructions",
        "branch-instructions": "branch_instructions",
    }
)
_PERF_LIST_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:-]+")


def probe_available_perf_events():
    try:
        completed = subprocess.run(
            ["perf", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logging.warning(
            "Could not probe perf events with 'perf list': %s. Falling back to configured perf_event_open attempts.",
            exc,
        )
        return None

    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0 and not output.strip():
        logging.warning(
            "Could not probe perf events with 'perf list' (exit=%s). Falling back to configured perf_event_open attempts.",
            completed.returncode,
        )
        return None

    available = {token.lower() for token in _PERF_LIST_TOKEN_RE.findall(output)}
    logging.info("Probed %s perf event names from 'perf list'.", len(available))
    return available


def _perf_name_available(spec, available_perf_events):
    if available_perf_events is None:
        return True
    return any(
        perf_name.lower() in available_perf_events for perf_name in spec.perf_names
    )


def _with_metric_name(spec, metric_name):
    return PerfEventSpec(metric_name, spec.event_type, spec.config, spec.perf_names)


def get_effective_max_open_perf_fds():
    try:
        soft_limit, _hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return MAX_OPEN_PERF_FDS

    if soft_limit == resource.RLIM_INFINITY:
        return MAX_OPEN_PERF_FDS

    rlimit_budget = max(int(soft_limit) - PERF_FD_RLIMIT_RESERVE, 0)
    return min(MAX_OPEN_PERF_FDS, rlimit_budget)


def select_supported_perf_event_specs(wanted_names):
    available_perf_events = probe_available_perf_events()
    selected = []

    for name in sorted(wanted_names):
        spec = EVENT_SPEC_BY_NAME.get(name)
        if spec is None:
            continue

        if _perf_name_available(spec, available_perf_events):
            selected.append(spec)
            continue

        fallback_spec = None
        fallback_name = None
        for candidate in PERF_EVENT_FALLBACKS.get(name, ()):
            candidate_spec = EVENT_SPEC_BY_NAME.get(candidate)
            if candidate_spec and _perf_name_available(
                candidate_spec, available_perf_events
            ):
                fallback_name = candidate
                fallback_spec = _with_metric_name(candidate_spec, name)
                break

        if fallback_spec is not None:
            selected.append(fallback_spec)
            logging.warning(
                "Perf event '%s' is not available on this host; using fallback '%s'.",
                name,
                fallback_name,
            )
        else:
            logging.warning(
                "Perf event '%s' is not available on this host and has no supported fallback; reporting it as 0/absent.",
                name,
            )

    return selected


class PersistentPerfCounters:
    def __init__(self):
        self.pid_to_fds = {}
        self.failed_events = set()
        self.failed_event_names_warned = set()
        self.fd_limit_warned = False
        self.max_open_perf_fds = get_effective_max_open_perf_fds()
        self.event_specs = [
            spec
            for spec in ALL_PERF_EVENT_SPECS
            if spec.metric_name in DEFAULT_PERF_EVENT_NAMES
        ]

    def configure(self, model_features=None, perf_events=None):
        wanted_names = resolve_perf_event_names(model_features, perf_events)
        self.event_specs = select_supported_perf_event_specs(wanted_names)
        self.fd_limit_warned = False
        self.max_open_perf_fds = get_effective_max_open_perf_fds()
        soft_limit, _hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        logging.info(
            "Configured perf events (%s, MAX_OPEN_PERF_FDS=%s, effective_perf_fd_cap=%s, RLIMIT_NOFILE=%s): %s",
            len(self.event_specs),
            MAX_OPEN_PERF_FDS,
            self.max_open_perf_fds,
            soft_limit,
            ", ".join(spec.metric_name for spec in self.event_specs) or "<none>",
        )
        self.close()

    def read_pid(self, pid):
        results = {}
        pid_fds = self.pid_to_fds.setdefault(pid, {})

        for spec in self.event_specs:
            name = spec.metric_name
            fd = pid_fds.get(name)
            if fd is None and (pid, name) not in self.failed_events:
                if self._open_fd_count() >= self.max_open_perf_fds:
                    self._warn_fd_limit_reached()
                    results[name] = None
                    continue
                fd = self._open(pid, spec.event_type, spec.config)
                if fd == -1:
                    err = ctypes.get_errno()
                    if err == errno.EMFILE:
                        self._warn_fd_limit_reached()
                        self.max_open_perf_fds = self._open_fd_count()
                    elif name not in self.failed_event_names_warned:
                        logging.warning(
                            "Failed to open perf event '%s' for pid=%s: errno=%s (%s). "
                            "This counter will report as 0 where unavailable.",
                            name,
                            pid,
                            err,
                            os.strerror(err) if err else "unknown",
                        )
                        self.failed_event_names_warned.add(name)
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
        self.failed_event_names_warned.clear()

    def _open_fd_count(self):
        return sum(len(pid_fds) for pid_fds in self.pid_to_fds.values())

    def _warn_fd_limit_reached(self):
        if self.fd_limit_warned:
            return
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        logging.warning(
            "Perf FD limit reached (open_perf_fds=%s, effective_perf_fd_cap=%s, "
            "MAX_OPEN_PERF_FDS=%s, RLIMIT_NOFILE soft/hard=%s/%s). Some PMU metrics "
            "will be reported as 0. Raise the OS nofile limit with prlimit/ulimit, "
            "increase MAX_OPEN_PERF_FDS if needed, or reduce --perf-events.",
            self._open_fd_count(),
            self.max_open_perf_fds,
            MAX_OPEN_PERF_FDS,
            soft_limit,
            hard_limit,
        )
        self.fd_limit_warned = True

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


def resolve_perf_event_names(model_features=None, perf_events=None):
    requested = _parse_perf_events(perf_events)

    if requested == ["no"]:
        return set()

    if not requested or requested == ["auto"]:
        requested = ["model"] if model_features else ["default"]

    names = set()
    for item in requested:
        if item == "all":
            names.update(ALL_PERF_EVENT_NAMES)
        elif item == "default":
            names.update(COMMON_SAFE_PERF_EVENT_NAMES)
        elif item == "no":
            continue
        elif item == "model":
            names.update(
                PERF_FEATURE_ALIASES[feature]
                for feature in (model_features or [])
                if feature in PERF_FEATURE_ALIASES
            )
        else:
            event_name = PERF_FEATURE_ALIASES.get(
                item, PERF_NATIVE_NAME_ALIASES.get(item, item)
            )
            if event_name in EVENT_SPEC_BY_NAME:
                names.add(event_name)
            else:
                logging.warning("Ignoring unknown perf event/feature: %s", item)

    if not names and "no" not in requested:
        names.update(COMMON_SAFE_PERF_EVENT_NAMES)
    return names


def _parse_perf_events(perf_events):
    if perf_events is None:
        return []
    if isinstance(perf_events, str):
        return [
            item.strip().lower()
            for item in perf_events.replace(",", " ").split()
            if item.strip()
        ]
    return [str(item).strip().lower() for item in perf_events if str(item).strip()]


def configure_pmu_metrics(model_features=None, perf_events=None):
    _PERF_COUNTERS.configure(model_features, perf_events)


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


def get_all_metrics(pid, include_perf=True):
    metrics = {}
    if include_perf:
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
