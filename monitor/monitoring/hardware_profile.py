"""Hardware-level metadata collection for cross-platform energy modelling.

Per-process performance counters capture *what* a workload does, but not
*where* it runs.  A process completing one billion instructions on a
server-class Xeon draws far more power than the same workload on a mobile
ARM core — not because the instructions differ, but because the underlying
microarchitecture, TDP envelope, and power-management state differ.
Hardware categorical features give a model trained across multiple
hardware generations the baseline context it needs to learn those platform-
level differences: a "low" TDP ARM core at 60 % frequency is a fundamentally
different operating point than a "high" TDP Intel Xeon at 100 % frequency,
even when the instruction mix looks identical.
"""

import glob
import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path

__all__ = ["HardwareProfiler", "HardwareProfile", "collect_hardware_profile"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class HardwareProfile:
    arch: str  # "x86_64", "arm64", "riscv64", "other"
    cpu_vendor: str  # "intel", "amd", "arm", "apple", "other"
    tdp_tier: str  # "low" (<35 W), "mid" (35–125 W), "high" (>125 W), "unknown"
    cpu_governor: str  # "performance", "powersave", "schedutil", "ondemand",
    # "conservative", "other", "unknown"
    numa_node_count: int  # number of NUMA nodes (1 = single-socket / no NUMA)
    freq_ratio: float  # mean(scaling_cur_freq / cpuinfo_max_freq) over online CPUs;
    # 0.0 if unavailable


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_arch() -> str:
    """Return a normalised architecture string from ``platform.machine()``."""
    machine = platform.machine().lower()
    if machine == "x86_64":
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    if machine == "riscv64":
        return "riscv64"
    return "other"


def _read_cpu_vendor() -> str:
    """Detect the CPU vendor by parsing ``/proc/cpuinfo``.

    On x86 the ``vendor_id`` field carries "GenuineIntel" or "AuthenticAMD".
    On ARM the ``CPU implementer`` field carries a hex value (e.g. ``0x41``
    for Arm Ltd, ``0x61`` for Apple).
    """
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("Cannot read /proc/cpuinfo: %s", exc)
        return "other"

    for line in cpuinfo.splitlines():
        lower = line.lower()

        # x86 path
        if lower.startswith("vendor_id"):
            value = line.split(":", 1)[-1].strip()
            if value == "GenuineIntel":
                return "intel"
            if value == "AuthenticAMD":
                return "amd"
            return "other"

        # ARM path
        if lower.startswith("cpu implementer"):
            value = line.split(":", 1)[-1].strip().lower()
            if value == "0x41":
                return "arm"
            if value == "0x61":
                return "apple"
            return "other"

    return "other"


def _read_tdp_watts() -> "float | None":
    """Attempt to read the package TDP from RAPL sysfs entries.

    Strategy (first success wins):
    1. The canonical Intel/AMD RAPL package-0 path.
    2. The first ``constraint_0_max_power_uw`` found anywhere under
       ``/sys/class/powercap/``.
    3. Return ``None`` if nothing is readable.
    """
    # Strategy 1 – preferred, package-0 only
    primary = Path(
        "/sys/class/powercap/intel-rapl/intel-rapl:0/constraint_0_max_power_uw"
    )
    try:
        uw = int(primary.read_text().strip())
        return uw / 1_000_000
    except OSError as exc:
        logger.debug("RAPL primary path unavailable: %s", exc)
    except ValueError as exc:
        logger.debug("RAPL primary path bad value: %s", exc)

    # Strategy 2 – any RAPL domain
    candidates = glob.glob("/sys/class/powercap/*/constraint_0_max_power_uw")
    for candidate in sorted(candidates):
        try:
            uw = int(Path(candidate).read_text().strip())
            return uw / 1_000_000
        except (OSError, ValueError) as exc:
            logger.debug("RAPL candidate %s unreadable: %s", candidate, exc)

    return None


def _tdp_tier(watts: "float | None") -> str:
    """Classify a TDP value in watts into a named tier."""
    if watts is None:
        return "unknown"
    if watts < 35:
        return "low"
    if watts <= 125:
        return "mid"
    return "high"


def _read_cpu_governor() -> str:
    """Read the cpufreq governor for cpu0 from sysfs."""
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        governor = path.read_text().strip()
    except OSError as exc:
        logger.debug("Cannot read cpu governor: %s", exc)
        return "unknown"

    known = {"performance", "powersave", "schedutil", "ondemand", "conservative"}
    return governor if governor in known else "other"


def _read_numa_node_count() -> int:
    """Count NUMA nodes by globbing ``/sys/devices/system/node/node*``."""
    try:
        nodes = glob.glob("/sys/devices/system/node/node*")
        count = len(nodes)
        return max(count, 1)
    except OSError as exc:
        logger.debug("Cannot enumerate NUMA nodes: %s", exc)
        return 1


def _read_freq_ratio() -> float:
    """Compute the mean ratio of current to maximum CPU frequency.

    Iterates over all CPUs that expose ``cpuinfo_max_freq`` in sysfs and
    reads the paired ``scaling_cur_freq``.  Returns 0.0 when no data is
    available.  Result is rounded to three decimal places.
    """
    max_freq_paths = glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq")
    ratios: list[float] = []

    for max_path_str in max_freq_paths:
        max_path = Path(max_path_str)
        cur_path = max_path.parent / "scaling_cur_freq"
        try:
            max_freq = int(max_path.read_text().strip())
            cur_freq = int(cur_path.read_text().strip())
            if max_freq > 0:
                ratios.append(cur_freq / max_freq)
        except (OSError, ValueError) as exc:
            logger.debug("Skipping freq ratio for %s: %s", max_path.parent.name, exc)

    if not ratios:
        return 0.0
    return round(sum(ratios) / len(ratios), 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_hardware_profile() -> HardwareProfile:
    """Collect all hardware metadata and return a :class:`HardwareProfile`.

    Every helper is called within its own try/except so that a failure in one
    source does not prevent the rest from being collected.
    """

    def _safe(fn, default):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Hardware probe %s failed: %s", fn.__name__, exc)
            return default

    return HardwareProfile(
        arch=_safe(_read_arch, "other"),
        cpu_vendor=_safe(_read_cpu_vendor, "other"),
        tdp_tier=_tdp_tier(_safe(_read_tdp_watts, None)),
        cpu_governor=_safe(_read_cpu_governor, "unknown"),
        numa_node_count=_safe(_read_numa_node_count, 1),
        freq_ratio=_safe(_read_freq_ratio, 0.0),
    )


class HardwareProfiler:
    """Stateful profiler that caches slow-changing hardware features.

    Static features (arch, vendor, TDP tier, NUMA topology) are collected
    once at construction time.  Dynamic features (CPU governor and current
    frequency ratio) are read fresh on every call to
    :meth:`get_interval_features` because they can change at runtime under
    frequency scaling or governor switches.
    """

    def __init__(self) -> None:
        self._arch = _read_arch()
        self._cpu_vendor = _read_cpu_vendor()
        self._tdp_tier = _tdp_tier(_read_tdp_watts())
        self._numa_node_count = _read_numa_node_count()

    def get_interval_features(self) -> dict:
        """Return a flat dict of hardware features for the current interval.

        Static fields are served from the cache established at init time;
        dynamic fields are read fresh each call.
        """
        return {
            "hw_arch": self._arch,
            "hw_cpu_vendor": self._cpu_vendor,
            "hw_tdp_tier": self._tdp_tier,
            "hw_numa_node_count": self._numa_node_count,
            "hw_cpu_governor": _read_cpu_governor(),
            "hw_freq_ratio": _read_freq_ratio(),
        }
