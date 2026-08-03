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

HARDWARE_TAG_DEFAULTS = {
    "hw_arch": "unknown",
    "hw_cpu_vendor": "unknown",
    "hw_tdp_tier": "unknown",
    "hw_cpu_governor": "unknown",
    "hw_core_count_bucket": "unknown",
    "hw_ram_size_bucket": "unknown",
    "hw_ram_slots_bucket": "unknown",
    "hw_fan_count_bucket": "unknown",
    "hw_temp_state": "unknown",
}

HARDWARE_ONE_HOT_CATEGORIES = {
    "hw_arch": {
        "x86_64": "hw_arch_x86_64",
        "arm64": "hw_arch_arm64",
        "riscv64": "hw_arch_riscv64",
        "other": "hw_arch_other",
    },
    "hw_cpu_vendor": {
        "intel": "hw_cpu_vendor_intel",
        "amd": "hw_cpu_vendor_amd",
        "arm": "hw_cpu_vendor_arm",
        "apple": "hw_cpu_vendor_apple",
        "other": "hw_cpu_vendor_other",
    },
    "hw_tdp_tier": {
        "low": "hw_tdp_tier_low",
        "mid": "hw_tdp_tier_mid",
        "high": "hw_tdp_tier_high",
        "unknown": "hw_tdp_tier_unknown",
    },
    "hw_cpu_governor": {
        "performance": "hw_cpu_governor_performance",
        "powersave": "hw_cpu_governor_powersave",
        "schedutil": "hw_cpu_governor_schedutil",
        "ondemand": "hw_cpu_governor_ondemand",
        "unknown": "hw_cpu_governor_unknown",
    },
    "hw_core_count_bucket": {
        "1_4": "hw_cores_1_4",
        "5_8": "hw_cores_5_8",
        "9_16": "hw_cores_9_16",
        "17_32": "hw_cores_17_32",
        "33_plus": "hw_cores_33_plus",
        "unknown": "hw_cores_unknown",
    },
    "hw_ram_size_bucket": {
        "lt16gb": "hw_ram_lt16gb",
        "16_32gb": "hw_ram_16_32gb",
        "33_64gb": "hw_ram_33_64gb",
        "65_128gb": "hw_ram_65_128gb",
        "129gb_plus": "hw_ram_129gb_plus",
        "unknown": "hw_ram_unknown",
    },
    "hw_ram_slots_bucket": {
        "single": "hw_ram_slots_single",
        "dual": "hw_ram_slots_dual",
        "quad_or_more": "hw_ram_slots_quad_or_more",
        "unknown": "hw_ram_slots_unknown",
    },
    "hw_fan_count_bucket": {
        "0": "hw_fans_0",
        "1": "hw_fans_1",
        "2_plus": "hw_fans_2_plus",
        "unknown": "hw_fans_unknown",
    },
    "hw_temp_state": {
        "cool": "hw_temp_cool",
        "normal": "hw_temp_normal",
        "hot": "hw_temp_hot",
        "unknown": "hw_temp_unknown",
    },
}

HARDWARE_ONE_HOT_FIELDS = tuple(
    field
    for category_fields in HARDWARE_ONE_HOT_CATEGORIES.values()
    for field in category_fields.values()
)


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
    core_count: int
    core_count_bucket: str
    ram_total_gb: float
    ram_size_bucket: str
    ram_slot_count: int  # populated slots; -1 if unavailable
    ram_slots_bucket: str
    fan_count: int  # readable fan sensors; -1 if unavailable
    fan_count_bucket: str
    temperature_c: float  # hottest readable sensor; 0.0 if unavailable
    temp_state: str


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


def _read_core_count() -> int:
    """Return the logical CPU count, or 0 if unavailable."""
    return int(os.cpu_count() or 0)


def _core_count_bucket(count: int) -> str:
    if count <= 0:
        return "unknown"
    if count <= 4:
        return "1_4"
    if count <= 8:
        return "5_8"
    if count <= 16:
        return "9_16"
    if count <= 32:
        return "17_32"
    return "33_plus"


def _read_ram_total_gb() -> float:
    """Read total RAM from /proc/meminfo in GiB."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return round(kb / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError) as exc:
        logger.debug("Cannot read total RAM: %s", exc)
    return 0.0


def _ram_size_bucket(ram_gb: float) -> str:
    if ram_gb <= 0:
        return "unknown"
    if ram_gb < 16:
        return "lt16gb"
    if ram_gb <= 32:
        return "16_32gb"
    if ram_gb <= 64:
        return "33_64gb"
    if ram_gb <= 128:
        return "65_128gb"
    return "129gb_plus"


def _read_ram_slot_count() -> int:
    """Best-effort count of populated DIMM slots from EDAC sysfs."""
    slot_paths = set(glob.glob("/sys/devices/system/edac/mc/mc*/dimm*"))
    if not slot_paths:
        slot_paths = set(glob.glob("/sys/devices/system/edac/mc/mc*/csrow*"))
    return len(slot_paths) if slot_paths else -1


def _ram_slots_bucket(slot_count: int) -> str:
    if slot_count <= 0:
        return "unknown"
    if slot_count == 1:
        return "single"
    if slot_count == 2:
        return "dual"
    return "quad_or_more"


def _read_fan_count() -> int:
    """Count readable hwmon fan input sensors."""
    readable = 0
    for path_str in glob.glob("/sys/class/hwmon/hwmon*/fan*_input"):
        try:
            value = int(Path(path_str).read_text().strip())
            if value >= 0:
                readable += 1
        except (OSError, ValueError) as exc:
            logger.debug("Skipping fan sensor %s: %s", path_str, exc)
    return readable if readable > 0 else -1


def _fan_count_bucket(fan_count: int) -> str:
    if fan_count < 0:
        return "unknown"
    if fan_count == 0:
        return "0"
    if fan_count == 1:
        return "1"
    return "2_plus"


def _read_temperature_c() -> float:
    """Return the hottest readable temperature sensor in Celsius."""
    temperatures: list[float] = []
    candidates = glob.glob("/sys/class/hwmon/hwmon*/temp*_input") + glob.glob(
        "/sys/class/thermal/thermal_zone*/temp"
    )
    for path_str in candidates:
        try:
            raw = float(Path(path_str).read_text().strip())
            # Linux thermal and hwmon sensors usually report millidegrees C.
            temperatures.append(raw / 1000 if raw > 1000 else raw)
        except (OSError, ValueError) as exc:
            logger.debug("Skipping temperature sensor %s: %s", path_str, exc)
    if not temperatures:
        return 0.0
    return round(max(temperatures), 1)


def _temperature_state(temp_c: float) -> str:
    if temp_c <= 0:
        return "unknown"
    if temp_c < 50:
        return "cool"
    if temp_c <= 75:
        return "normal"
    return "hot"


def _one_hot_features(values: dict[str, str]) -> dict[str, int]:
    encoded: dict[str, int] = {}
    for source, category_map in HARDWARE_ONE_HOT_CATEGORIES.items():
        value = values.get(source, "unknown")
        if value not in category_map:
            value = "unknown" if "unknown" in category_map else "other"
        for category, field in category_map.items():
            encoded[field] = int(category == value)
    return encoded


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

    core_count = _safe(_read_core_count, 0)
    ram_total_gb = _safe(_read_ram_total_gb, 0.0)
    ram_slot_count = _safe(_read_ram_slot_count, -1)
    fan_count = _safe(_read_fan_count, -1)
    temperature_c = _safe(_read_temperature_c, 0.0)

    return HardwareProfile(
        arch=_safe(_read_arch, "other"),
        cpu_vendor=_safe(_read_cpu_vendor, "other"),
        tdp_tier=_tdp_tier(_safe(_read_tdp_watts, None)),
        cpu_governor=_safe(_read_cpu_governor, "unknown"),
        numa_node_count=_safe(_read_numa_node_count, 1),
        freq_ratio=_safe(_read_freq_ratio, 0.0),
        core_count=core_count,
        core_count_bucket=_core_count_bucket(core_count),
        ram_total_gb=ram_total_gb,
        ram_size_bucket=_ram_size_bucket(ram_total_gb),
        ram_slot_count=ram_slot_count,
        ram_slots_bucket=_ram_slots_bucket(ram_slot_count),
        fan_count=fan_count,
        fan_count_bucket=_fan_count_bucket(fan_count),
        temperature_c=temperature_c,
        temp_state=_temperature_state(temperature_c),
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
        self._core_count = _read_core_count()
        self._core_count_bucket = _core_count_bucket(self._core_count)
        self._ram_total_gb = _read_ram_total_gb()
        self._ram_size_bucket = _ram_size_bucket(self._ram_total_gb)
        self._ram_slot_count = _read_ram_slot_count()
        self._ram_slots_bucket = _ram_slots_bucket(self._ram_slot_count)
        self._fan_count = _read_fan_count()
        self._fan_count_bucket = _fan_count_bucket(self._fan_count)

    def get_interval_features(self) -> dict:
        """Return a flat dict of hardware features for the current interval.

        Static fields are served from the cache established at init time;
        dynamic fields are read fresh each call.
        """
        governor = _read_cpu_governor()
        temperature_c = _read_temperature_c()
        temp_state = _temperature_state(temperature_c)
        features = {
            "hw_arch": self._arch,
            "hw_cpu_vendor": self._cpu_vendor,
            "hw_tdp_tier": self._tdp_tier,
            "hw_numa_node_count": self._numa_node_count,
            "hw_cpu_governor": governor,
            "hw_freq_ratio": _read_freq_ratio(),
            "hw_core_count": self._core_count,
            "hw_core_count_bucket": self._core_count_bucket,
            "hw_ram_total_gb": self._ram_total_gb,
            "hw_ram_size_bucket": self._ram_size_bucket,
            "hw_ram_slot_count": self._ram_slot_count,
            "hw_ram_slots_bucket": self._ram_slots_bucket,
            "hw_fan_count": self._fan_count,
            "hw_fan_count_bucket": self._fan_count_bucket,
            "hw_temperature_c": temperature_c,
            "hw_temp_state": temp_state,
        }
        features.update(_one_hot_features(features))
        return features
