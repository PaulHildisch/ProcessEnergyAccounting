from typing import Any

import prometheus_client as prom
from monitoring.hardware_profile import HARDWARE_ONE_HOT_FIELDS
from monitoring.proc_monitoring_client import FP_ARITH_METRIC_NAMES

CORE_PROCESS_METRICS = [
    ("delta_cpu_ns", "CPU time delta in nanoseconds"),
    ("delta_io_bytes", "I/O bytes delta"),
    ("delta_net_send_bytes", "Network send bytes delta"),
    ("context_switches", "Context switches count"),
    ("syscall_count", "Syscall count"),
    ("delta_rss_memory", "RSS memory delta"),
    ("delta_cpu_time_psutil", "CPU time delta (psutil)"),
    ("delta_cpu_time_proc", "CPU time delta (proc)"),
]

PERF_METRICS = [
    ("delta_instructions", "Instructions delta"),
    ("delta_cycles", "CPU cycles delta"),
    ("delta_branch_instructions", "Branch instructions delta"),
    ("delta_cache_references", "Cache references delta"),
    ("delta_cache_misses", "Cache misses delta"),
    ("delta_stalled_cycles_backend", "Backend stalled cycles delta"),
    ("delta_llc_load_misses", "LLC load misses delta"),
    ("delta_llc_store_misses", "LLC store misses delta"),
    ("delta_cpu_migrations", "CPU migrations delta"),
    ("delta_page_faults_min", "Minor page faults delta"),
    ("delta_page_faults_maj", "Major page faults delta"),
    ("delta_stalled_cycles_frontend", "Frontend stalled cycles delta"),
    ("delta_branch_misses", "Branch misses delta"),
    ("delta_ref_cpu_cycles", "Reference CPU cycles delta"),
    ("delta_l1d_load_misses", "L1D load misses delta"),
    ("delta_dtlb_load_misses", "DTLB load misses delta"),
    ("delta_dtlb_store_misses", "DTLB store misses delta"),
    ("delta_node_load_misses", "NUMA node load misses delta"),
]

BPF_IO_NET_METRICS = [
    ("delta_disk_read_bytes", "Disk read bytes delta"),
    ("delta_disk_write_bytes", "Disk write bytes delta"),
    ("delta_net_recv_bytes", "Network receive bytes delta"),
    ("delta_net_send_packets", "Network send packets delta"),
    ("delta_net_recv_packets", "Network receive packets delta"),
]

HARDWARE_NUMERIC_METRICS = [
    ("hw_numa_node_count", "NUMA node count attached to the sample"),
    ("hw_freq_ratio", "Mean current-to-max CPU frequency ratio"),
    ("hw_core_count", "Logical CPU core count attached to the sample"),
    ("hw_ram_total_gb", "Total RAM in GiB attached to the sample"),
    ("hw_ram_slot_count", "Populated RAM slot count; -1 when unavailable"),
    ("hw_fan_count", "Readable fan sensor count; -1 when unavailable"),
    ("hw_temperature_c", "Hottest readable temperature sensor in Celsius"),
    *[
        (field, f"One-hot hardware feature {field}")
        for field in HARDWARE_ONE_HOT_FIELDS
    ],
]

FP_ARITH_METRICS = [
    ("delta_fp_scalar", "Scalar floating-point arithmetic ops delta"),
    ("delta_fp_128b_packed", "128-bit packed floating-point arithmetic ops delta"),
    ("delta_fp_256b_packed", "256-bit packed floating-point arithmetic ops delta"),
    ("delta_fp_512b_packed", "512-bit packed floating-point arithmetic ops delta"),
    ("delta_fp_add_sub", "Floating-point add/sub ops delta"),
    ("delta_fp_mult", "Floating-point multiply ops delta"),
    ("delta_fp_div", "Floating-point divide ops delta"),
    ("delta_fp_mac", "Floating-point multiply-accumulate ops delta"),
]

SYSCALL_CLASS_METRICS = [
    ("syscall_class_file", "Syscall class file count"),
    ("syscall_class_network", "Syscall class network count"),
    ("syscall_class_memory", "Syscall class memory count"),
    ("syscall_class_process", "Syscall class process count"),
    ("syscall_class_other", "Syscall class other count"),
    ("syscall_class_sched", "Syscall class sched count"),
    ("syscall_class_signal", "Syscall class signal count"),
    ("syscall_class_time", "Syscall class time count"),
]

PROCESS_METRICS = (
    CORE_PROCESS_METRICS
    + PERF_METRICS
    + BPF_IO_NET_METRICS
    + HARDWARE_NUMERIC_METRICS
    + FP_ARITH_METRICS
)
ALL_NUMERIC_METRICS = PROCESS_METRICS + SYSCALL_CLASS_METRICS
METRIC_NAMES = [name for name, _desc in ALL_NUMERIC_METRICS]

METRIC_LABELS = ["node", "pid", "process_name", "ppid"]
CONTAINER_METRIC_LABELS = ["node", "container_name"]
POD_METRIC_LABELS = ["node", "pod_name"]

DELTA_ALIAS_SOURCES = {
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


class PrometheusExporter:
    def __init__(self, node, addr, port, mode="process"):
        self.node = node
        self.addr = addr
        self.port = port
        self.mode = mode
        self.process_metrics = {
            name: prom.Gauge(name, desc, METRIC_LABELS)
            for name, desc in ALL_NUMERIC_METRICS
        }
        self.container_metrics = {
            name: prom.Gauge(
                f"container_{name}",
                f"Container aggregated {desc.lower()}",
                CONTAINER_METRIC_LABELS,
            )
            for name, desc in ALL_NUMERIC_METRICS
        }
        self.pod_metrics = {
            name: prom.Gauge(
                f"pod_{name}",
                f"Pod aggregated {desc.lower()}",
                POD_METRIC_LABELS,
            )
            for name, desc in ALL_NUMERIC_METRICS
        }
        self.process_predicted_energy = prom.Gauge(
            "process_energy_estimate",
            "Estimated process energy per interval",
            METRIC_LABELS,
        )
        self.container_predicted_energy = prom.Gauge(
            "container_energy_estimate",
            "Estimated container energy per interval",
            CONTAINER_METRIC_LABELS,
        )
        self.pod_predicted_energy = prom.Gauge(
            "pod_energy_estimate",
            "Estimated pod energy per interval",
            POD_METRIC_LABELS,
        )
        prom.start_http_server(port, addr)

    def set_process_metrics(
        self, timestamp, interval, deltas, node="localhost"
    ) -> dict:
        for pid, metrics in deltas.items():
            sample = _flatten_metric_sample(metrics)
            labels = {
                "node": node,
                "pid": pid,
                "process_name": metrics.get("name", ""),
                "ppid": metrics.get("ppid", ""),
            }
            _set_metric_group(self.process_metrics, labels, sample)
        return self.process_metrics

    def set_container_metrics(
        self, timestamp, interval, container_metrics, node="localhost"
    ) -> dict:
        for container_name, metrics in container_metrics.items():
            sample = _flatten_metric_sample(metrics)
            labels = {"node": node, "container_name": container_name}
            _set_metric_group(self.container_metrics, labels, sample)
        return self.container_metrics

    def set_pod_metrics(
        self, timestamp, interval, pod_metrics, node="localhost"
    ) -> dict:
        for pod_name, metrics in pod_metrics.items():
            sample = _flatten_metric_sample(metrics)
            labels = {"node": node, "pod_name": pod_name}
            _set_metric_group(self.pod_metrics, labels, sample)
        return self.pod_metrics

    def set_process_energy_predictions(
        self, timestamp, interval, predictions, deltas, node="localhost"
    ) -> None:
        for pid, predicted_energy in predictions.items():
            pid_int = int(pid)
            process_metrics = deltas.get(pid_int, {})
            self.process_predicted_energy.labels(
                node=node,
                pid=pid_int,
                process_name=process_metrics.get("name", ""),
                ppid=process_metrics.get("ppid", ""),
            ).set(float(predicted_energy))

    def set_container_energy_predictions(
        self, timestamp, interval, predictions, node="localhost"
    ) -> None:
        for container_name, predicted_energy in predictions.items():
            self.container_predicted_energy.labels(
                node=node,
                container_name=container_name,
            ).set(float(predicted_energy))

    def set_pod_energy_predictions(
        self, timestamp, interval, predictions, node="localhost"
    ) -> None:
        for pod_name, predicted_energy in predictions.items():
            self.pod_predicted_energy.labels(
                node=node,
                pod_name=pod_name,
            ).set(float(predicted_energy))


def _set_metric_group(
    gauges: dict, labels: dict[str, Any], sample: dict[str, Any]
) -> None:
    for metric_name in METRIC_NAMES:
        gauges[metric_name].labels(**labels).set(_metric_value(sample, metric_name))


def _metric_value(sample: dict[str, Any], metric_name: str) -> float:
    value = sample.get(metric_name)
    if value is None:
        alias_source = DELTA_ALIAS_SOURCES.get(metric_name)
        if alias_source:
            value = sample.get(alias_source)
    return _to_float(value)


def _flatten_metric_sample(metrics: dict[str, Any]) -> dict[str, Any]:
    sample = {
        key: value
        for key, value in metrics.items()
        if key not in {"pid", "ppid", "name", "syscall_class_deltas", "fp_op_deltas"}
        and not isinstance(value, dict)
    }

    # Backward-compatible aliases for callers that still pass raw perf counter keys.
    for alias_key, source_key in DELTA_ALIAS_SOURCES.items():
        if alias_key not in sample and source_key in sample:
            sample[alias_key] = sample[source_key]

    for cls, count in (metrics.get("syscall_class_deltas", {}) or {}).items():
        sample[f"syscall_class_{cls}"] = count

    for name, count in (metrics.get("fp_op_deltas", {}) or {}).items():
        sample[f"delta_{name}"] = count

    return sample


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
