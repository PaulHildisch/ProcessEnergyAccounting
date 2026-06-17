import prometheus_client as prom
from typing_extensions import Dict

PROCESS_METRICS = [
    ("delta_cpu_ns", "CPU time delta in nanoseconds"),
    ("delta_io_bytes", "I/O bytes delta"),
    ("delta_net_send_bytes", "Network send bytes delta"),
    ("context_switches", "Context switches count"),
    ("syscall_count", "Syscall count"),
    ("delta_rss_memory", "RSS memory delta"),
    ("delta_cpu_time_psutil", "CPU time delta (psutil)"),
    ("delta_cpu_time_proc", "CPU time delta (proc)"),
    ("delta_instructions", "Instructions delta"),
    ("delta_cycles", "CPU cycles delta"),
    ("delta_branch_instructions", "Branch instructions delta"),
    ("delta_cache_misses", "Cache misses delta"),
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

METRIC_NAMES = [
    "delta_cpu_ns",
    "delta_io_bytes",
    "delta_net_send_bytes",
    "context_switches",
    "syscall_count",
    "delta_rss_memory",
    "delta_cpu_time_psutil",
    "delta_cpu_time_proc",
    "delta_instructions",
    "delta_cycles",
    "delta_branch_instructions",
    "delta_cache_misses",
]

METRIC_LABELS = ["node", "pid", "process_name", "ppid"]
CONTAINER_METRIC_LABELS = ["node", "container_name"]
POD_METRIC_LABELS = ["node", "pod_name"]


class PrometheusExporter:
    def __init__(self, node, addr, port, mode="process"):
        self.node = node
        self.addr = addr
        self.port = port
        self.mode = mode
        self.process_metrics = {
            name: prom.Gauge(name, desc, METRIC_LABELS)
            for name, desc in PROCESS_METRICS + SYSCALL_CLASS_METRICS
        }
        self.pod_metrics = {
            name: prom.Gauge(
                f"pod_{name}",
                f"Pod aggregated {desc.lower()}",
                POD_METRIC_LABELS,
            )
            for name, desc in PROCESS_METRICS
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
    ) -> Dict:
        for pid, d in deltas.items():
            self.process_metrics["delta_cpu_ns"].labels(
                node=node,
                pid=pid,
                process_name=d.get("name", ""),
                ppid=d.get("ppid", ""),
            ).set(int(d.get("delta_cpu_ns", 0)))
            self.process_metrics["delta_io_bytes"].labels(
                node=node,
                pid=pid,
                process_name=d.get("name", ""),
                ppid=d.get("ppid", ""),
            ).set(int(d.get("delta_io_bytes", 0)))
            self.process_metrics["delta_net_send_bytes"].labels(
                node=node,
                pid=pid,
                process_name=d.get("name", ""),
                ppid=d.get("ppid", ""),
            ).set(int(d.get("delta_net_send_bytes", 0)))
            self.process_metrics["context_switches"].labels(
                node=node,
                pid=pid,
                process_name=d.get("name", ""),
                ppid=d.get("ppid", ""),
            ).set(int(d.get("context_switches", 0)))
            self.process_metrics["syscall_count"].labels(
                node=node,
                pid=pid,
                process_name=d.get("name", ""),
                ppid=d.get("ppid", ""),
            ).set(int(d.get("syscall_count", 0)))
            self.process_metrics["delta_rss_memory"].labels(
                node=node,
                pid=pid,
                process_name=d.get("name", ""),
                ppid=d.get("ppid", ""),
            ).set(int(d.get("delta_rss_memory", 0)))
            self.process_metrics["delta_cpu_time_psutil"].labels(
                node=node,
                pid=pid,
                process_name=d.get("name", ""),
                ppid=d.get("ppid", ""),
            ).set(float(d.get("delta_cpu_time_psutil", 0)))
            self.process_metrics["delta_cpu_time_proc"].labels(
                node=node,
                pid=pid,
                process_name=d.get("name", ""),
                ppid=d.get("ppid", ""),
            ).set(float(d.get("delta_cpu_time_proc", 0)))
            self.process_metrics["delta_instructions"].labels(
                node=node,
                pid=pid,
                process_name=d.get("name", ""),
                ppid=d.get("ppid", ""),
            ).set(int(d.get("instructions", 0)))
            self.process_metrics["delta_cycles"].labels(
                node=node,
                pid=pid,
                process_name=d.get("name", ""),
                ppid=d.get("ppid", ""),
            ).set(int(d.get("cycles", 0)))
            self.process_metrics["delta_branch_instructions"].labels(
                node=node,
                pid=pid,
                process_name=d.get("name", ""),
                ppid=d.get("ppid", ""),
            ).set(int(d.get("branch_instructions", 0)))
            self.process_metrics["delta_cache_misses"].labels(
                node=node,
                pid=pid,
                process_name=d.get("name", ""),
                ppid=d.get("ppid", ""),
            ).set(int(d.get("cache_misses", 0)))
            for cls, cnt in d.get("syscall_class_deltas", {}).items():
                self.process_metrics[f"syscall_class_{cls}"].labels(
                    node=node,
                    pid=pid,
                    process_name=d.get("name", ""),
                    ppid=d.get("ppid", ""),
                ).set(int(cnt))
        return self.process_metrics

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

    def set_pod_metrics(
        self, timestamp, interval, pod_metrics, node="localhost"
    ) -> Dict:
        for pod_name, metrics in pod_metrics.items():
            self.pod_metrics["delta_cpu_ns"].labels(
                node=node,
                pod_name=pod_name,
            ).set(int(metrics.get("delta_cpu_ns", 0)))
            self.pod_metrics["delta_io_bytes"].labels(
                node=node,
                pod_name=pod_name,
            ).set(int(metrics.get("delta_io_bytes", 0)))
            self.pod_metrics["delta_net_send_bytes"].labels(
                node=node,
                pod_name=pod_name,
            ).set(int(metrics.get("delta_net_send_bytes", 0)))
            self.pod_metrics["context_switches"].labels(
                node=node,
                pod_name=pod_name,
            ).set(int(metrics.get("context_switches", 0)))
            self.pod_metrics["syscall_count"].labels(
                node=node,
                pod_name=pod_name,
            ).set(int(metrics.get("syscall_count", 0)))
            self.pod_metrics["delta_rss_memory"].labels(
                node=node,
                pod_name=pod_name,
            ).set(int(metrics.get("delta_rss_memory", 0)))
            self.pod_metrics["delta_cpu_time_psutil"].labels(
                node=node,
                pod_name=pod_name,
            ).set(float(metrics.get("delta_cpu_time_psutil", 0)))
            self.pod_metrics["delta_cpu_time_proc"].labels(
                node=node,
                pod_name=pod_name,
            ).set(float(metrics.get("delta_cpu_time_proc", 0)))
            self.pod_metrics["delta_instructions"].labels(
                node=node,
                pod_name=pod_name,
            ).set(int(metrics.get("instructions", 0)))
            self.pod_metrics["delta_cycles"].labels(
                node=node,
                pod_name=pod_name,
            ).set(int(metrics.get("cycles", 0)))
            self.pod_metrics["delta_branch_instructions"].labels(
                node=node,
                pod_name=pod_name,
            ).set(int(metrics.get("branch_instructions", 0)))
            self.pod_metrics["delta_cache_misses"].labels(
                node=node,
                pod_name=pod_name,
            ).set(int(metrics.get("cache_misses", 0)))
        return self.pod_metrics
