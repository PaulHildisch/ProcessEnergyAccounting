import os

import psutil
from monitoring.bpf_monitoring_client import BPFMonitoringClient
from monitoring.proc_monitoring_client import (
    cleanup_pmu_metrics,
    close_pmu_metrics,
    configure_pmu_metrics,
    get_all_metrics,
)


class MonitoringClient:
    def __init__(
        self, model_features=None, perf_events=None, enable_perf_counters=True
    ):
        self.enable_perf_counters = enable_perf_counters
        if self.enable_perf_counters:
            configure_pmu_metrics(model_features, perf_events)
        self.bpf_client = BPFMonitoringClient()

    def get_process_list(self):
        process_list = self.bpf_client.get_process_list()
        psutil_cpu_times = self.get_all_process_cpu_times()
        ppids = self.get_all_process_ppids()
        for process in process_list:
            metrics = get_all_metrics(
                process["pid"], include_perf=self.enable_perf_counters
            )
            metrics["psutil_cpu_time_ns"] = psutil_cpu_times.get(process["pid"], 0)
            metrics["ppid"] = ppids.get(process["pid"])
            process.update(metrics)
            # print(process["pid"], process["cpu_time_ns"], process["psutil_cpu_time_ns"])
        cleanup_pmu_metrics(process["pid"] for process in process_list)
        return process_list

    def close(self):
        close_pmu_metrics()

    def get_all_process_cpu_times(self):
        cpu_time_map = {}
        config = os.sysconf("SC_CLK_TCK")
        for proc in psutil.process_iter(["pid", "cpu_times"]):
            try:
                times = proc.info["cpu_times"]
                cpu_time_map[proc.info["pid"]] = (times.user + times.system) / config
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                cpu_time_map[proc.info["pid"]] = None
        return cpu_time_map

    def get_all_process_ppids(self):
        ppid_map = {}
        for proc in psutil.process_iter(["pid", "ppid"]):
            try:
                ppid_map[proc.info["pid"]] = proc.info["ppid"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                ppid_map[proc.info["pid"]] = None
        return ppid_map
