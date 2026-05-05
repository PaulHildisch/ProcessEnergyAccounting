import os

import psutil

from monitoring.bpf_monitoring_client import BPFMonitoringClient
from monitoring.proc_monitoring_client import (
    get_all_metrics,
    open_pmc_fds,
    read_and_close_pmc_fds,
)


class MonitoringClient:
    def __init__(self):
        self.bpf_client = BPFMonitoringClient()
        # Open perf fds are kept between calls so counters accumulate for a full interval.
        self._pmc_fds = {}           # {pid: {counter_name: fd}} — currently open
        self._pmc_cumulative = {}    # {pid: {counter_name: total}} — running sums

    def get_process_list(self):
        # 1. Read PMC values accumulated since the last call and close old fds.
        if self._pmc_fds:
            interval_readings = read_and_close_pmc_fds(self._pmc_fds)
            for pid, counters in interval_readings.items():
                acc = self._pmc_cumulative.setdefault(pid, {})
                for name, value in counters.items():
                    acc[name] = acc.get(name, 0) + value

        # 2. Get the current process list from BPF + psutil metadata.
        process_list = self.bpf_client.get_process_list()
        psutil_cpu_times = self.get_all_process_cpu_times()
        process_metadata = self.get_all_process_metadata()

        current_pids = set()
        for process in process_list:
            pid = process["pid"]
            current_pids.add(pid)
            metrics = get_all_metrics(pid)
            metrics["psutil_cpu_time_ns"] = psutil_cpu_times.get(pid, 0)
            metrics.update(process_metadata.get(pid, {}))
            # Attach cumulative PMC values (0 on first call — no prior interval).
            pmc = self._pmc_cumulative.get(pid, {})
            metrics["cycles"] = pmc.get("cycles", 0)
            metrics["instructions"] = pmc.get("instructions", 0)
            metrics["cache_misses"] = pmc.get("cache_misses", 0)
            metrics["branch_instructions"] = pmc.get("branch_instructions", 0)
            process.update(metrics)

        # 3. Open fresh fds for all current PIDs — they will accumulate until the next call.
        self._pmc_fds = open_pmc_fds(current_pids)

        # 4. Clean up cumulative state for PIDs that no longer exist.
        dead_pids = set(self._pmc_cumulative) - current_pids
        for pid in dead_pids:
            del self._pmc_cumulative[pid]

        return process_list

    def get_all_process_cpu_times(self):
        cpu_time_map = {}
        for proc in psutil.process_iter(["pid", "cpu_times"]):
            try:
                times = proc.info["cpu_times"]
                cpu_time_seconds = times.user + times.system
                cpu_time_map[proc.info["pid"]] = int(cpu_time_seconds * 1e9)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                cpu_time_map[proc.info["pid"]] = None
        return cpu_time_map

    def get_all_process_metadata(self):
        metadata_map = {}
        attrs = ["pid", "ppid", "cmdline", "exe", "cwd", "create_time"]
        for proc in psutil.process_iter(attrs):
            pid = proc.info["pid"]
            metadata_map[pid] = {
                "ppid": proc.info.get("ppid"),
                "cmdline": self._join_cmdline(proc.info.get("cmdline")),
                "exe": proc.info.get("exe"),
                "cwd": proc.info.get("cwd"),
                "cgroup": self._read_cgroup(pid),
                "create_time": self._to_epoch_ns(proc.info.get("create_time")),
                "session_id": self._get_session_id(pid),
            }
        return metadata_map

    def _join_cmdline(self, cmdline):
        if not cmdline:
            return ""
        return " ".join(part for part in cmdline if part)

    def _read_cgroup(self, pid):
        try:
            with open(f"/proc/{pid}/cgroup", "r", encoding="utf-8") as handle:
                return handle.read().strip()
        except OSError:
            return ""

    def _get_session_id(self, pid):
        try:
            return os.getsid(pid)
        except OSError:
            return None

    def _to_epoch_ns(self, timestamp_seconds):
        if timestamp_seconds is None:
            return None
        return int(timestamp_seconds * 1e9)
