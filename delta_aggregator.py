import argparse
import os
import threading
import time
from collections import deque
from math import isfinite

from dotenv import load_dotenv

from database.client import DBClient
from monitoring.monitor_client import MonitoringClient
from smart_meter.client import SmartMeterAPIClient

HZ = os.sysconf("SC_CLK_TCK")
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "my-super-secret-auth-token"
INFLUX_ORG = "myorg"
INFLUX_BUCKET = "mybucket"

load_dotenv()


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class DeltaAggregator:
    def __init__(
        self,
        interval=1.0,
        sample_rate=0.1,
        db_client=None,
        meter_client=None,
        meter_sensor_id="L1",
        workflow_resolver=None,
    ):
        self.monitor = MonitoringClient()
        self.interval = interval
        self.sample_rate = sample_rate
        self.db_client = db_client
        self.meter_client = meter_client
        self.meter_sensor_id = meter_sensor_id
        self.workflow_resolver = workflow_resolver
        self.snapshots = deque(maxlen=2)  # Store only last two process metric snapshots
        self.running = False
        self.thread = threading.Thread(target=self._collect, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _collect(self):
        while self.running:
            interval_start = time.time()
            power_samples = []
            while (time.time() - interval_start) < self.interval:
                sample_time = time.time()

                # Skip recording of power consumption in case monitoring has been disabled
                if meter_client is None:
                    print("Powermeter has been disabled")
                    time.sleep(self.interval)
                    break
                
                meter_data = self.meter_client.get_sensor_data()
                sensor_ids = {sid.strip() for sid in self.meter_sensor_id.split(",")}
                matched = [s for s in meter_data if s["id"] in sensor_ids]
                if matched:
                    powers = [
                        s["data"].get("ActivePower")
                        for s in matched
                        if s["data"].get("ActivePower") is not None
                    ]
                    if powers:
                        power_samples.append(sum(powers))
                
                sleep_time = self.sample_rate - (time.time() - sample_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            interval_end = time.time()
            actual_interval = interval_end - interval_start
            avg_power = (
                sum(power_samples) / len(power_samples) if power_samples else 0.0
            )
            interval_energy = avg_power * actual_interval

            process_data = self.monitor.get_process_list()
            self._attach_workflow_metadata(process_data)
            self.snapshots.append((interval_end, process_data))

            if len(self.snapshots) == 2:
                interval, deltas = self.get_delta()
                if deltas and self.db_client:
                    print(
                        f"[{time.strftime('%X')}] delta count: {len(deltas)}, avg_power: {avg_power if meter_client else "N/A"}, interval_energy: {interval_energy if meter_client else "N/A"}"
                    )
                    self.db_client.write_deltas(
                        timestamp=interval_end,
                        interval=interval,
                        deltas=deltas,
                        interval_energy=interval_energy,
                        avg_power=avg_power,
                    )

    def get_delta(self):
        if len(self.snapshots) < 2:
            return None, {}

        (t1, d1), (t2, d2) = self.snapshots[0], self.snapshots[-1]
        dict1 = {proc["pid"]: proc for proc in d1}
        dict2 = {proc["pid"]: proc for proc in d2}
        interval = t2 - t1
        deltas = {}

        for pid in set(dict1) & set(dict2):
            prev = dict1[pid]
            curr = dict2[pid]

            delta_cpu_ns = self._delta(curr, prev, "cpu_time_ns", clamp_monotonic=True)
            delta_io_bytes = self._delta(
                curr, prev, "disk_io_bytes", clamp_monotonic=True
            )
            delta_net_send_bytes = self._delta(
                curr, prev, "net_send_bytes", clamp_monotonic=True
            )
            delta_syscalls = self._delta(
                curr, prev, "syscall_count", clamp_monotonic=True
            )
            delta_ctx_switches = self._delta(
                curr, prev, "context_switches", clamp_monotonic=True
            )
            delta_cpu_time_psutil = self._delta(
                curr, prev, "psutil_cpu_time_ns", clamp_monotonic=True
            )
            delta_cpu_time_ticks = self._delta(
                curr, prev, "cpu_time_ticks", clamp_monotonic=True
            )
            delta_instruction = self._delta(
                curr, prev, "instructions", clamp_monotonic=True
            )
            delta_branch_instr = self._delta(
                curr, prev, "branch_instructions", clamp_monotonic=True
            )
            delta_cycles = self._delta(curr, prev, "cycles", clamp_monotonic=True)
            delta_cache_misses = self._delta(
                curr, prev, "cache_misses", clamp_monotonic=True
            )
            delta_rss_memory = self._delta(
                curr, prev, "memory_rss_bytes", clamp_monotonic=False
            )
            # ticks in ns
            delta_cpu_time_proc_ns = delta_cpu_time_ticks * (1e9 / HZ)

            deltas[pid] = {
                "pid": pid,
                "ppid": curr.get("ppid"),
                "name": curr.get("name") or "",
                "cmdline": curr.get("cmdline") or "",
                "exe": curr.get("exe") or "",
                "cwd": curr.get("cwd") or "",
                "cgroup": curr.get("cgroup") or "",
                "create_time": curr.get("create_time"),
                "session_id": curr.get("session_id"),
                "delta_cpu_ns": int(delta_cpu_ns),
                "delta_io_bytes": int(delta_io_bytes),
                "delta_net_send_bytes": int(delta_net_send_bytes),
                "context_switches": int(delta_ctx_switches),
                "syscall_count": int(delta_syscalls),
                "delta_rss_memory": int(delta_rss_memory),
                "delta_cpu_time_psutil": int(delta_cpu_time_psutil),
                "delta_cpu_time_proc": int(delta_cpu_time_proc_ns),
                "instructions": int(delta_instruction),
                "cycles": int(delta_cycles),
                "branch_instructions": int(delta_branch_instr),
                "cache_misses": int(delta_cache_misses),
                "workflow_run_id": curr.get("workflow_run_id", ""),
                "pipeline_name": curr.get("pipeline_name", ""),
                "task_id": curr.get("task_id", ""),
                "task_name": curr.get("task_name", ""),
                "task_tag": curr.get("task_tag", ""),
                "executor": curr.get("executor", ""),
                "work_dir": curr.get("work_dir", ""),
                "native_id": curr.get("native_id", ""),
                "group_id": curr.get("group_id", ""),
            }

            prev_classes = prev.get("syscall_classes") or {}
            curr_classes = curr.get("syscall_classes") or {}
            all_classes = set(prev_classes) | set(curr_classes)
            deltas[pid]["syscall_class_deltas"] = {
                cls: int(
                    self._num(curr_classes.get(cls)) - self._num(prev_classes.get(cls))
                )
                for cls in all_classes
            }

        return interval, deltas

    def _num(self, v):
        """Coerce any value to a finite float; None/NaN/invalid -> 0."""
        if v is None:
            return 0.0
        try:
            x = float(v)
            return x if isfinite(x) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _attach_workflow_metadata(self, process_data):
        if self.workflow_resolver is None:
            return

        assignments = self.workflow_resolver.resolve_processes(process_data)
        for process in process_data:
            process.update(assignments.get(process["pid"], {}))

    def _delta(self, curr, prev, key, clamp_monotonic=True):
        """
        Safe delta for (mostly) monotonic counters.
        If clamp_monotonic is True, negative deltas (reset/rollover) are clamped to 0.
        """
        d = self._num(curr.get(key)) - self._num(prev.get(key))
        if clamp_monotonic and d < 0:
            d = 0.0
        return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DeltaAggregator and write deltas to InfluxDB"
    )
    parser.add_argument(
        "--influx-url", default=os.getenv("INFLUX_URL", INFLUX_URL), help="InfluxDB URL"
    )
    parser.add_argument(
        "--influx-token",
        default=os.getenv("INFLUX_TOKEN", INFLUX_TOKEN),
        help="InfluxDB token",
    )
    parser.add_argument(
        "--influx-org", default=os.getenv("INFLUX_ORG", INFLUX_ORG), help="InfluxDB org"
    )
    parser.add_argument(
        "--influx-bucket",
        default=os.getenv("INFLUX_BUCKET", INFLUX_BUCKET),
        help="InfluxDB bucket",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Aggregation window in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Sampling rate in seconds (optional; defaults to interval)",
    )
    parser.add_argument(
        "--meter-host",
        default=os.getenv("SMARTMETER_HOST"),
        help="Smart meter host (optional)",
    )
    parser.add_argument(
        "--meter-user",
        default=os.getenv("SMARTMETER_USER"),
        help="Smart meter username (optional)",
    )
    parser.add_argument(
        "--meter-password",
        default=os.getenv("SMARTMETER_PASSWORD"),
        help="Smart meter password (optional)",
    )
    parser.add_argument(
        "--meter-ssl",
        action="store_true",
        default=env_flag("SMARTMETER_SSL", False),
        help="Enable SSL for smart meter client (optional)",
    )
    parser.add_argument(
        "--meter-sensor-id",
        default="L1",
        help="Sensor id to read from smart meter (default: L1)",
    )
    parser.add_argument(
        "--smart-meter-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle usage of smart meter for developing purposes (optional)",
    )

    args = parser.parse_args()

    sample_rate = args.sample_rate if args.sample_rate is not None else args.interval

    print(
        f"Starting DeltaAggregator: interval={args.interval}, sample_rate={sample_rate}"
    )
    print(
        f"Influx: {args.influx_url} (org={args.influx_org}, bucket={args.influx_bucket})"
    )

    db_client = DBClient(
        args.influx_url, args.influx_token, args.influx_org, args.influx_bucket
    )

    if args.smart_meter_enabled:
        meter_client = SmartMeterAPIClient(
            host=args.meter_host,
            ssl=args.meter_ssl,
            username=args.meter_user,
            password=args.meter_password,
        )
    else:
        meter_client = None

    monitor = DeltaAggregator(
        interval=args.interval,
        sample_rate=sample_rate,
        db_client=db_client,
        meter_client=meter_client,
        meter_sensor_id=args.meter_sensor_id,
    )

    monitor.start()
    print("Monitoring started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        monitor.stop()
        db_client.close()
        print("Monitoring stopped.")
