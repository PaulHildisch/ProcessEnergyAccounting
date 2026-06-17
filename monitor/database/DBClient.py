import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision


class DBClient:
    def __init__(self, url, token, org, bucket):
        # Fall back to the docker-compose defaults so reads/writes keep working
        # when INFLUX_ORG/INFLUX_BUCKET are not supplied via env or CLI.
        self.org = org or "myorg"
        self.bucket = bucket or "mybucket"
        self.client = InfluxDBClient(
            url=url, token=token, org=self.org, timeout=360_000
        )
        self.write_api = self.client.write_api()

    def write_deltas(
        self,
        timestamp,
        interval,
        deltas,
        interval_energy=None,
        avg_power=None,
        node="localhost",
        pid_to_container=None,
    ):
        for pid, d in deltas.items():
            # Resolve container/pod identity for this PID from the live cgroup
            # snapshot passed in by DeltaAggregator. Falls back to empty strings
            # (bare process) when no mapping exists or none was provided.
            _cinfo = (pid_to_container or {}).get(pid, {})
            point = (
                Point("process_interval_metrics")
                .tag("node", node)
                .tag("pid", str(pid))
                .tag("process_name", d.get("name", ""))
                # Container/pod identity — empty string means bare (non-containerised)
                # process. Stored as tags so they are indexed and flow through pivot.
                .tag("container_runtime", _cinfo.get("container_runtime", ""))
                .tag("container_name", _cinfo.get("container_name", ""))
                .tag("pod_name", _cinfo.get("pod_name", ""))
                # Hardware categorical features as tags (low-cardinality strings;
                # stored indexed in InfluxDB and flow through pivot as columns).
                .tag("hw_arch", d.get("hw_arch", "unknown"))
                .tag("hw_cpu_vendor", d.get("hw_cpu_vendor", "unknown"))
                .tag("hw_tdp_tier", d.get("hw_tdp_tier", "unknown"))
                .tag("hw_cpu_governor", d.get("hw_cpu_governor", "unknown"))
                .field("ppid", int(d.get("ppid", -1)))
                .field("interval", float(interval))
                # ── existing process counters ──────────────────────────────────
                .field("delta_cpu_ns", int(d.get("delta_cpu_ns", 0)))
                .field("delta_io_bytes", int(d.get("delta_io_bytes", 0)))
                .field("delta_net_send_bytes", int(d.get("delta_net_send_bytes", 0)))
                .field("context_switches", int(d.get("context_switches", 0)))
                .field("syscall_count", int(d.get("syscall_count", 0)))
                .field("delta_rss_memory", int(d.get("delta_rss_memory", 0)))
                .field(
                    "delta_cpu_time_psutil", float(d.get("delta_cpu_time_psutil", 0))
                )
                .field("delta_cpu_time_proc", float(d.get("delta_cpu_time_proc", 0)))
                # ── perf hardware counters (key names updated to delta_ prefix) ─
                .field("delta_instructions", int(d.get("delta_instructions", 0)))
                .field("delta_cycles", int(d.get("delta_cycles", 0)))
                .field(
                    "delta_branch_instructions",
                    int(d.get("delta_branch_instructions", 0)),
                )
                .field("delta_cache_misses", int(d.get("delta_cache_misses", 0)))
                .field(
                    "delta_stalled_cycles_backend",
                    int(d.get("delta_stalled_cycles_backend", 0)),
                )
                .field("delta_llc_load_misses", int(d.get("delta_llc_load_misses", 0)))
                .field(
                    "delta_llc_store_misses", int(d.get("delta_llc_store_misses", 0))
                )
                .field("delta_cpu_migrations", int(d.get("delta_cpu_migrations", 0)))
                .field("delta_page_faults_min", int(d.get("delta_page_faults_min", 0)))
                .field("delta_page_faults_maj", int(d.get("delta_page_faults_maj", 0)))
                # ── new perf counters ──────────────────────────────────────────
                .field(
                    "delta_stalled_cycles_frontend",
                    int(d.get("delta_stalled_cycles_frontend", 0)),
                )
                .field("delta_branch_misses", int(d.get("delta_branch_misses", 0)))
                .field("delta_ref_cpu_cycles", int(d.get("delta_ref_cpu_cycles", 0)))
                .field("delta_l1d_load_misses", int(d.get("delta_l1d_load_misses", 0)))
                .field(
                    "delta_dtlb_load_misses", int(d.get("delta_dtlb_load_misses", 0))
                )
                .field(
                    "delta_dtlb_store_misses", int(d.get("delta_dtlb_store_misses", 0))
                )
                .field(
                    "delta_node_load_misses", int(d.get("delta_node_load_misses", 0))
                )
                # ── new BPF counters ───────────────────────────────────────────
                .field("delta_disk_read_bytes", int(d.get("delta_disk_read_bytes", 0)))
                .field(
                    "delta_disk_write_bytes", int(d.get("delta_disk_write_bytes", 0))
                )
                .field("delta_net_recv_bytes", int(d.get("delta_net_recv_bytes", 0)))
                .field(
                    "delta_net_send_packets", int(d.get("delta_net_send_packets", 0))
                )
                .field(
                    "delta_net_recv_packets", int(d.get("delta_net_recv_packets", 0))
                )
                # ── hardware numeric features ──────────────────────────────────
                .field("hw_numa_node_count", int(d.get("hw_numa_node_count", 1)))
                .field("hw_freq_ratio", float(d.get("hw_freq_ratio", 0.0)))
                .time(int(timestamp * 1e9), WritePrecision.NS)
            )
            if avg_power is not None:
                point = point.field("avg_power", float(avg_power))
            if interval_energy is not None:
                point = point.field("interval_energy", float(interval_energy))

            for cls, cnt in d.get("syscall_class_deltas", {}).items():
                point = point.field(f"syscall_class_{cls}", int(cnt))

            # Vendor-specific FP/SIMD counters (Intel by width, AMD by op type).
            for name, cnt in d.get("fp_op_deltas", {}).items():
                point = point.field(f"delta_{name}", int(cnt))

            self.write_api.write(bucket=self.bucket, record=point)

    def close(self):
        if self.client:
            self.client.close()
            self.client = None
            self.write_api = None

    def get_benchmark_window(self):
        """Return (start, end) timestamps from benchmark_marker events, or None if not found."""
        query = f"""
            from(bucket: "{self.bucket}")
              |> range(start: -24h)
              |> filter(fn: (r) => r._measurement == "benchmark_marker")
              |> sort(columns: ["_time"])
        """
        try:
            dfs = self.client.query_api().query_data_frame(query, org=self.org)
            df = pd.concat(dfs) if isinstance(dfs, list) else dfs
            if df.empty:
                return None, None
            start = df[df["_value"] == "start"]["_time"].min()
            end = df[df["_value"] == "end"]["_time"].max()
            return start, end
        except Exception:
            return None, None

    def load_data(self, start="-2h", stop=None, aggregate_every="1s"):
        # An explicit window (e.g. session-based export) is authoritative; only
        # the default no-argument call falls back to benchmark-marker trimming.
        explicit_window = stop is not None or start != "-2h"
        range_args = f"start: {start}"
        if stop is not None:
            range_args += f", stop: {stop}"
        query = f"""
            from(bucket: "{self.bucket}")
              |> range({range_args})
              |> filter(fn: (r) =>
                r._measurement == "process_interval_metrics" and
                r._field != "cmdline" and
                r._field != "exe" and
                r._field != "cwd" and
                r._field != "cgroup"
              )
              |> map(fn: (r) => ({{ r with _value: float(v: r._value) }}))
              |> aggregateWindow(every: {aggregate_every}, fn: mean, createEmpty: false)
              |> map(fn: (r) => ({{ r with
                  container_runtime: if exists r.container_runtime then r.container_runtime else "",
                  container_name:    if exists r.container_name    then r.container_name    else "",
                  pod_name:          if exists r.pod_name          then r.pod_name          else ""
              }}))
              |> group(columns: ["pid", "process_name", "container_runtime", "container_name", "pod_name"])
              |> pivot(
                  rowKey: ["_time", "pid", "process_name", "container_runtime", "container_name", "pod_name"],
                  columnKey: ["_field"],
                  valueColumn: "_value"
              )
              |> sort(columns: ["_time", "pid"])
        """
        dfs = self.client.query_api().query_data_frame(query, org=self.org)
        if isinstance(dfs, list):
            dfs = [d for d in dfs if d is not None and not d.empty]
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        else:
            df = dfs if dfs is not None else pd.DataFrame()

        if df.empty:
            return df
        df = df.drop(columns=[c for c in ["result", "table"] if c in df.columns])
        df["_time"] = pd.to_datetime(df["_time"])
        df = df.sort_values(["_time", "pid"])

        if not explicit_window:
            start_marker, end_marker = self.get_benchmark_window()
            if start_marker is not None and end_marker is not None:
                print(
                    f"Trimming to benchmark window: {start_marker} \u2192 {end_marker}"
                )
                df = df[(df["_time"] >= start_marker) & (df["_time"] <= end_marker)]
            else:
                print("No benchmark markers found, returning all data.")

        return df
