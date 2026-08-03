import warnings
from datetime import datetime, timedelta, timezone

import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.warnings import MissingPivotFunction
from monitoring.hardware_profile import HARDWARE_ONE_HOT_FIELDS, HARDWARE_TAG_DEFAULTS


class DBClient:
    DEFAULT_TAG_COLUMNS = [
        "container_runtime",
        "container_name",
        "pod_name",
        *HARDWARE_TAG_DEFAULTS.keys(),
    ]

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
                .tag("hw_core_count_bucket", d.get("hw_core_count_bucket", "unknown"))
                .tag("hw_ram_size_bucket", d.get("hw_ram_size_bucket", "unknown"))
                .tag("hw_ram_slots_bucket", d.get("hw_ram_slots_bucket", "unknown"))
                .tag("hw_fan_count_bucket", d.get("hw_fan_count_bucket", "unknown"))
                .tag("hw_temp_state", d.get("hw_temp_state", "unknown"))
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
                .field(
                    "delta_cache_references", int(d.get("delta_cache_references", 0))
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
                .field("hw_core_count", int(d.get("hw_core_count", 0)))
                .field("hw_ram_total_gb", float(d.get("hw_ram_total_gb", 0.0)))
                .field("hw_ram_slot_count", int(d.get("hw_ram_slot_count", -1)))
                .field("hw_fan_count", int(d.get("hw_fan_count", -1)))
                .field("hw_temperature_c", float(d.get("hw_temperature_c", 0.0)))
                .time(int(timestamp * 1e9), WritePrecision.NS)
            )
            for field_name in HARDWARE_ONE_HOT_FIELDS:
                point = point.field(field_name, int(d.get(field_name, 0)))
            if avg_power is not None:
                point = point.field("avg_power", float(avg_power))
            # interval_energy is stored in joules (avg_power_w * interval_seconds).
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

    def _query_data_frame(self, query):
        dfs = self.client.query_api().query_data_frame(query, org=self.org)
        if isinstance(dfs, list):
            dfs = [d for d in dfs if d is not None and not d.empty]
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        return dfs if dfs is not None else pd.DataFrame()

    @staticmethod
    def _parse_absolute_time(value):
        if value is None:
            return None
        value = str(value).strip()
        if value.startswith("-"):
            return None
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    @staticmethod
    def _format_flux_time(value):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @classmethod
    def _time_slices(cls, start, stop, slice_seconds):
        start_dt = cls._parse_absolute_time(start)
        stop_dt = cls._parse_absolute_time(stop)
        slice_seconds = int(slice_seconds or 0)
        if start_dt is None or stop_dt is None or slice_seconds <= 0:
            yield start, stop
            return

        delta = timedelta(seconds=slice_seconds)
        current = start_dt
        while current < stop_dt:
            next_stop = min(current + delta, stop_dt)
            yield cls._format_flux_time(current), cls._format_flux_time(next_stop)
            current = next_stop

    def _load_process_field_keys(self, start, stop=None):
        range_args = f"start: {start}"
        if stop is not None:
            range_args += f", stop: {stop}"
        query = f"""
            import "influxdata/influxdb/schema"

            schema.measurementFieldKeys(
                bucket: "{self.bucket}",
                measurement: "process_interval_metrics",
                {range_args}
            )
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MissingPivotFunction)
            df = self._query_data_frame(query)
        if df.empty or "_value" not in df.columns:
            return []
        excluded = {"cmdline", "exe", "cwd", "cgroup"}
        return sorted(
            field
            for field in df["_value"].dropna().astype(str).unique()
            if field not in excluded
        )

    def load_data(
        self,
        start="-2h",
        stop=None,
        aggregate_every=None,
        tag_columns=None,
        field_batch_size=3,
        query_slice_seconds=60,
        raw_query_mode="time_pivot",
    ):
        # An explicit window (e.g. session-based export) is authoritative; only
        # the default no-argument call falls back to benchmark-marker trimming.
        explicit_window = stop is not None or start != "-2h"
        range_args = f"start: {start}"
        if stop is not None:
            range_args += f", stop: {stop}"

        tag_defaults = {
            "container_runtime": "",
            "container_name": "",
            "pod_name": "",
            **HARDWARE_TAG_DEFAULTS,
        }
        selected_tag_columns = list(tag_columns or self.DEFAULT_TAG_COLUMNS)
        tag_map_lines = ",\n                  ".join(
            f'{column}: if exists r.{column} then r.{column} else "{tag_defaults.get(column, "")}"'
            for column in selected_tag_columns
        )
        pivot_columns = ["_time", "pid", "process_name", *selected_tag_columns]
        group_columns_flux = ", ".join(
            f'"{column}"' for column in ["pid", "process_name", *selected_tag_columns]
        )
        pivot_columns_flux = ", ".join(f'"{column}"' for column in pivot_columns)

        if aggregate_every:
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
                      {tag_map_lines}
                  }}))
                  |> group(columns: [{group_columns_flux}])
                  |> pivot(
                      rowKey: [{pivot_columns_flux}],
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
        else:
            fields = self._load_process_field_keys(start=start, stop=stop)
            if not fields:
                return pd.DataFrame()

            raw_query_mode = str(raw_query_mode or "time_pivot").strip().lower()
            time_slices = list(self._time_slices(start, stop, query_slice_seconds))

            keep_columns = [
                "_time",
                "_field",
                "_value",
                "pid",
                "process_name",
                *selected_tag_columns,
            ]
            keep_columns_flux = ", ".join(f'"{column}"' for column in keep_columns)
            slice_frames = []

            if raw_query_mode == "time_pivot":
                for slice_index, (slice_start, slice_stop) in enumerate(
                    time_slices, start=1
                ):
                    slice_range_args = f"start: {slice_start}"
                    if slice_stop is not None:
                        slice_range_args += f", stop: {slice_stop}"
                    print(
                        "    query page "
                        f"time {slice_index}/{len(time_slices)} "
                        f"({slice_start} → {slice_stop}; all fields)",
                        flush=True,
                    )
                    query = f"""
                        from(bucket: "{self.bucket}")
                          |> range({slice_range_args})
                          |> filter(fn: (r) =>
                            r._measurement == "process_interval_metrics" and
                            r._field != "cmdline" and
                            r._field != "exe" and
                            r._field != "cwd" and
                            r._field != "cgroup"
                          )
                          |> map(fn: (r) => ({{ r with
                              {tag_map_lines}
                          }}))
                          |> pivot(
                              rowKey: [{pivot_columns_flux}],
                              columnKey: ["_field"],
                              valueColumn: "_value"
                          )
                    """
                    slice_df = self._query_data_frame(query)
                    if slice_df.empty:
                        continue
                    slice_df = slice_df.drop(
                        columns=[
                            c for c in ["result", "table"] if c in slice_df.columns
                        ]
                    )
                    slice_df["_time"] = pd.to_datetime(slice_df["_time"])
                    slice_frames.append(slice_df)
            elif raw_query_mode == "field_batch":
                field_batch_size = max(1, int(field_batch_size or 1))
                total_field_batches = (
                    len(fields) + field_batch_size - 1
                ) // field_batch_size

                for slice_index, (slice_start, slice_stop) in enumerate(
                    time_slices, start=1
                ):
                    slice_range_args = f"start: {slice_start}"
                    if slice_stop is not None:
                        slice_range_args += f", stop: {slice_stop}"
                    field_frames = []

                    for field_batch_index, offset in enumerate(
                        range(0, len(fields), field_batch_size), start=1
                    ):
                        field_batch = fields[offset : offset + field_batch_size]
                        print(
                            "    query page "
                            f"time {slice_index}/{len(time_slices)} "
                            f"fields {field_batch_index}/{total_field_batches} "
                            f"({slice_start} → {slice_stop}; {', '.join(field_batch)})",
                            flush=True,
                        )
                        field_filter_flux = " or ".join(
                            f'r._field == "{field}"' for field in field_batch
                        )
                        query = f"""
                            from(bucket: "{self.bucket}")
                              |> range({slice_range_args})
                              |> filter(fn: (r) =>
                                r._measurement == "process_interval_metrics" and
                                ({field_filter_flux})
                              )
                              |> keep(columns: [{keep_columns_flux}])
                        """
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", MissingPivotFunction)
                            batch_df = self._query_data_frame(query)

                        if batch_df.empty:
                            continue

                        batch_df = batch_df.drop(
                            columns=[
                                c for c in ["result", "table"] if c in batch_df.columns
                            ]
                        )
                        for column in ["pid", "process_name"]:
                            if column not in batch_df.columns:
                                batch_df[column] = ""
                            else:
                                batch_df[column] = batch_df[column].fillna("")
                        for column in selected_tag_columns:
                            default = tag_defaults.get(column, "")
                            if column not in batch_df.columns:
                                batch_df[column] = default
                            else:
                                batch_df[column] = batch_df[column].fillna(default)

                        batch_df["_time"] = pd.to_datetime(batch_df["_time"])
                        batch_df = (
                            batch_df.groupby([*pivot_columns, "_field"], dropna=False)[
                                "_value"
                            ]
                            .last()
                            .unstack("_field")
                            .reset_index()
                        )
                        batch_df.columns.name = None
                        field_frames.append(batch_df)

                    if not field_frames:
                        continue

                    slice_df = field_frames[0]
                    for frame in field_frames[1:]:
                        slice_df = slice_df.merge(frame, on=pivot_columns, how="outer")
                    slice_frames.append(slice_df)
            else:
                raise ValueError(
                    "raw_query_mode must be 'time_pivot' or 'field_batch', "
                    f"got {raw_query_mode!r}"
                )

            if not slice_frames:
                return pd.DataFrame()

            df = pd.concat(slice_frames, ignore_index=True, sort=False)
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
