import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision


class DBClient:
    TASK_TAG_COLUMNS = [
        "workflow_run_id",
        "pipeline_name",
        "task_id",
        "task_name",
        "task_tag",
        "executor",
        "work_dir",
        "native_id",
        "group_id",
    ]
    WORKFLOW_TAG_COLUMNS = [
        "workflow_run_id",
        "pipeline_name",
    ]
    DEFAULT_TAG_COLUMNS = [
        "pid",
        "process_name",
    ]
    DEFAULT_LOAD_FIELDS = [
        "ppid",
        "create_time",
        "session_id",
        "delta_cpu_ns",
        "delta_io_bytes",
        "delta_net_send_bytes",
        "context_switches",
        "syscall_count",
        "delta_rss_memory",
        "delta_cpu_time_psutil",
        "delta_cpu_time_proc",
        "delta_cycles",
        "delta_instructions",
        "delta_branch_instructions",
        "delta_cache_misses",
        "syscall_class_file",
        "syscall_class_network",
        "syscall_class_memory",
        "syscall_class_process",
        "syscall_class_other",
        "syscall_class_sched",
        "syscall_class_signal",
        "syscall_class_time",
        "interval_energy",
        "avg_power",
    ]
    SUM_FIELDS = [
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
        "syscall_class_file",
        "syscall_class_network",
        "syscall_class_memory",
        "syscall_class_process",
        "syscall_class_other",
        "syscall_class_sched",
        "syscall_class_signal",
        "syscall_class_time",
    ]
    CARRY_FIELDS = [
        "interval",
        "interval_energy",
        "avg_power",
    ]

    def __init__(self, url, token, org, bucket):
        self.client = InfluxDBClient(url=url, token=token, org=org, timeout=1_200_000)
        self.write_api = self.client.write_api()
        self.org = org
        self.bucket = bucket

    def write_deltas(
        self, timestamp, interval, deltas, avg_power, interval_energy, node="localhost"
    ):
        for pid, d in deltas.items():
            point = (
                Point("process_interval_metrics")
                .tag("node", node)
                .tag("pid", str(pid))
                .tag("process_name", d.get("name", ""))
                .tag("cmdline", d.get("cmdline", ""))
                .tag("exe", d.get("exe", ""))
                .tag("cwd", d.get("cwd", ""))
                .tag("cgroup", d.get("cgroup", ""))
                .tag("workflow_run_id", d.get("workflow_run_id", ""))
                .tag("pipeline_name", d.get("pipeline_name", ""))
                .tag("task_id", d.get("task_id", ""))
                .tag("task_name", d.get("task_name", ""))
                .tag("task_tag", d.get("task_tag", ""))
                .tag("executor", d.get("executor", ""))
                .tag("work_dir", d.get("work_dir", ""))
                .tag("native_id", d.get("native_id", ""))
                .tag("group_id", d.get("group_id", ""))
                .field("interval", float(interval))
                .field("ppid", int(d.get("ppid", -1) or -1))
                .field("cmdline", str(d.get("cmdline", "")))
                .field("exe", str(d.get("exe", "")))
                .field("cwd", str(d.get("cwd", "")))
                .field("cgroup", str(d.get("cgroup", "")))
                .field("create_time", int(d.get("create_time", -1) or -1))
                .field("session_id", int(d.get("session_id", -1) or -1))
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
                .field("avg_power", float(avg_power))
                .field("interval_energy", float(interval_energy))
                .field("delta_instructions", int(d.get("instructions", 0)))
                .field("delta_cycles", int(d.get("cycles", 0)))
                .field(
                    "delta_branch_instructions", int(d.get("branch_instructions", 0))
                )
                .field("delta_cache_misses", int(d.get("cache_misses", 0)))
                .time(int(timestamp * 1e9), WritePrecision.NS)
            )
            for cls, cnt in d.get("syscall_class_deltas", {}).items():
                point = point.field(f"syscall_class_{cls}", int(cnt))

            self.write_api.write(bucket=self.bucket, record=point)

    def close(self):
        if self.client:
            self.client.close()
            self.client = None
            self.write_api = None

    def load_data(
        self,
        start="-1h",
        stop=None,
        measurement="process_interval_metrics",
        fields=None,
        tag_columns=None,
        aggregate_every=None,
        aggregate_fn="mean",
    ):
        fields = fields or self.DEFAULT_LOAD_FIELDS
        tag_columns = tag_columns or self.DEFAULT_TAG_COLUMNS

        field_filter = " or\n                  ".join(
            f'r._field == "{field}"' for field in fields
        )
        range_args = [f"start: {start}"]
        if stop is not None:
            range_args.append(f"stop: {stop}")

        query_lines = [
            f'from(bucket: "{self.bucket}")',
            f"  |> range({', '.join(range_args)})",
            "  |> filter(fn: (r) =>",
            f'    r._measurement == "{measurement}" and',
            "    (",
            f"      {field_filter}",
            "    )",
            "  )",
            "  |> map(fn: (r) => ({ r with _value: float(v: r._value) }))",
        ]

        if aggregate_every is not None:
            query_lines.append(
                f"  |> aggregateWindow(every: {aggregate_every}, fn: {aggregate_fn}, createEmpty: false)"
            )

        if tag_columns:
            query_lines.append(
                "  |> group(columns: [{}])".format(
                    ", ".join(f'"{column}"' for column in tag_columns)
                )
            )

        row_key_columns = ["_time", *tag_columns]
        keep_columns = row_key_columns + fields

        query_lines.extend(
            [
                "  |> pivot(",
                "      rowKey: [{}],".format(
                    ", ".join(f'"{column}"' for column in row_key_columns)
                ),
                '      columnKey: ["_field"],',
                '      valueColumn: "_value"',
                "  )",
                "  |> keep(columns: [{}])".format(
                    ", ".join(f'"{column}"' for column in keep_columns)
                ),
                "  |> sort(columns: [{}])".format(
                    ", ".join(f'"{column}"' for column in row_key_columns)
                ),
            ]
        )

        query = "\n".join(query_lines)
        dfs = self.client.query_api().query_data_frame(query, org=self.org)
        df = pd.concat(dfs) if isinstance(dfs, list) else dfs
        if df is None or df.empty:
            return pd.DataFrame(columns=keep_columns)
        df = df.drop(columns=[c for c in ["result", "table"] if c in df.columns])
        df["_time"] = pd.to_datetime(df["_time"])
        df = df.sort_values(row_key_columns)
        return df

    def load_pid_string_fields(
        self,
        start="-1h",
        stop=None,
        fields=None,
        measurement="process_interval_metrics",
    ):
        """Return one row per (pid, process_name) with the last value of each string field."""
        fields = fields or []
        if not fields:
            return pd.DataFrame(columns=["pid", "process_name"])

        field_filter = " or ".join(f'r._field == "{f}"' for f in fields)
        range_args = [f"start: {start}"]
        if stop is not None:
            range_args.append(f"stop: {stop}")
        keep_cols = ", ".join(f'"{c}"' for c in ["pid", "process_name"] + fields)

        query = "\n".join([
            f'from(bucket: "{self.bucket}")',
            f'  |> range({", ".join(range_args)})',
            f'  |> filter(fn: (r) => r._measurement == "{measurement}" and ({field_filter}))',
            '  |> group(columns: ["pid", "process_name", "_field"])',
            "  |> last()",
            '  |> group(columns: ["pid", "process_name"])',
            '  |> pivot(rowKey: ["pid", "process_name"], columnKey: ["_field"], valueColumn: "_value")',
            f"  |> keep(columns: [{keep_cols}])",
        ])

        dfs = self.client.query_api().query_data_frame(query, org=self.org)
        df = pd.concat(dfs) if isinstance(dfs, list) else dfs
        if df is None or df.empty:
            return pd.DataFrame(columns=["pid", "process_name"] + fields)
        df = df.drop(columns=[c for c in ["result", "table"] if c in df.columns])
        return df

    def load_task_data(self, start="-1h", stop=None, aggregate_every=None):
        process_df = self.load_data(
            start=start,
            stop=stop,
            aggregate_every=aggregate_every,
        )
        return self._aggregate_intervals(process_df, self.TASK_TAG_COLUMNS)

    def load_workflow_data(self, start="-1h", stop=None, aggregate_every=None):
        process_df = self.load_data(
            start=start,
            stop=stop,
            aggregate_every=aggregate_every,
        )
        return self._aggregate_intervals(process_df, self.WORKFLOW_TAG_COLUMNS)

    def _aggregate_intervals(self, df, group_columns):
        if df.empty:
            return pd.DataFrame(
                columns=["_time", *group_columns, *self.DEFAULT_LOAD_FIELDS]
            )

        group_keys = ["_time", *group_columns]
        sum_fields = [field for field in self.SUM_FIELDS if field in df.columns]
        carry_fields = [field for field in self.CARRY_FIELDS if field in df.columns]

        aggregations = {field: "sum" for field in sum_fields}
        aggregations.update({field: "first" for field in carry_fields})

        aggregated = (
            df.groupby(group_keys, dropna=False)
            .agg(aggregations)
            .reset_index()
            .sort_values(group_keys)
        )
        return aggregated
