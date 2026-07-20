import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# The `database` package lives under the repo's `monitor/` directory. db-export.sh
# puts it on PYTHONPATH, but running this script directly does not, so make it
# importable here to keep both entry points working.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra_path in (_REPO_ROOT / "monitor", _REPO_ROOT):
    _extra_path_str = str(_extra_path)
    if _extra_path_str not in sys.path:
        sys.path.insert(0, _extra_path_str)

from database.DBClient import DBClient  # noqa: E402

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


# A .env discovered from the working directory upward is loaded at import time;
# an explicit --env-file (resolved in __main__) can override it.
load_dotenv()


def _load_data_compat(
    client,
    start,
    stop=None,
    aggregate_every=None,
    field_batch_size=None,
    query_slice_seconds=None,
    raw_query_mode=None,
):
    """Call DBClient.load_data with backward-compatible signatures."""
    kwargs = {
        "start": start,
        "stop": stop,
        "aggregate_every": aggregate_every,
    }
    if field_batch_size is not None:
        kwargs["field_batch_size"] = field_batch_size
    if query_slice_seconds is not None:
        kwargs["query_slice_seconds"] = query_slice_seconds
    if raw_query_mode is not None:
        kwargs["raw_query_mode"] = raw_query_mode
    if hasattr(DBClient, "DEFAULT_TAG_COLUMNS"):
        kwargs["tag_columns"] = DBClient.DEFAULT_TAG_COLUMNS

    try:
        return client.load_data(**kwargs)
    except TypeError:
        # Fallback for older DBClient implementations that expose load_data() with no args.
        return client.load_data()


def add_hardware_one_hot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric one-hot hardware columns from low-cardinality hardware tags."""
    if df.empty:
        return df

    encoded = df.copy()
    for source_column, category_map in HARDWARE_ONE_HOT_CATEGORIES.items():
        if source_column not in encoded.columns:
            continue
        values = encoded[source_column].fillna("unknown").astype(str)
        for category, output_column in category_map.items():
            encoded[output_column] = (values == category).astype(int)
    return encoded


def load_dataset(
    client,
    level,
    start,
    stop=None,
    aggregate_every=None,
    field_batch_size=None,
    query_slice_seconds=None,
    raw_query_mode=None,
):
    if level == "process":
        # Fetch numeric fields normally (float cast + aggregation) when supported.
        df = _load_data_compat(
            client=client,
            start=start,
            stop=stop,
            aggregate_every=aggregate_every,
            field_batch_size=field_batch_size,
            query_slice_seconds=query_slice_seconds,
            raw_query_mode=raw_query_mode,
        )

        # cmdline, exe, cwd, cgroup are string fields — merge only when API exists.
        if hasattr(client, "load_pid_string_fields"):
            string_fields = ["cmdline", "exe", "cwd", "cgroup"]
            str_df = client.load_pid_string_fields(
                start=start,
                stop=stop,
                fields=string_fields,
            )
            if not str_df.empty:
                df = df.merge(str_df, on=["pid", "process_name"], how="left")
        return df

    if level == "task":
        if not hasattr(client, "load_task_data"):
            raise NotImplementedError("This DBClient does not implement load_task_data")
        return client.load_task_data(
            start=start,
            stop=stop,
            aggregate_every=aggregate_every,
        )

    if level == "workflow":
        if not hasattr(client, "load_workflow_data"):
            raise NotImplementedError(
                "This DBClient does not implement load_workflow_data"
            )
        return client.load_workflow_data(
            start=start,
            stop=stop,
            aggregate_every=aggregate_every,
        )

    raise ValueError(f"Unsupported level: {level}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export process, task, or workflow interval data from InfluxDB"
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to a .env file with INFLUX_* settings (overrides the environment). "
        "If omitted, a .env is auto-discovered from the working directory upward.",
    )
    parser.add_argument(
        "--influx-url",
        default=None,
        help="InfluxDB URL (default: INFLUX_URL from --env-file/env)",
    )
    parser.add_argument(
        "--influx-token",
        default=None,
        help="InfluxDB token (default: INFLUX_TOKEN from --env-file/env)",
    )
    parser.add_argument(
        "--influx-org",
        default=None,
        help="InfluxDB org (default: INFLUX_ORG from --env-file/env)",
    )
    parser.add_argument(
        "--influx-bucket",
        default=None,
        help="InfluxDB bucket (default: INFLUX_BUCKET from --env-file/env)",
    )
    parser.add_argument(
        "--level",
        choices=["process", "task", "workflow"],
        default="process",
        help="Aggregation level for the exported dataset",
    )
    parser.add_argument(
        "--start",
        default="-1h",
        help="Flux range start expression, for example -1h or 2026-03-12T09:00:00Z",
    )
    parser.add_argument(
        "--stop",
        default=None,
        help="Optional Flux range stop expression",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=None,
        help="Capture the last N days. Combinable with --hours/--minutes; overrides --start.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Capture the last N hours. Combinable with --days/--minutes; overrides --start.",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=None,
        help="Capture the last N minutes. Combinable with --days/--hours; overrides --start.",
    )
    parser.add_argument(
        "--aggregate-every",
        default=None,
        help="Optional Flux aggregate window, for example 1s or 5s. Omit this for raw process-interval exports used by estimator training.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional parquet output path",
    )
    parser.add_argument(
        "--chunk-minutes",
        type=int,
        default=30,
        help="Split the time range into chunks of this many minutes to avoid query timeouts. Default: 30.",
    )
    parser.add_argument(
        "--field-batch-size",
        type=int,
        default=3,
        help="For raw process exports, query this many fields at a time and merge locally. Exports all fields; lower values reduce Influx response size, higher values reduce query count. Default: 3.",
    )
    parser.add_argument(
        "--query-slice-seconds",
        type=int,
        default=60,
        help="For raw process exports, internally page each chunk into time slices of this many seconds. Exports all timestamps; lower values reduce each Influx response size. Default: 60.",
    )
    parser.add_argument(
        "--raw-query-mode",
        choices=["time_pivot", "field_batch"],
        default="time_pivot",
        help="Raw process export strategy. time_pivot queries all fields per time slice using Flux pivot; field_batch queries a few fields at a time and pivots locally. Default: time_pivot.",
    )

    args = parser.parse_args()

    # Accepts RFC3339 timestamps (e.g. 2026-04-15T09:42:00Z) or relative (e.g. -1h).
    def _parse_ts(s):
        if s is None:
            return None
        s = s.strip()
        if s.startswith("-"):
            return None  # relative — can't chunk, fall through to single query
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    # An explicit --env-file overrides the environment; otherwise values come
    # from the environment / auto-discovered .env loaded at import time.
    if args.env_file is not None:
        if not args.env_file.is_file():
            parser.error(f"--env-file not found: {args.env_file}")
        load_dotenv(args.env_file.resolve(), override=True)
        print(f"Loaded InfluxDB settings from {args.env_file}")

    # CLI flags take precedence over --env-file/env values.
    args.influx_url = args.influx_url or os.getenv("INFLUX_URL", "")
    args.influx_token = args.influx_token or os.getenv("INFLUX_TOKEN", "")
    args.influx_org = args.influx_org or os.getenv("INFLUX_ORG", "")
    args.influx_bucket = args.influx_bucket or os.getenv("INFLUX_BUCKET", "")

    if not args.influx_url:
        parser.error(
            "InfluxDB URL is empty. Pass --influx-url, set INFLUX_URL, or supply "
            "--env-file pointing at a .env that defines INFLUX_URL."
        )

    # --days/--hours/--minutes are a convenience: they define a relative look-back
    # window that ends at --stop (when absolute) or now, and is converted to
    # concrete start/stop timestamps so chunking still applies.
    if any(v is not None for v in (args.days, args.hours, args.minutes)):
        lookback = timedelta(
            days=args.days or 0,
            hours=args.hours or 0,
            minutes=args.minutes or 0,
        )
        if lookback <= timedelta(0):
            parser.error("--days/--hours/--minutes must add up to a positive duration.")
        window_end = _parse_ts(args.stop) or datetime.now(timezone.utc)
        args.start = (window_end - lookback).strftime("%Y-%m-%dT%H:%M:%SZ")
        args.stop = window_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"Capturing last {lookback} (start={args.start} stop={args.stop})")

    output_path = args.output or f"{args.level}_interval_data.parquet"

    data_client = DBClient(
        args.influx_url,
        args.influx_token,
        args.influx_org,
        args.influx_bucket,
    )

    start_dt = _parse_ts(args.start)
    stop_dt = _parse_ts(args.stop) or datetime.now(timezone.utc)

    if start_dt is None or args.chunk_minutes <= 0:
        # Relative start or chunking disabled — single query as before
        df = load_dataset(
            client=data_client,
            level=args.level,
            start=args.start,
            stop=args.stop,
            aggregate_every=args.aggregate_every,
            field_batch_size=args.field_batch_size,
            query_slice_seconds=args.query_slice_seconds,
            raw_query_mode=args.raw_query_mode,
        )
        chunks = [df]
    else:
        # Chunked export: split [start, stop] into chunk_minutes windows
        chunk_delta = timedelta(minutes=args.chunk_minutes)
        chunks = []
        chunk_start = start_dt
        chunk_num = 0
        total_chunks = int((stop_dt - start_dt) / chunk_delta) + 1

        while chunk_start < stop_dt:
            chunk_end = min(chunk_start + chunk_delta, stop_dt)
            start_str = chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ")
            stop_str = chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")
            chunk_num += 1
            print(
                f"Chunk {chunk_num}/{total_chunks}: {start_str} → {stop_str} ...",
                flush=True,
            )

            chunk_df = load_dataset(
                client=data_client,
                level=args.level,
                start=start_str,
                stop=stop_str,
                aggregate_every=args.aggregate_every,
                field_batch_size=args.field_batch_size,
                query_slice_seconds=args.query_slice_seconds,
                raw_query_mode=args.raw_query_mode,
            )
            print(f"  → {len(chunk_df)} rows", flush=True)
            if not chunk_df.empty:
                chunks.append(chunk_df)
            chunk_start = chunk_end

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    df = add_hardware_one_hot_features(df)

    print(f"\nLoaded {len(df)} rows from InfluxDB for level={args.level}")

    if "interval_energy" in df.columns:
        active = df[df["interval_energy"] > 0]
        print(f"Active interval shape: {active.shape}")

    print("Memory usage (MB):", df.memory_usage(deep=True).sum() / 1e6)
    df.to_parquet(output_path)
    data_client.close()
