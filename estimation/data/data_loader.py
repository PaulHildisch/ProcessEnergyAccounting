import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from database.client import DBClient

INFLUX_URL = ""
INFLUX_TOKEN = ""
INFLUX_ORG = ""
INFLUX_BUCKET = ""

load_dotenv()


def load_dataset(client, level, start, stop=None, aggregate_every=None):
    if level == "process":
        # Fetch numeric fields normally (float cast + aggregation).
        df = client.load_data(
            start=start,
            stop=stop,
            aggregate_every=aggregate_every,
            tag_columns=DBClient.DEFAULT_TAG_COLUMNS,
        )
        # cmdline, exe, cwd, cgroup are string fields — they can't survive the float
        # cast in load_data. Fetch them separately as one value per pid and merge.
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
        return client.load_task_data(
            start=start,
            stop=stop,
            aggregate_every=aggregate_every,
        )
    if level == "workflow":
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
        "--influx-url",
        default=os.getenv("INFLUX_URL", INFLUX_URL),
        help="InfluxDB URL",
    )
    parser.add_argument(
        "--influx-token",
        default=os.getenv("INFLUX_TOKEN", INFLUX_TOKEN),
        help="InfluxDB token",
    )
    parser.add_argument(
        "--influx-org",
        default=os.getenv("INFLUX_ORG", INFLUX_ORG),
        help="InfluxDB org",
    )
    parser.add_argument(
        "--influx-bucket",
        default=os.getenv("INFLUX_BUCKET", INFLUX_BUCKET),
        help="InfluxDB bucket",
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
        "--aggregate-every",
        default="1s",
        help="Optional aggregate window passed to Flux, for example 1s or 5s",
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

    args = parser.parse_args()

    output_path = args.output or f"{args.level}_interval_data.parquet"

    data_client = DBClient(
        args.influx_url,
        args.influx_token,
        args.influx_org,
        args.influx_bucket,
    )

    # Parse start/stop into datetime objects for chunking.
    # Accepts RFC3339 timestamps (e.g. 2026-04-15T09:42:00Z) or relative (e.g. -1h).
    def _parse_ts(s):
        if s is None:
            return None
        s = s.strip()
        if s.startswith("-"):
            return None  # relative — can't chunk, fall through to single query
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

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
            print(f"Chunk {chunk_num}/{total_chunks}: {start_str} → {stop_str} ...", flush=True)

            chunk_df = load_dataset(
                client=data_client,
                level=args.level,
                start=start_str,
                stop=stop_str,
                aggregate_every=args.aggregate_every,
            )
            print(f"  → {len(chunk_df)} rows", flush=True)
            if not chunk_df.empty:
                chunks.append(chunk_df)
            chunk_start = chunk_end

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    print(f"\nLoaded {len(df)} rows from InfluxDB for level={args.level}")

    if "interval_energy" in df.columns:
        active = df[df["interval_energy"] > 0]
        print(f"Active interval shape: {active.shape}")

    print("Memory usage (MB):", df.memory_usage(deep=True).sum() / 1e6)
    df.to_parquet(output_path)
    data_client.close()
