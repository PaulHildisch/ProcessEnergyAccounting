#!/usr/bin/env python3
"""
Standalone script to export power meter readings from InfluxDB.
Can be called directly or via bash wrapper with --session-dir.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

load_dotenv()


class PowerMeterClient:
    """Minimal InfluxDB client for power meter readings."""

    def __init__(self, url, token, org, bucket):
        self.client = InfluxDBClient(url=url, token=token, org=org, timeout=360_000)
        self.query_api = self.client.query_api()
        self.bucket = bucket
        self.org = org

    def load_power_readings(self, start, stop):
        """Load power meter readings from InfluxDB."""
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start}, stop: {stop})
          |> filter(fn: (r) => r["_measurement"] == "power_meter_readings")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''

        tables = self.query_api.query(query, org=self.org)

        records = []
        for table in tables:
            for record in table.records:
                row = {
                    "time": record.get_time(),
                    "power_w": record.values.get("power_w"),
                    "energy_j": record.values.get("energy_j"),
                    "n_samples": record.values.get("n_samples"),
                }
                # Include any tags
                for key, value in record.values.items():
                    if not key.startswith("_") and key not in [
                        "power_w",
                        "energy_j",
                        "n_samples",
                    ]:
                        row[key] = value
                records.append(row)

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("time").reset_index(drop=True)
        return df

    def close(self):
        self.client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Export power meter readings from InfluxDB to parquet"
    )
    parser.add_argument(
        "--session-dir",
        help="Session directory containing session_start.txt and session_stop.txt",
    )
    parser.add_argument(
        "--influx-url",
        default=os.getenv("INFLUX_URL"),
        help="InfluxDB URL",
    )
    parser.add_argument(
        "--influx-token",
        default=os.getenv("INFLUX_TOKEN"),
        help="InfluxDB token",
    )
    parser.add_argument(
        "--influx-org",
        default=os.getenv("INFLUX_ORG"),
        help="InfluxDB org",
    )
    parser.add_argument(
        "--influx-bucket",
        default=os.getenv("INFLUX_BUCKET"),
        help="InfluxDB bucket",
    )
    parser.add_argument(
        "--start",
        help="Start time (RFC3339 or relative like -1h)",
    )
    parser.add_argument(
        "--stop",
        help="Stop time (RFC3339 or relative like now())",
    )
    parser.add_argument(
        "--output",
        help="Output parquet file path",
    )

    args = parser.parse_args()

    # Handle session-dir mode
    if args.session_dir:
        session_path = Path(args.session_dir)
        start_file = session_path / "session_start.txt"
        stop_file = session_path / "session_stop.txt"

        if not start_file.exists() or not stop_file.exists():
            print(
                f"Error: Session timestamps not found in {args.session_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

        args.start = start_file.read_text().strip()
        args.stop = stop_file.read_text().strip()

        if not args.output:
            args.output = str(
                session_path / "datasets" / "node_power_readings.parquet"
            )

    # Validate required args
    if not args.start or not args.stop:
        parser.error("Either --session-dir or both --start and --stop are required")

    if not all(
        [args.influx_url, args.influx_token, args.influx_org, args.influx_bucket]
    ):
        parser.error(
            "InfluxDB connection parameters are required (URL, token, org, bucket)"
        )

    if not args.output:
        args.output = "power_readings.parquet"

    print(f"Connecting to InfluxDB: {args.influx_url}")
    print(f"Time range: {args.start} → {args.stop}")

    client = PowerMeterClient(
        args.influx_url,
        args.influx_token,
        args.influx_org,
        args.influx_bucket,
    )

    df = client.load_power_readings(args.start, args.stop)

    print(f"\nLoaded {len(df)} power readings")
    if not df.empty:
        print(f"Time range: {df['time'].min()} → {df['time'].max()}")
        if "power_w" in df.columns:
            print(
                f"Power (W): mean={df['power_w'].mean():.2f}, max={df['power_w'].max():.2f}"
            )
        if "energy_j" in df.columns:
            print(f"Total energy (J): {df['energy_j'].sum():.2f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output)
    print(f"\nExported power readings to {args.output}")

    client.close()


if __name__ == "__main__":
    main()
