"""
Simple power meter reader that samples power consumption from a smart meter
and writes the data to InfluxDB.

Standalone script with no external dependencies (except standard libraries and influxdb-client).
"""

import argparse
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point, WritePrecision
from requests.auth import HTTPBasicAuth

# ========================= Configuration =========================

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "my-super-secret-auth-token"
INFLUX_ORG = "myorg"
INFLUX_BUCKET = "mybucket"
MEASUREMENT = "power_meter_readings"

load_dotenv()


# ========================= Smart Meter Client =========================


class SmartMeterAPIClient:
    """Client for fetching smart meter data via HTTP JSON."""

    DESCR = 0x10000
    VALUES = 0x4000
    EXTENDED = 0x800000

    SENSOR_TYPE_MAP = {
        1: "Line power meter",
        9: "Line power meter with residual current",
        8: "Outlet power meter",
        7: "Digital Inputs",
        12: "Bender RCMB Module",
        20: "System Data (sensor group)",
        51: "Temperature Sensor",
        52: "Temperature/Humidity Sensor",
        53: "Temperature/Humidity/AirPressure Sensor",
        101: "Bank (eFuses Port-groups) Sensor",
        102: "DC Power Sources",
    }

    def __init__(
        self,
        host: str,
        ssl: bool = True,
        timeout: int = 10,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        scheme = "https" if ssl else "http"
        self.base_url = f"{scheme}://{host}/status.json"
        self.timeout = timeout
        self.auth = HTTPBasicAuth(username, password) if username else None

    def _fetch(self):
        components = self.DESCR + self.VALUES + self.EXTENDED
        params = {"components": components}
        resp = requests.get(
            self.base_url,
            params=params,
            verify=False,
            auth=self.auth,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_sensor_data(self):
        """Retrieve sensor readings."""
        data = self._fetch()
        descr_map = {d["type"]: d for d in data.get("sensor_descr", [])}
        readings = []

        for entry in data.get("sensor_values", []):
            t = entry.get("type")
            descr = descr_map.get(t, {})
            props = descr.get("properties", [])

            if "fields" in descr:
                fields = descr["fields"]
                for idx, prop in enumerate(props):
                    raw_vals = entry["values"][idx]
                    readings.append(
                        {
                            "id": prop["id"],
                            "name": prop.get("name"),
                            "type_code": t,
                            "type_name": self.SENSOR_TYPE_MAP.get(t, "Unknown"),
                            "data": {
                                f["name"]: raw_vals[i].get("v")
                                for i, f in enumerate(fields)
                            },
                        }
                    )

            elif "groups" in descr:
                groups = descr["groups"]
                for prop_idx, prop in enumerate(props):
                    flat = {}
                    prop_values = entry["values"][prop_idx]
                    for g_idx, group in enumerate(groups):
                        group_values = prop_values[g_idx]
                        if not group_values:
                            continue
                        instance_vals = group_values[0]
                        for f_idx, field in enumerate(group["fields"]):
                            if f_idx < len(instance_vals):
                                flat[field["name"]] = instance_vals[f_idx].get("v")
                            else:
                                flat[field["name"]] = None
                    readings.append(
                        {
                            "id": prop["id"],
                            "name": prop.get("name"),
                            "type_code": t,
                            "type_name": self.SENSOR_TYPE_MAP.get(t, "Unknown"),
                            "data": flat,
                        }
                    )

        return readings


# ========================= InfluxDB Client =========================


class DBClient:
    """Simple InfluxDB writer."""

    def __init__(self, url, token, org, bucket):
        self.client = InfluxDBClient(url=url, token=token, org=org, timeout=360_000)
        self.write_api = self.client.write_api()
        self.bucket = bucket
        self.org = org

    def close(self):
        if self.client:
            self.client.close()
            self.client = None
            self.write_api = None


# ========================= Power Monitor =========================


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class IntervalResult:
    timestamp: float
    duration: float
    meter_power_w: float
    meter_energy_j: float
    meter_samples: int


class PowerMonitor:
    def __init__(
        self,
        interval=1.0,
        sample_rate=0.1,
        db_client=None,
        meter_client=None,
        meter_sensor_id="L1",
        node=None,
    ):
        self.interval = interval
        self.sample_rate = sample_rate
        self.db_client = db_client
        self.meter_client = meter_client
        self.sensor_ids = {sid.strip() for sid in meter_sensor_id.split(",")}
        self.node = node or socket.gethostname()
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
            meter_samples = self._sample_interval(interval_start)
            interval_end = time.time()

            result = self._build_result(interval_start, interval_end, meter_samples)
            self._log(result)
            self._write(result)

    def _sample_interval(self, interval_start):
        """Poll smart meter repeatedly until the interval window closes."""
        meter_samples = []

        while (time.time() - interval_start) < self.interval:
            sample_start = time.time()

            try:
                meter_data = self.meter_client.get_sensor_data()
                matched = [s for s in meter_data if s["id"] in self.sensor_ids]
                powers = [
                    s["data"].get("ActivePower")
                    for s in matched
                    if s["data"].get("ActivePower") is not None
                ]
                if powers:
                    meter_samples.append(sum(powers))
            except Exception as exc:
                print(f"[WARN] Smart meter read failed: {exc}")

            sleep_time = self.sample_rate - (time.time() - sample_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        return meter_samples

    def _build_result(self, interval_start, interval_end, meter_samples):
        duration = interval_end - interval_start
        avg_meter_w = sum(meter_samples) / len(meter_samples) if meter_samples else 0.0
        meter_energy_j = avg_meter_w * duration

        return IntervalResult(
            timestamp=interval_end,
            duration=duration,
            meter_power_w=avg_meter_w,
            meter_energy_j=meter_energy_j,
            meter_samples=len(meter_samples),
        )

    def _log(self, r: IntervalResult):
        print(
            f"[{time.strftime('%X')}]"
            f"  interval={r.duration:.2f}s"
            f"  meter: {r.meter_power_w:.1f}W"
            f"  energy: {r.meter_energy_j:.1f}J"
            f"  samples: {r.meter_samples}",
            flush=True,
        )

    def _write(self, r: IntervalResult):
        point = (
            Point(MEASUREMENT)
            .tag("node", self.node)
            .field("interval", float(r.duration))
            .field("power_w", float(r.meter_power_w))
            .field("energy_j", float(r.meter_energy_j))
            .field("n_samples", int(r.meter_samples))
            .time(int(r.timestamp * 1e9), WritePrecision.NS)
        )
        self.db_client.write_api.write(bucket=self.db_client.bucket, record=point)


# ========================= Main =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor power consumption from smart meter and write to InfluxDB."
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
        help="Length of each measurement window in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="How often to poll the smart meter within each interval in seconds (default: same as --interval)",
    )
    parser.add_argument(
        "--meter-host",
        default=os.getenv("SMARTMETER_HOST"),
        help="Smart meter hostname or IP",
    )
    parser.add_argument(
        "--meter-user",
        default=os.getenv("SMARTMETER_USER"),
        help="Smart meter username",
    )
    parser.add_argument(
        "--meter-password",
        default=os.getenv("SMARTMETER_PASSWORD"),
        help="Smart meter password",
    )
    parser.add_argument(
        "--meter-ssl",
        action="store_true",
        default=env_flag("SMARTMETER_SSL", False),
        help="Use HTTPS for smart meter",
    )
    parser.add_argument(
        "--meter-sensor-id",
        default="L1",
        help="Comma-separated sensor id(s) to read from the smart meter (default: L1)",
    )
    parser.add_argument(
        "--node",
        default=None,
        help="Node label written as a tag to InfluxDB (default: hostname)",
    )

    args = parser.parse_args()
    sample_rate = args.sample_rate if args.sample_rate is not None else args.interval

    # Parse sensor IDs
    sensor_ids = [sid.strip() for sid in args.meter_sensor_id.split(",")]

    print(
        f"Starting PowerMonitor: interval={args.interval}s  sample_rate={sample_rate}s"
    )
    print(
        f"InfluxDB: {args.influx_url}  org={args.influx_org}  bucket={args.influx_bucket}  measurement={MEASUREMENT}"
    )
    print(f"Smart meter: {args.meter_host}  sensor(s): {', '.join(sensor_ids)}")
    if len(sensor_ids) > 1:
        print(
            f"  Note: Power from {len(sensor_ids)} sensors will be summed per interval"
        )

    db_client = DBClient(
        args.influx_url, args.influx_token, args.influx_org, args.influx_bucket
    )
    meter_client = SmartMeterAPIClient(
        host=args.meter_host,
        ssl=args.meter_ssl,
        username=args.meter_user,
        password=args.meter_password,
    )

    monitor = PowerMonitor(
        interval=args.interval,
        sample_rate=sample_rate,
        db_client=db_client,
        meter_client=meter_client,
        meter_sensor_id=args.meter_sensor_id,
        node=args.node,
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
