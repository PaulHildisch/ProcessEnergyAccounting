import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Process Monitoring with optional Prometheus Exporter, "
            "InfluxDB and Smart Meter integration"
        )
    )
    parser.add_argument(
        "--use-prometheus-exporter",
        action="store_true",
        help="Enable Prometheus Exporter (disabled unless this flag is passed)",
    )
    parser.add_argument(
        "--exporter-addr",
        default=os.getenv("EXPORTER_ADDR"),
        help="Prometheus Expoter host (default: env EXPORTER_ADDR)",
    )
    parser.add_argument(
        "--exporter-port",
        default=os.getenv("EXPORTER_PORT"),
        help="Prometheus Expoter port (default: env EXPORTER_PORT)",
    )
    parser.add_argument(
        "--exporter-mode",
        choices=["process", "container", "pod"],
        default="process",
        help="Prometheus export granularity (default: process)",
    )
    parser.add_argument(
        "--use-influxdb",
        action="store_true",
        help="Enable InfluxDB integration (disabled unless this flag is passed)",
    )
    parser.add_argument(
        "--influx-url",
        default=os.getenv("INFLUX_URL"),
        help="InfluxDB URL (default: env INFLUX_URL)",
    )
    parser.add_argument(
        "--influx-token",
        default=os.getenv("INFLUX_TOKEN"),
        help="InfluxDB token (default: env INFLUX_TOKEN)",
    )
    parser.add_argument(
        "--influx-org",
        default=os.getenv("INFLUX_ORG"),
        help="InfluxDB org (default: env INFLUX_ORG)",
    )
    parser.add_argument(
        "--influx-bucket",
        default=os.getenv("INFLUX_BUCKET"),
        help="InfluxDB bucket (default: env INFLUX_BUCKET)",
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
        "--use-meter",
        action="store_true",
        help="Enable smart meter integration (disabled unless this flag is passed)",
    )
    parser.add_argument(
        "--meter-host",
        default=os.getenv("SMARTMETER_HOST"),
        help="Smart meter host (default: env SMARTMETER_HOST)",
    )
    parser.add_argument(
        "--meter-user",
        default=os.getenv("SMARTMETER_USER"),
        help="Smart meter username (default: env SMARTMETER_USER)",
    )
    parser.add_argument(
        "--meter-password",
        default=os.getenv("SMARTMETER_PASSWORD"),
        help="Smart meter password (default: env SMARTMETER_PASSWORD)",
    )
    parser.add_argument(
        "--meter-ssl",
        action="store_true",
        help="Enable SSL for smart meter client (optional)",
    )
    parser.add_argument(
        "--meter-sensor-id",
        default="L1",
        help="Sensor id to read from smart meter (default: L1)",
    )
    parser.add_argument(
        "--online-energy-estimation",
        action="store_true",
        help="Placeholder flag for future online energy estimation logic (not yet implemented)",
    )
    parser.add_argument(
        "--docker-integration",
        action="store_true",
        help="Enable metric aggregation for Docker environments (disabled unless this flag is passed)",
    )
    parser.add_argument(
        "--kubernetes-integration",
        action="store_true",
        help="Enable metric aggregation for Kubernetes environments (disabled unless this flag is passed)",
    )
    parser.add_argument(
        "--use-pod-regex",
        action="store_true",
        help="Enable pod name regex filtering via K8S_POD_REGEX loaded from .env",
    )
    parser.add_argument(
        "--kubeconfig",
        help="Path to kubeconfig file",
    )

    parser.add_argument(
        "--model-pkl",
        help="Path to model pickle (.pkl) file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args()
