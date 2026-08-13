import os
from argparse import Namespace
from pathlib import Path

import dotenv


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def load_env_values(env_path: Path) -> dict[str, str]:
    # Populate process env for components that directly call os.getenv (e.g. K8S regex).
    dotenv.load_dotenv(dotenv_path=env_path, override=False)

    values = {
        key: value
        for key, value in dotenv.dotenv_values(dotenv_path=env_path).items()
        if value is not None
    }

    # Fallback parser for edge cases where dotenv parsing unexpectedly returns empty.
    if not values and env_path.exists():
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key:
                    values[key] = value
        except OSError:
            pass

    return values


def apply_env_defaults(args: Namespace, env_values: dict[str, str]) -> None:
    args.influx_url = _first_non_empty(
        args.influx_url,
        env_values.get("INFLUX_URL"),
        os.getenv("INFLUX_URL"),
    )
    args.influx_token = _first_non_empty(
        args.influx_token,
        env_values.get("INFLUX_TOKEN"),
        os.getenv("INFLUX_TOKEN"),
    )
    args.influx_org = _first_non_empty(
        args.influx_org,
        env_values.get("INFLUX_ORG"),
        os.getenv("INFLUX_ORG"),
    )
    args.influx_bucket = _first_non_empty(
        args.influx_bucket,
        env_values.get("INFLUX_BUCKET"),
        os.getenv("INFLUX_BUCKET"),
    )

    args.meter_host = _first_non_empty(
        args.meter_host,
        env_values.get("SMARTMETER_HOST"),
        os.getenv("SMARTMETER_HOST"),
    )
    args.meter_user = _first_non_empty(
        args.meter_user,
        env_values.get("SMARTMETER_USER"),
        os.getenv("SMARTMETER_USER"),
    )
    args.meter_password = _first_non_empty(
        args.meter_password,
        env_values.get("SMARTMETER_PASSWORD"),
        os.getenv("SMARTMETER_PASSWORD"),
    )
