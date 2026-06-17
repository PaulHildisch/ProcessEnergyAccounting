import logging
import random

from tabulate import tabulate

MAX_ROWS = 15
MAX_NO_MODEL_METRIC_COLUMNS = 8
IDENTIFIER_COLUMNS = ["PID", "Name"]


def log_process_metrics_table(
    deltas,
    process_energy_predictions=None,
    model_features=None,
):
    if not deltas:
        logging.info("No PIDs with metrics found.")
        return

    per_pid_metrics = []
    for pid, metrics in deltas.items():
        row = {
            "PID": pid,
            "Name": metrics.get("name", ""),
            **_flatten_metric_sample(metrics),
        }
        if process_energy_predictions is not None:
            row["predicted_energy"] = process_energy_predictions.get(str(pid), "")
        row["_metric_total"] = _numeric_total(row, model_features)
        per_pid_metrics.append(row)

    per_pid_metrics = sorted(
        per_pid_metrics,
        key=lambda row: row.get("_metric_total", 0),
        reverse=True,
    )[:MAX_ROWS]

    if model_features:
        headers = _model_headers(model_features, process_energy_predictions)
    else:
        headers = _sample_non_zero_headers(per_pid_metrics)

    ordered_rows = [
        {
            key: row.get(key, 0 if key not in IDENTIFIER_COLUMNS else "")
            for key in headers
        }
        for row in per_pid_metrics
    ]
    logging.info("\n" + tabulate(ordered_rows, headers="keys", tablefmt="github"))


def _model_headers(model_features, process_energy_predictions):
    headers = IDENTIFIER_COLUMNS.copy()
    if process_energy_predictions is not None:
        headers.append("predicted_energy")
    headers.extend(feature for feature in model_features if feature not in headers)
    return headers


def _sample_non_zero_headers(rows):
    non_zero_columns = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {*IDENTIFIER_COLUMNS, "_metric_total", "predicted_energy"}
            and _is_non_zero_number(value)
        }
    )
    sampled_columns = random.sample(
        non_zero_columns,
        min(MAX_NO_MODEL_METRIC_COLUMNS, len(non_zero_columns)),
    )
    sampled_columns.sort()
    return IDENTIFIER_COLUMNS + sampled_columns


def _flatten_metric_sample(metrics):
    sample = {
        key: value
        for key, value in metrics.items()
        if key not in {"pid", "ppid", "name", "syscall_class_deltas", "fp_op_deltas"}
        and not isinstance(value, dict)
    }

    for cls, count in (metrics.get("syscall_class_deltas", {}) or {}).items():
        sample[f"syscall_class_{cls}"] = count

    for name, count in (metrics.get("fp_op_deltas", {}) or {}).items():
        sample[f"delta_{name}"] = count

    return sample


def _numeric_total(row, model_features=None):
    if model_features:
        return sum(
            abs(float(row.get(feature, 0)))
            for feature in model_features
            if _is_number(row.get(feature, 0))
        )
    return sum(
        abs(float(value))
        for key, value in row.items()
        if key not in {*IDENTIFIER_COLUMNS, "predicted_energy"} and _is_number(value)
    )


def _is_non_zero_number(value):
    return _is_number(value) and float(value) != 0.0


def _is_number(value):
    if isinstance(value, bool):
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
