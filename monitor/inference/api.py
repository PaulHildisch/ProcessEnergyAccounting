import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("energy-monitor-inference")


class InferenceRequest:
    def __init__(self, model_pkl: str):
        self.model_path = Path(model_pkl)
        self.model = self._load_model(self.model_path)
        self.features = list(self.model.get("features", []))
        self.weights = np.asarray(self.model.get("weights", []), dtype=float)
        self.scaler = self.model.get("scaler")
        self.static_energy = float(self.model.get("static_energy", 0.0) or 0.0)
        self._validate_model()
        self._log_model_expectations()

    @staticmethod
    def _load_model(model_path: Path):
        with model_path.open("rb") as handle:
            return pickle.load(handle)

    def _validate_model(self) -> None:
        if not self.features:
            raise ValueError(
                "Model artifact does not define a non-empty 'features' list"
            )
        if self.scaler is None:
            raise ValueError("Model artifact does not define 'scaler'")
        if len(self.features) != len(self.weights):
            raise ValueError(
                "Model artifact is invalid: number of features does not match number of weights"
            )

    def _log_model_expectations(self) -> None:
        logger.info("Loaded inference model from %s", self.model_path)
        logger.info("Model expects %d features", len(self.features))
        logger.info(
            "Expected feature input format: flat numeric row with keys in this order: %s",
            ", ".join(str(feature) for feature in self.features),
        )

    def predict_sample(self, sample: dict[str, Any]) -> float:
        feature_row = np.array(
            [[self._to_float(sample.get(feature, 0.0)) for feature in self.features]],
            dtype=float,
        )
        scaled_features = self.scaler.transform(feature_row)
        prediction = float((scaled_features @ self.weights).sum() + self.static_energy)
        return prediction

    def predict_many(self, samples: dict[str, dict[str, Any]]) -> dict[str, float]:
        return {name: self.predict_sample(sample) for name, sample in samples.items()}

    def run_online_estimation(
        self,
        *,
        timestamp,
        interval,
        deltas,
        container_metrics,
        pod_metrics,
        mode,
        exporter=None,
        node="localhost",
    ) -> dict[str, float] | None:
        if not deltas:
            return None

        if mode == "process":
            samples = {
                str(pid): self._flatten_metric_sample(metrics)
                for pid, metrics in deltas.items()
            }
            return self.predict_many(samples)

        if mode == "container":
            if not container_metrics:
                logger.info("No aggregated container metrics available for prediction.")
                return
            samples = {
                container_name: self._flatten_metric_sample(metrics)
                for container_name, metrics in container_metrics.items()
            }
            predictions = self.predict_many(samples)
            for container_name, predicted_energy in predictions.items():
                logger.info(
                    "Predicted container energy: container=%s energy=%s",
                    container_name,
                    predicted_energy,
                )
            if exporter is not None:
                exporter.set_container_energy_predictions(
                    timestamp=timestamp,
                    interval=interval,
                    predictions=predictions,
                    node=node,
                )
            return

        if mode == "pod":
            if not pod_metrics:
                logger.info("No aggregated pod metrics available for prediction.")
                return
            samples = {
                pod_name: self._flatten_metric_sample(metrics)
                for pod_name, metrics in pod_metrics.items()
            }
            predictions = self.predict_many(samples)
            for pod_name, predicted_energy in predictions.items():
                logger.info(
                    "Predicted pod energy: pod=%s energy=%s",
                    pod_name,
                    predicted_energy,
                )
            if exporter is not None:
                exporter.set_pod_energy_predictions(
                    timestamp=timestamp,
                    interval=interval,
                    predictions=predictions,
                    node=node,
                )
            return

        logger.warning("Unknown online estimation mode: %s", mode)

    def _flatten_metric_sample(self, metrics: dict[str, Any]) -> dict[str, Any]:
        sample = {
            key: value
            for key, value in metrics.items()
            if key not in {"pid", "ppid", "name", "syscall_class_deltas"}
        }

        # Backward-compatible aliases for model features trained with delta_* names.
        alias_map = {
            "instructions": "delta_instructions",
            "cycles": "delta_cycles",
            "branch_instructions": "delta_branch_instructions",
            "cache_misses": "delta_cache_misses",
        }
        for source_key, alias_key in alias_map.items():
            if alias_key not in sample and source_key in metrics:
                sample[alias_key] = metrics.get(source_key, 0)

        for cls, count in (metrics.get("syscall_class_deltas", {}) or {}).items():
            sample[f"syscall_class_{cls}"] = count

        for name, count in (metrics.get("fp_op_deltas", {}) or {}).items():
            sample[f"delta_{name}"] = count

        return sample

    @staticmethod
    def _to_float(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
