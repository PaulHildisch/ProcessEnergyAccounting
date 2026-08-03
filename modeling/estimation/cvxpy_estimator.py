import argparse
import itertools
import os
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parent
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))

try:
    from cvxpy_estimator_plots import save_estimator_plots

    _PLOTS_AVAILABLE = True
except ImportError:
    _PLOTS_AVAILABLE = False
    save_estimator_plots = None  # type: ignore[assignment]

DEFAULT_FEATURES = [
    "delta_instructions",
    "delta_cache_misses",
    "delta_branch_instructions",
    "syscall_class_other",
]
DEFAULT_L1_PENALTY = 0.1
DEFAULT_STATIC_PENALTY = 0.0

CV_L1_GRID = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
CV_STATIC_GRID = [0.0, 0.001, 0.01, 0.1, 1.0]
CV_N_SPLITS = 5

TARGET_COLUMN = "interval_energy"
TARGET_UNIT = "J"
TIME_COLUMN = "_time"
TIME_ROUNDING = "1ms"
TEST_SIZE = 0.2
SOLVER = None

MODEL_PATH = SCRIPT_DIR / "pretrained-models" / "model.pkl"
PLOT_OUTPUT_DIR = MODELING_DIR / "plots" / "output"
PLOT_PREFIX = "actual_vs_predicted_interval_energy"


@dataclass
class EstimatorConfig:
    features: list[str]
    l1_penalty: float
    static_penalty: float
    # Weighted static baseline model.
    # "scalar": single fitted constant (default, existing behaviour).
    # "weighted": static = Z @ v where Z = [baseline, n_processes, *static_features].
    static_model: str = "scalar"
    # Extra feature names (from the parquet) to include in Z beyond baseline+n_processes.
    # These are summed per interval, just like dynamic features are.
    static_features: list[str] = field(default_factory=list)


@dataclass
class CVSearchResult:
    best_l1_penalty: float
    best_static_penalty: float
    best_mae: float
    all_results: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CVXPY energy estimator")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to input parquet dataset",
    )
    parser.add_argument(
        "--hostname",
        nargs="+",
        required=True,
        help="Source hostname(s) of the dataset. One or more values are allowed.",
    )
    parser.add_argument(
        "--workload-name",
        required=True,
        help="Workload name used as a subdirectory under the hostname plot directory.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=SCRIPT_DIR / ".env",
        help="Path to .env file (features + penalties)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run cross-validated hyperparameter search before final training",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=CV_N_SPLITS,
        help="Number of time-series CV folds used during --tune (default: 5)",
    )
    return parser.parse_args()


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


def load_config(env_file: Path) -> EstimatorConfig:
    load_dotenv(env_file.resolve(), override=True)
    return EstimatorConfig(
        features=_env_list("EST_FEATURES", DEFAULT_FEATURES),
        l1_penalty=_env_float("EST_L1_PENALTY", DEFAULT_L1_PENALTY),
        static_penalty=_env_float("EST_STATIC_PENALTY", DEFAULT_STATIC_PENALTY),
        static_model=_env_str("EST_STATIC_MODEL", "scalar"),
        static_features=_env_list("EST_STATIC_FEATURES", []),
    )


def load_dataset(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def validate_columns_in_file(path: Path, required_columns: list[str]) -> None:
    """Check required columns exist in a parquet file without loading row data."""
    import pyarrow.parquet as pq

    available = set(pq.read_schema(path).names)
    missing = [col for col in required_columns if col not in available]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def prepare_dataset(df: pd.DataFrame, cfg: EstimatorConfig):
    prepared = df.copy()
    prepared[TIME_COLUMN] = pd.to_datetime(prepared[TIME_COLUMN]).dt.round(
        TIME_ROUNDING
    )
    prepared[cfg.features] = prepared[cfg.features].fillna(0)

    interval_energy = (
        prepared.groupby(TIME_COLUMN, sort=True)[TARGET_COLUMN].first().dropna()
    )
    prepared = prepared[
        prepared[TIME_COLUMN].isin(interval_energy.index.tolist())
    ].copy()
    return prepared, interval_energy


def split_train_test(df: pd.DataFrame, interval_energy: pd.Series):
    time_values = interval_energy.index.sort_values()
    train_times, test_times = train_test_split(
        time_values,
        test_size=TEST_SIZE,
        shuffle=False,
    )

    interval_energy_train = interval_energy.loc[train_times].sort_index()
    interval_energy_test = interval_energy.loc[test_times].sort_index()
    df_train = df[df[TIME_COLUMN].isin(train_times.tolist())].copy()
    df_test = df[df[TIME_COLUMN].isin(test_times.tolist())].copy()
    return df_train, df_test, interval_energy_train, interval_energy_test


def _compute_static_features(
    df: pd.DataFrame,
    times: pd.Index,
    cfg: EstimatorConfig,
) -> pd.DataFrame:
    """Build Z (interval-level static features) from the process-level DataFrame.

    Z always contains:
      - baseline:     constant 1.0 — gives the model an explicit intercept for
                      the static component (replaces the scalar static_energy).
      - n_processes:  count of active processes per interval — captures OS and
                      scheduling overhead that scales with process count.

    Additional columns from cfg.static_features are appended to Z. Hardware
    columns (``hw_*``) are machine/interval-level metadata and use max per
    interval so one-hot values stay 0/1 instead of scaling with process count.
    Other extra columns keep the previous sum aggregation.
    """
    # n_processes: count rows per interval
    n_proc = (
        df[df[TIME_COLUMN].isin(times)]
        .groupby(TIME_COLUMN)
        .size()
        .rename("n_processes")
        .reindex(times)
        .fillna(0)
        .astype(float)
    )

    frames: dict[str, pd.Series] = {
        "baseline": pd.Series(1.0, index=times),
        "n_processes": n_proc,
    }

    if cfg.static_features:
        grouped = df[df[TIME_COLUMN].isin(times)].groupby(TIME_COLUMN)
        for feat in cfg.static_features:
            if feat.startswith("hw_"):
                values = grouped[feat].max()
            else:
                values = grouped[feat].sum()
            frames[feat] = values.reindex(times).fillna(0)

    return pd.DataFrame(frames)


def train_cvxpy_model(
    df_train: pd.DataFrame, interval_energy: pd.Series, cfg: EstimatorConfig
):
    agg = df_train.groupby(TIME_COLUMN)[cfg.features].sum()
    agg = agg.reindex(interval_energy.index).fillna(0)

    scaler = MaxAbsScaler()
    x_matrix = scaler.fit_transform(agg.values)
    y_values = interval_energy.values

    weights = cp.Variable(x_matrix.shape[1])
    static_energy = cp.Variable()

    predictions = x_matrix @ weights + static_energy
    loss = cp.sum_squares(predictions - y_values)
    reg = cfg.l1_penalty * cp.norm1(weights) + cfg.static_penalty * cp.abs(
        static_energy
    )

    problem = cp.Problem(
        cp.Minimize(loss + reg), constraints=[weights >= 0, static_energy >= 0]
    )
    if SOLVER:
        problem.solve(solver=SOLVER)
    else:
        problem.solve()

    if weights.value is None or static_energy.value is None:
        raise RuntimeError(f"Optimization failed with status: {problem.status}")

    return {
        "weights": weights.value,
        "static_energy": float(static_energy.value),
        "scaler": scaler,
        "solver_status": problem.status,
    }


def train_cvxpy_weighted_model(
    df_train: pd.DataFrame, interval_energy: pd.Series, cfg: EstimatorConfig
) -> dict:
    """Train the weighted static baseline model.

    Model: E_interval = X_agg @ w + Z_interval @ v
      - X_agg: per-process features summed per interval (dynamic component)
      - Z_interval: interval-level features [baseline, n_processes, ...]
      - w >= 0 with L1 penalty cfg.l1_penalty
      - v >= 0 with L1 penalty cfg.static_penalty

    Cross-platform use: w (dynamic weights) can be transferred to a new host
    with similar hardware; v (static weights) can be re-estimated from a small
    calibration set on the target host since Z features are always observable.
    """
    # Dynamic component: per-process features aggregated per interval
    agg = df_train.groupby(TIME_COLUMN)[cfg.features].sum()
    agg = agg.reindex(interval_energy.index).fillna(0)

    x_scaler = MaxAbsScaler()
    x_matrix = x_scaler.fit_transform(agg.values)

    # Static component: interval-level features
    z_df = _compute_static_features(df_train, interval_energy.index, cfg)
    z_scaler = MaxAbsScaler()
    z_matrix = z_scaler.fit_transform(z_df.values)
    static_feature_names = list(z_df.columns)

    y_values = interval_energy.values

    w_var = cp.Variable(x_matrix.shape[1])
    v_var = cp.Variable(z_matrix.shape[1])

    predictions = x_matrix @ w_var + z_matrix @ v_var
    loss = cp.sum_squares(predictions - y_values)
    reg = cfg.l1_penalty * cp.norm1(w_var) + cfg.static_penalty * cp.norm1(v_var)

    problem = cp.Problem(cp.Minimize(loss + reg), constraints=[w_var >= 0, v_var >= 0])
    if SOLVER:
        problem.solve(solver=SOLVER)
    else:
        problem.solve()

    if w_var.value is None or v_var.value is None:
        raise RuntimeError(f"Optimization failed with status: {problem.status}")

    return {
        "static_model": "weighted",
        "weights": w_var.value,
        "scaler": x_scaler,
        "static_weights": v_var.value,
        "static_scaler": z_scaler,
        "static_feature_names": static_feature_names,
        # Kept for API compatibility with scalar model consumers.
        "static_energy": float(np.dot(z_scaler.scale_**0, v_var.value)),
        "solver_status": problem.status,
    }


def predict_per_interval(
    df_test: pd.DataFrame, model: dict, cfg: EstimatorConfig
) -> pd.DataFrame:
    predicted = df_test.copy()
    predicted[cfg.features] = predicted[cfg.features].fillna(0)
    predicted[cfg.features] = model["scaler"].transform(predicted[cfg.features])
    predicted["predicted_process_energy"] = (
        predicted[cfg.features].values @ model["weights"]
    )

    interval_pred = (
        predicted.groupby(TIME_COLUMN)["predicted_process_energy"].sum().reset_index()
    )
    interval_pred["predicted_total_energy"] = (
        interval_pred["predicted_process_energy"] + model["static_energy"]
    )
    return interval_pred


def predict_per_interval_weighted(
    df_test: pd.DataFrame, model: dict, cfg: EstimatorConfig
) -> pd.DataFrame:
    """Generate interval-level predictions with the weighted static model."""
    predicted = df_test.copy()
    predicted[cfg.features] = predicted[cfg.features].fillna(0)
    predicted[cfg.features] = model["scaler"].transform(predicted[cfg.features])
    predicted["predicted_process_energy"] = (
        predicted[cfg.features].values @ model["weights"]
    )

    interval_pred = (
        predicted.groupby(TIME_COLUMN)["predicted_process_energy"].sum().reset_index()
    )

    # Compute Z for each test interval
    test_times = interval_pred[TIME_COLUMN]
    z_df = _compute_static_features(df_test, test_times, cfg)
    z_scaled = model["static_scaler"].transform(z_df.values)
    static_contributions = z_scaled @ model["static_weights"]

    interval_pred["static_energy"] = static_contributions
    interval_pred["predicted_total_energy"] = (
        interval_pred["predicted_process_energy"] + interval_pred["static_energy"]
    )
    return interval_pred


def evaluate_predictions(
    predictions: pd.DataFrame, interval_energy: pd.Series
) -> tuple[pd.DataFrame, dict]:
    evaluation = predictions.merge(
        interval_energy.rename(TARGET_COLUMN),
        left_on=TIME_COLUMN,
        right_index=True,
    )
    actual = evaluation[TARGET_COLUMN]
    predicted = evaluation["predicted_total_energy"]

    mae = mean_absolute_error(actual, predicted)
    mean_energy = float(actual.mean())
    metrics = {
        "r2": r2_score(actual, predicted),
        "mae": mae,
        "mean_energy": mean_energy,
        "mae_pct": 100 * mae / mean_energy if mean_energy else 0.0,
    }
    return evaluation, metrics


def _sanitize_hostname(hostname: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", hostname.strip())
    normalized = normalized.strip("-._")
    return normalized or "unknown"


def build_model_path(hostnames: list[str]) -> Path:
    ordered_unique = list(dict.fromkeys(hostnames))
    safe_names = [_sanitize_hostname(hostname) for hostname in ordered_unique]
    filename = "_".join(safe_names) + ".pkl"
    return MODEL_PATH.parent / filename


def save_model(model: dict, cfg: EstimatorConfig, hostnames: list[str]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_path = build_model_path(hostnames)
    payload = {
        "weights": model["weights"],
        "static_energy": model.get("static_energy", 0.0),
        "scaler": model["scaler"],
        "features": cfg.features,
        "target": TARGET_COLUMN,
        "target_unit": TARGET_UNIT,
        "time_column": TIME_COLUMN,
        "l1_penalty": cfg.l1_penalty,
        "static_penalty": cfg.static_penalty,
        "solver": SOLVER,
        "source_hostnames": list(dict.fromkeys(hostnames)),
        # Weighted static model fields (None for scalar mode)
        "static_model": cfg.static_model,
        "static_weights": model.get("static_weights"),
        "static_scaler": model.get("static_scaler"),
        "static_feature_names": model.get("static_feature_names"),
        "static_features_cfg": cfg.static_features,
    }
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle)
    print(f"Model saved to {model_path}")


def compute_learning_curve(
    df_train: pd.DataFrame,
    interval_train: pd.Series,
    df_test: pd.DataFrame,
    interval_test: pd.Series,
    cfg: EstimatorConfig,
    n_points: int = 8,
) -> list[dict]:
    """
    Evaluate model performance on a fixed test set as the training set grows.
    Uses time-ordered prefixes to respect the temporal split and avoid leakage.
    Each point trains a fresh model on the first `fraction` of training intervals
    and measures MAE / R² on the full, unchanged test set.
    """
    time_values = interval_train.index.sort_values()
    n_total = len(time_values)
    results = []
    print(f"Computing learning curve ({n_points} points) …")
    for i in range(1, n_points + 1):
        subset_size = max(2, int(n_total * i / n_points))
        subset_times = time_values[:subset_size]
        subset_energy = interval_train.loc[subset_times]
        subset_train = df_train[
            df_train[TIME_COLUMN].isin(subset_times.tolist())
        ].copy()
        try:
            if cfg.static_model == "weighted":
                sub_model = train_cvxpy_weighted_model(subset_train, subset_energy, cfg)
                sub_preds = predict_per_interval_weighted(df_test, sub_model, cfg)
            else:
                sub_model = train_cvxpy_model(subset_train, subset_energy, cfg)
                sub_preds = predict_per_interval(df_test, sub_model, cfg)
            _, sub_metrics = evaluate_predictions(sub_preds, interval_test)
            results.append(
                {
                    "train_size": subset_size,
                    "fraction": i / n_points,
                    "mae": sub_metrics["mae"],
                    "r2": sub_metrics["r2"],
                    "mae_pct": sub_metrics["mae_pct"],
                }
            )
            print(
                f"  {subset_size:>5} intervals "
                f"→ MAE={sub_metrics['mae']:.4f}  R²={sub_metrics['r2']:.4f}"
            )
        except RuntimeError as exc:
            print(f"  {subset_size:>5} intervals → skipped ({exc})")
    return results


def run_cv_search(
    df_train: pd.DataFrame,
    interval_train: pd.Series,
    cfg: EstimatorConfig,
    l1_grid: list[float] | None = None,
    static_grid: list[float] | None = None,
    n_splits: int = CV_N_SPLITS,
) -> CVSearchResult:
    """Grid search over (l1_penalty, static_penalty) using TimeSeriesSplit CV.

    Aggregations and scaling are precomputed once per fold so the expensive
    groupby only runs n_splits times regardless of grid size.  The remaining
    work is purely CVXPY solves (~0.1 s each).

    L1 regularisation also acts as implicit feature selection: features whose
    weight goes to zero under the best l1_penalty are effectively excluded.
    """
    from sklearn.model_selection import TimeSeriesSplit

    if l1_grid is None:
        l1_grid = CV_L1_GRID
    if static_grid is None:
        static_grid = CV_STATIC_GRID

    tscv = TimeSeriesSplit(n_splits=n_splits)
    time_values = interval_train.index.sort_values()
    n_combos = len(l1_grid) * len(static_grid)

    print(
        f"CV search: {len(l1_grid)} l1 × {len(static_grid)} static_penalty values "
        f"× {n_splits} folds = {n_combos * n_splits} total fits"
    )

    # Precompute scaled X / y arrays per fold — avoids repeating groupby for
    # every hyperparameter combination.
    print("Precomputing fold aggregations…")
    fold_data: list[tuple] = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(time_values)):
        cv_train_times = time_values[train_idx]
        cv_val_times = time_values[val_idx]

        y_train = interval_train.reindex(cv_train_times).values
        y_val = interval_train.reindex(cv_val_times).values

        agg_train = (
            df_train[df_train[TIME_COLUMN].isin(cv_train_times.tolist())]
            .groupby(TIME_COLUMN)[cfg.features]
            .sum()
            .reindex(cv_train_times)
            .fillna(0)
        )
        agg_val = (
            df_train[df_train[TIME_COLUMN].isin(cv_val_times.tolist())]
            .groupby(TIME_COLUMN)[cfg.features]
            .sum()
            .reindex(cv_val_times)
            .fillna(0)
        )

        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(agg_train.values)
        X_val = scaler.transform(agg_val.values)

        fold_data.append((X_train, y_train, X_val, y_val))
        print(
            f"  Fold {fold_idx + 1}/{n_splits}: "
            f"{len(y_train)} train / {len(y_val)} val intervals"
        )

    # Grid search — only CVXPY solves from here on.
    print(f"\nSearching {n_combos} combinations…")
    all_results: list[dict] = []
    for combo_idx, (l1, sp) in enumerate(itertools.product(l1_grid, static_grid), 1):
        fold_maes: list[float] = []
        for X_train, y_train, X_val, y_val in fold_data:
            try:
                w = cp.Variable(X_train.shape[1])
                s = cp.Variable()
                loss = cp.sum_squares(X_train @ w + s - y_train)
                reg = l1 * cp.norm1(w) + sp * cp.abs(s)
                prob = cp.Problem(cp.Minimize(loss + reg), constraints=[w >= 0, s >= 0])
                if SOLVER:
                    prob.solve(solver=SOLVER)
                else:
                    prob.solve()

                if w.value is None:
                    fold_maes.append(float("inf"))
                    continue

                val_preds = X_val @ w.value + float(s.value)
                fold_maes.append(float(np.mean(np.abs(val_preds - y_val))))
            except Exception:
                fold_maes.append(float("inf"))

        mean_mae = float(np.mean(fold_maes))
        std_mae = float(np.std(fold_maes))
        all_results.append(
            {
                "l1_penalty": l1,
                "static_penalty": sp,
                "mean_mae": mean_mae,
                "std_mae": std_mae,
            }
        )
        print(
            f"  [{combo_idx:>3}/{n_combos}]  "
            f"l1={l1:.4g}  static={sp:.4g}  "
            f"→  CV MAE = {mean_mae:.2f} ± {std_mae:.2f} {TARGET_UNIT}"
        )

    best = min(all_results, key=lambda x: x["mean_mae"])
    return CVSearchResult(
        best_l1_penalty=best["l1_penalty"],
        best_static_penalty=best["static_penalty"],
        best_mae=best["mean_mae"],
        all_results=all_results,
    )


def print_run_summary(cfg: EstimatorConfig, metrics: dict, model: dict) -> None:
    print("Learned dynamic weights:")
    for feature, weight in zip(cfg.features, model["weights"]):
        print(f"  {feature}: {weight:.4e}")

    if cfg.static_model == "weighted" and model.get("static_weights") is not None:
        print("\nLearned static weights (Z @ v):")
        for fname, weight in zip(
            model["static_feature_names"], model["static_weights"]
        ):
            print(f"  {fname}: {weight:.4e}")
    else:
        print(f"Static energy component (scalar): {model['static_energy']:.4f}")

    print(f"\nSolver status: {model['solver_status']}")
    print(f"R² (interval-level): {metrics['r2']:.4f}")
    print(f"MAE (interval-level): {metrics['mae']:.4f}")
    print(f"Mean interval energy: {metrics['mean_energy']:.4f}")
    print(f"MAE (% of mean): {metrics['mae_pct']:.2f}%")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.env_file)

    required_columns = list(
        dict.fromkeys([TIME_COLUMN, TARGET_COLUMN, *cfg.features, *cfg.static_features])
    )
    validate_columns_in_file(args.data, required_columns)
    dataset = load_dataset(args.data, columns=required_columns)

    prepared, interval_energy = prepare_dataset(dataset, cfg)
    print(f"Number of intervals with energy: {len(interval_energy)}")
    print(f"Process rows after filtering: {len(prepared)}")

    df_train, df_test, interval_train, interval_test = split_train_test(
        prepared, interval_energy
    )

    # Optional hyperparameter search (scalar model only)
    cv_result: CVSearchResult | None = None
    if args.tune:
        if cfg.static_model != "scalar":
            print(
                "Warning: --tune is only supported for static_model='scalar'. Skipping."
            )
        else:
            cv_result = run_cv_search(
                df_train, interval_train, cfg, n_splits=args.cv_splits
            )
            print(
                f"\nBest hyperparameters found by CV:\n"
                f"  l1_penalty     = {cv_result.best_l1_penalty}\n"
                f"  static_penalty = {cv_result.best_static_penalty}\n"
                f"  CV MAE         = {cv_result.best_mae:.4f} {TARGET_UNIT}"
            )
            cfg = EstimatorConfig(
                features=cfg.features,
                l1_penalty=cv_result.best_l1_penalty,
                static_penalty=cv_result.best_static_penalty,
                static_model=cfg.static_model,
                static_features=cfg.static_features,
            )

    if cfg.static_model == "weighted":
        print(
            f"Static model: weighted  "
            f"(Z features: baseline, n_processes"
            + (f", {cfg.static_features}" if cfg.static_features else "")
            + ")"
        )
        model = train_cvxpy_weighted_model(df_train, interval_train, cfg)
        predictions = predict_per_interval_weighted(df_test, model, cfg)
    else:
        print("Static model: scalar")
        model = train_cvxpy_model(df_train, interval_train, cfg)
        predictions = predict_per_interval(df_test, model, cfg)
    evaluation, metrics = evaluate_predictions(predictions, interval_test)

    print_run_summary(cfg, metrics, model)
    save_model(model, cfg, args.hostname)

    learning_curve = compute_learning_curve(
        df_train, interval_train, df_test, interval_test, cfg
    )

    if _PLOTS_AVAILABLE:
        saved_plot_paths = save_estimator_plots(
            evaluation_df=evaluation,
            output_dir=PLOT_OUTPUT_DIR,
            prefix=PLOT_PREFIX,
            time_column=TIME_COLUMN,
            target_column=TARGET_COLUMN,
            model_weights=model["weights"],
            features=cfg.features,
            learning_curve_data=learning_curve,
            cv_results=cv_result.all_results if cv_result is not None else None,
            hostname=args.hostname,
            workload_name=args.workload_name,
        )
        for path in saved_plot_paths:
            print(f"Saved plot: {path}")
    else:
        print("(plots skipped — modeling/plots module not found)")


if __name__ == "__main__":
    main()
