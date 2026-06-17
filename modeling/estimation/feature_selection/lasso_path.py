"""
Regularization path and stability selection — inspection tool.

Purpose
-------
This script is a *diagnostic* tool for feature selection.  It sweeps the L1
penalty over a log-spaced grid and shows which features drop to zero first,
giving a penalty-vs-importance picture for both the dynamic feature weights (w)
and, optionally, the static baseline weights (v).  Use the outputs to:

  1. Identify which features are robust across the full penalty range.
  2. Decide on a candidate feature set before running best_subset.py.
  3. Inspect how aggressively the static component needs to be penalised.

Production model
----------------
The actual scalar and weighted static models used for fitting and inference
live in ``../cvxpy_estimator.py``.  The weighted static model there is
controlled by ``EST_STATIC_MODEL=weighted`` in the ``.env`` file.

This script re-implements the same CVXPY optimisation locally because it
pre-aggregates the data once and reuses the matrices across 2 500+ solves —
a different data pipeline from the estimator's one-shot training path.

Usage
-----
    # Dynamic features only (default)
    python lasso_path.py --data PATH --penalty-max 5000

    # Also inspect the weighted static component
    python lasso_path.py --data PATH --penalty-max 5000 --static-model weighted
"""

import argparse
import gc
import os
from dataclasses import dataclass, field
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

CANDIDATE_FEATURES = [
    "delta_cpu_ns",
    "delta_cycles",
    "delta_instructions",
    "delta_cache_misses",
    "delta_branch_instructions",
    "delta_io_bytes",
    "delta_net_send_bytes",
    "context_switches",
    "syscall_count",
    "delta_rss_memory",
    "syscall_class_file",
    "syscall_class_network",
    "syscall_class_memory",
    "syscall_class_process",
    "syscall_class_other",
    "syscall_class_sched",
    "syscall_class_signal",
    "syscall_class_time",
]


@dataclass
class DatasetSplit:
    # Pre-aggregated interval-level matrices (intervals × features).
    # Raw process-level DataFrames are discarded after loading to avoid
    # duplicating large frames on every train_and_evaluate / _bootstrap_train call.
    x_train_agg: pd.DataFrame
    x_test_agg: pd.DataFrame
    interval_energy_train: pd.Series
    interval_energy_test: pd.Series
    available_features: list[str]
    # Interval-level features for the weighted static model (--static-model weighted).
    # None when using scalar mode.
    z_train_agg: pd.DataFrame | None = field(default=None)
    z_test_agg: pd.DataFrame | None = field(default=None)
    static_feature_names: list[str] | None = field(default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LASSO regularization path and stability selection for the energy model"
    )
    parser.add_argument(
        "--data", default="runs/benchmark-siena06-v6/process_interval_data.parquet"
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--penalties",
        type=int,
        default=50,
        help="Number of log-spaced penalty values",
    )
    parser.add_argument("--penalty-min", type=float, default=0.001)
    parser.add_argument("--penalty-max", type=float, default=50.0)
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=50,
        help="Bootstrap iterations for stability selection",
    )
    parser.add_argument(
        "--bootstrap-fraction",
        type=float,
        default=0.5,
        help="Fraction of training intervals per bootstrap sample",
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=1e-4,
        help="Weight threshold below which a feature is considered inactive",
    )
    parser.add_argument("--output-dir", type=str, default="plots")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--static-model",
        choices=["scalar", "weighted"],
        default="scalar",
        help=(
            "Static baseline model. 'scalar': single fitted constant (default). "
            "'weighted': baseline = Z @ v where Z contains interval-level features "
            "(n_processes, constant) with their own L1 penalty."
        ),
    )
    parser.add_argument(
        "--static-penalty-ratio",
        type=float,
        default=1.0,
        help=(
            "Ratio of static L1 penalty to dynamic L1 penalty "
            "(static_penalty = ratio * l1_penalty). Only used with --static-model weighted. "
            "Default: 1.0 (same penalty for both components)."
        ),
    )
    parser.add_argument(
        "--static-features",
        nargs="+",
        default=None,
        help=(
            "Additional CANDIDATE_FEATURES to include in Z (interval-level static "
            "component). These are used AS WELL AS the default baseline+n_processes. "
            "Useful for including 'dead' dynamic features as static predictors."
        ),
    )
    return parser.parse_args()


def load_and_split_data(args: argparse.Namespace) -> DatasetSplit:
    print(f"Loading {args.data} ...")

    # Read only the columns we need — parquet files often contain many more
    # metadata columns that would waste gigabytes of memory on load.
    file_cols = set(pq.read_schema(args.data).names)
    cols_needed = {"_time", "interval_energy"} | set(CANDIDATE_FEATURES)
    cols_to_load = [c for c in file_cols if c in cols_needed]
    df = pd.read_parquet(args.data, columns=cols_to_load)
    df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")

    available = [feature for feature in CANDIDATE_FEATURES if feature in df.columns]
    missing = [feature for feature in CANDIDATE_FEATURES if feature not in df.columns]
    if missing:
        print(f"Skipping (not in dataset): {missing}")

    df[available] = df[available].fillna(0)

    interval_energy_all = (
        df[["_time", "interval_energy"]]
        .dropna()
        .drop_duplicates("_time")
        .set_index("_time")["interval_energy"]
    )
    df = df[df["_time"].isin(interval_energy_all.index)]

    time_values = interval_energy_all.index.sort_values()
    train_times, test_times = train_test_split(
        time_values,
        test_size=args.test_size,
        shuffle=False,
    )

    interval_energy_train = interval_energy_all.loc[train_times].sort_index()
    interval_energy_test = interval_energy_all.loc[test_times].sort_index()

    # Aggregate to interval level once here and discard process-level rows.
    # This avoids copying the large process DataFrame on every solver call
    # (50 penalties × 50 bootstrap = 2 500 calls), which is the OOM source.
    x_train_agg = (
        df[df["_time"].isin(train_times)]
        .groupby("_time")[available]
        .sum()
        .reindex(interval_energy_train.index)
        .fillna(0)
    )
    x_test_agg = (
        df[df["_time"].isin(test_times)]
        .groupby("_time")[available]
        .sum()
        .reindex(interval_energy_test.index)
        .fillna(0)
    )

    # --- static features for weighted mode ---
    z_train_agg: pd.DataFrame | None = None
    z_test_agg: pd.DataFrame | None = None
    static_feature_names: list[str] | None = None

    if args.static_model == "weighted":
        # n_processes: count of active processes per interval.
        # Derived from process-level row count before aggregation.
        n_proc_train = (
            df[df["_time"].isin(train_times)]
            .groupby("_time")
            .size()
            .rename("n_processes")
            .reindex(interval_energy_train.index)
            .fillna(0)
            .astype(float)
        )
        n_proc_test = (
            df[df["_time"].isin(test_times)]
            .groupby("_time")
            .size()
            .rename("n_processes")
            .reindex(interval_energy_test.index)
            .fillna(0)
            .astype(float)
        )

        # Build Z: constant baseline + n_processes + any user-specified features.
        extra = [f for f in (args.static_features or []) if f in available]
        static_cols = ["baseline", "n_processes"] + extra
        static_feature_names = static_cols

        def _build_z(n_proc: pd.Series, x_agg: pd.DataFrame) -> pd.DataFrame:
            frames = {
                "baseline": pd.Series(1.0, index=n_proc.index),
                "n_processes": n_proc,
            }
            for feat in extra:
                frames[feat] = x_agg[feat]
            return pd.DataFrame(frames)

        z_train_agg = _build_z(n_proc_train, x_train_agg)
        z_test_agg = _build_z(n_proc_test, x_test_agg)

        print(f"Static features (Z): {static_cols}")

    # Discard the full process-level DataFrame now that we have the aggregated
    # matrices — keeps only the small interval-level data in memory going forward.
    del df, interval_energy_all
    gc.collect()

    print(f"Train intervals: {len(train_times)}  |  Test intervals: {len(test_times)}")
    print(
        f"Aggregated train shape: {x_train_agg.shape}  |  test shape: {x_test_agg.shape}"
    )
    print(f"Candidate features: {available}\n")

    return DatasetSplit(
        x_train_agg=x_train_agg,
        x_test_agg=x_test_agg,
        interval_energy_train=interval_energy_train,
        interval_energy_test=interval_energy_test,
        available_features=available,
        z_train_agg=z_train_agg,
        z_test_agg=z_test_agg,
        static_feature_names=static_feature_names,
    )


def train_and_evaluate(
    features: list[str],
    split: DatasetSplit,
    l1_penalty: float,
    static_penalty: float = 0.0,
) -> tuple[
    float | None, float | None, dict[str, float] | None, dict[str, float] | None
]:
    """Fit the energy model and evaluate on the test set.

    Returns (r2, mae, dynamic_weights, static_weights).
    static_weights is None in scalar mode and a dict in weighted mode.
    """
    x_tr = split.x_train_agg[features].values
    y_tr = split.interval_energy_train.values

    x_scaler = MaxAbsScaler()
    x_matrix = x_scaler.fit_transform(x_tr)

    weights_var = cp.Variable(x_matrix.shape[1])

    if split.z_train_agg is not None:
        # Weighted static model: E = X @ w + Z @ v
        z_scaler = MaxAbsScaler()
        z_matrix = z_scaler.fit_transform(split.z_train_agg.values)
        v_var = cp.Variable(z_matrix.shape[1])
        interval_preds = x_matrix @ weights_var + z_matrix @ v_var
        reg = l1_penalty * cp.norm1(weights_var) + static_penalty * cp.norm1(v_var)
        constraints = [weights_var >= 0, v_var >= 0]
    else:
        # Scalar static model: E = X @ w + static_scalar
        static_var = cp.Variable()
        interval_preds = x_matrix @ weights_var + static_var
        reg = l1_penalty * cp.norm1(weights_var)
        constraints = [weights_var >= 0, static_var >= 0]

    loss = cp.sum_squares(interval_preds - y_tr)
    problem = cp.Problem(cp.Minimize(loss + reg), constraints=constraints)
    problem.solve(solver=cp.CLARABEL)

    if weights_var.value is None:
        if split.z_train_agg is not None:
            del problem, weights_var, v_var
        else:
            del problem, weights_var, static_var
        return None, None, None, None

    dyn_weights = weights_var.value.copy()

    if split.z_train_agg is not None:
        stat_weights = v_var.value.copy()
        del problem, weights_var, v_var
        x_te = x_scaler.transform(split.x_test_agg[features].values)
        z_te = z_scaler.transform(split.z_test_agg.values)
        pred = x_te @ dyn_weights + z_te @ stat_weights
    else:
        static_energy = float(static_var.value)
        del problem, weights_var, static_var
        x_te = x_scaler.transform(split.x_test_agg[features].values)
        pred = x_te @ dyn_weights + static_energy
        stat_weights = None

    actual = split.interval_energy_test.values
    r2 = r2_score(actual, pred)
    mae = mean_absolute_error(actual, pred)

    dyn_dict = dict(zip(features, dyn_weights))
    stat_dict = (
        dict(zip(split.static_feature_names, stat_weights))
        if stat_weights is not None
        else None
    )
    return r2, mae, dyn_dict, stat_dict


def _bootstrap_train(
    split: DatasetSplit,
    features: list[str],
    penalty: float,
    static_penalty: float,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """Train on a random subsample; return (dynamic_weights, static_weights).

    Samples at interval level from pre-aggregated matrices.
    static_weights is None in scalar mode.
    """
    n_intervals = len(split.interval_energy_train)
    n_sample = max(1, int(n_intervals * fraction))
    idx = rng.choice(n_intervals, size=n_sample, replace=False)

    x_sub = split.x_train_agg[features].iloc[idx].values
    y_sub = split.interval_energy_train.iloc[idx].values

    x_scaler = MaxAbsScaler()
    x_matrix = x_scaler.fit_transform(x_sub)

    weights_var = cp.Variable(x_matrix.shape[1])
    zeros_dyn = {f: 0.0 for f in features}

    if split.z_train_agg is not None:
        z_sub = split.z_train_agg.iloc[idx].values
        z_scaler = MaxAbsScaler()
        z_matrix = z_scaler.fit_transform(z_sub)
        v_var = cp.Variable(z_matrix.shape[1])
        interval_preds = x_matrix @ weights_var + z_matrix @ v_var
        reg = penalty * cp.norm1(weights_var) + static_penalty * cp.norm1(v_var)
        constraints = [weights_var >= 0, v_var >= 0]
    else:
        static_var = cp.Variable()
        interval_preds = x_matrix @ weights_var + static_var
        reg = penalty * cp.norm1(weights_var)
        constraints = [weights_var >= 0, static_var >= 0]

    loss = cp.sum_squares(interval_preds - y_sub)
    problem = cp.Problem(cp.Minimize(loss + reg), constraints=constraints)
    problem.solve(solver=cp.CLARABEL)

    if weights_var.value is None:
        if split.z_train_agg is not None:
            del problem, weights_var, v_var
            return zeros_dyn, {f: 0.0 for f in split.static_feature_names}
        else:
            del problem, weights_var, static_var
            return zeros_dyn, None

    dyn = dict(zip(features, weights_var.value.copy()))

    if split.z_train_agg is not None:
        stat = dict(zip(split.static_feature_names, v_var.value.copy()))
        del problem, weights_var, v_var
        return dyn, stat
    else:
        del problem, weights_var, static_var
        return dyn, None


def run_path_sweep(
    args: argparse.Namespace,
    split: DatasetSplit,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Sweep L1 penalties; fit model and run stability selection at each.

    Returns
    -------
    path_df
        Columns: penalty, r2, mae_pct, <dynamic_feature>...
    stability_df
        Columns: penalty, <dynamic_feature>... (selection frequencies)
    static_path_df
        Columns: penalty, <static_feature>... — None in scalar mode.
    static_stability_df
        Columns: penalty, <static_feature>... — None in scalar mode.
    """
    features = split.available_features
    static_features = split.static_feature_names  # None in scalar mode
    penalties = np.logspace(
        np.log10(args.penalty_min),
        np.log10(args.penalty_max),
        num=args.penalties,
    )
    n_penalties = len(penalties)
    mean_energy = split.interval_energy_test.mean()

    path_records: list[dict] = []
    stability_records: list[dict] = []
    static_path_records: list[dict] = []
    static_stability_records: list[dict] = []

    weighted = split.z_train_agg is not None
    print("=== Regularization Path + Stability Selection ===")
    if weighted:
        print(f"    Static model: weighted  (Z features: {static_features})")
        print(f"    static_penalty = {args.static_penalty_ratio} × l1_penalty\n")
    else:
        print("    Static model: scalar\n")

    for i, penalty in enumerate(penalties):
        static_penalty = args.static_penalty_ratio * penalty if weighted else 0.0

        r2, mae, dyn_weights, stat_weights = train_and_evaluate(
            features, split, penalty, static_penalty
        )

        if r2 is None:
            r2_val, mae_pct_val = np.nan, np.nan
            dyn_vals = {f: 0.0 for f in features}
            stat_vals = {f: 0.0 for f in (static_features or [])}
        else:
            r2_val = r2
            mae_pct_val = 100.0 * mae / mean_energy
            dyn_vals = {f: dyn_weights.get(f, 0.0) for f in features}
            stat_vals = (
                {f: stat_weights.get(f, 0.0) for f in static_features}
                if stat_weights is not None
                else {}
            )

        n_active_dyn = sum(
            1 for f in features if dyn_vals.get(f, 0.0) > args.zero_threshold
        )
        n_active_stat = (
            sum(
                1
                for f in (static_features or [])
                if stat_vals.get(f, 0.0) > args.zero_threshold
            )
            if weighted
            else 0
        )

        suffix = f"  static_active={n_active_stat}" if weighted else ""
        print(
            f"Penalty {i + 1}/{n_penalties}: "
            f"λ={penalty:.4f}  R²={r2_val:.4f}  dyn_active={n_active_dyn}{suffix}"
            f"  [bootstrapping...]"
        )

        path_records.append(
            {"penalty": penalty, "r2": r2_val, "mae_pct": mae_pct_val, **dyn_vals}
        )
        if weighted:
            static_path_records.append({"penalty": penalty, **stat_vals})

        # stability selection
        dyn_counts: dict[str, int] = {f: 0 for f in features}
        stat_counts: dict[str, int] = {f: 0 for f in (static_features or [])}

        for j in range(args.n_bootstrap):
            rng = np.random.default_rng(args.random_state + j)
            boot_dyn, boot_stat = _bootstrap_train(
                split, features, penalty, static_penalty, args.bootstrap_fraction, rng
            )
            for f in features:
                if boot_dyn.get(f, 0.0) > args.zero_threshold:
                    dyn_counts[f] += 1
            if weighted and boot_stat is not None:
                for f in static_features:
                    if boot_stat.get(f, 0.0) > args.zero_threshold:
                        stat_counts[f] += 1

        stability_records.append(
            {
                "penalty": penalty,
                **{f: dyn_counts[f] / args.n_bootstrap for f in features},
            }
        )
        if weighted:
            static_stability_records.append(
                {
                    "penalty": penalty,
                    **{f: stat_counts[f] / args.n_bootstrap for f in static_features},
                }
            )

        gc.collect()

    path_df = pd.DataFrame(path_records)
    stability_df = pd.DataFrame(stability_records)
    static_path_df = pd.DataFrame(static_path_records) if weighted else None
    static_stability_df = pd.DataFrame(static_stability_records) if weighted else None
    return path_df, stability_df, static_path_df, static_stability_df


def _find_elbow_penalty(path_df: pd.DataFrame, threshold: float = 0.95) -> float:
    """Return the largest penalty where R² >= threshold * max(R²)."""
    r2_series = path_df["r2"].dropna()
    if r2_series.empty:
        return path_df["penalty"].iloc[0]
    max_r2 = r2_series.max()
    cutoff = threshold * max_r2
    eligible = path_df.loc[r2_series.index][
        path_df.loc[r2_series.index, "r2"] >= cutoff
    ]
    if eligible.empty:
        return path_df["penalty"].iloc[0]
    return eligible["penalty"].iloc[-1]


def plot_regularization_path(
    path_df: pd.DataFrame,
    features: list[str],
    dataset_name: str,
    args: argparse.Namespace,
) -> None:
    """Plot weight of each feature vs log10(λ), with R² on a twin y-axis."""
    log_penalties = np.log10(path_df["penalty"].values)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(k / max(len(features) - 1, 1)) for k in range(len(features))]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    for k, feature in enumerate(features):
        ax1.plot(
            log_penalties,
            path_df[feature].values,
            color=colors[k],
            linewidth=1.5,
            label=feature,
        )

    ax2.plot(
        log_penalties,
        path_df["r2"].values,
        color="black",
        linewidth=2,
        linestyle="--",
        label="R²",
        alpha=0.7,
    )

    ax1.set_xlabel("log₁₀(λ)")
    # Weights are in the MaxAbsScaler-transformed feature space, not raw units.
    ax1.set_ylabel("Weight (scaled feature space)")
    ax2.set_ylabel("R²")
    ax1.set_title(f"Regularization Path — {dataset_name}")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
        fontsize=7,
        ncol=2,
        framealpha=0.8,
    )

    fig.tight_layout()
    out_path = os.path.join(args.output_dir, f"lasso_path_{dataset_name}.png")
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved: {out_path}")
    if args.show_plots:
        plt.show()
    plt.close(fig)


def plot_static_path(
    static_path_df: pd.DataFrame,
    static_features: list[str],
    dataset_name: str,
    args: argparse.Namespace,
) -> None:
    """Plot static feature weights vs log10(λ) for the weighted static model."""
    log_penalties = np.log10(static_path_df["penalty"].values)
    cmap = plt.get_cmap("tab10")
    colors = [
        cmap(k / max(len(static_features) - 1, 1)) for k in range(len(static_features))
    ]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    for k, feat in enumerate(static_features):
        ax1.plot(
            log_penalties,
            static_path_df[feat].values,
            color=colors[k],
            linewidth=2.0,
            label=feat,
        )

    ax1.set_xlabel("log₁₀(λ)")
    ax1.set_ylabel("Static weight (scaled feature space)")
    ax1.set_title(f"Static Component Regularization Path — {dataset_name}")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.8)

    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax1.spines["top"].set_visible(False)
    fig.tight_layout()

    out_path = os.path.join(args.output_dir, f"lasso_static_path_{dataset_name}.png")
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved: {out_path}")
    if args.show_plots:
        plt.show()
    plt.close(fig)


def plot_metrics_curve(
    path_df: pd.DataFrame,
    dataset_name: str,
    args: argparse.Namespace,
) -> None:
    """Plot R² and MAE% vs log10(λ); mark the elbow penalty."""
    log_penalties = np.log10(path_df["penalty"].values)
    elbow_penalty = _find_elbow_penalty(path_df)
    log_elbow = np.log10(elbow_penalty)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(
        log_penalties, path_df["r2"].values, color="steelblue", linewidth=2, label="R²"
    )
    ax2.plot(
        log_penalties,
        path_df["mae_pct"].values,
        color="tomato",
        linewidth=2,
        linestyle="--",
        label="MAE%",
    )

    ax1.axvline(
        log_elbow,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=f"Elbow λ={elbow_penalty:.4f}",
    )

    ax1.set_xlabel("log₁₀(λ)")
    ax1.set_ylabel("R²", color="steelblue")
    ax2.set_ylabel("MAE%", color="tomato")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax1.set_title(f"R² / MAE% vs Penalty — {dataset_name}")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(args.output_dir, f"lasso_path_metrics_{dataset_name}.png")
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved: {out_path}")
    if args.show_plots:
        plt.show()
    plt.close(fig)


def plot_stability_heatmap(
    stability_df: pd.DataFrame,
    features: list[str],
    dataset_name: str,
    args: argparse.Namespace,
) -> None:
    """Heatmap of selection frequency (features × penalties) from stability selection."""
    # Build matrix: rows = features (sorted by mean freq, highest at top),
    # columns = penalty grid points.
    freq_matrix = stability_df[features].T  # shape: (n_features, n_penalties)
    mean_freq = freq_matrix.mean(axis=1)
    sorted_features = mean_freq.sort_values(ascending=False).index.tolist()
    freq_matrix = freq_matrix.loc[sorted_features]

    # x-tick labels: log10(penalty), shown every stride ticks to avoid crowding.
    n_penalties = len(stability_df)
    stride = max(1, n_penalties // 10)
    tick_positions = list(range(0, n_penalties, stride))
    tick_labels = [
        f"{np.log10(stability_df['penalty'].iloc[p]):.2f}" for p in tick_positions
    ]

    fig_height = max(5, len(sorted_features) * 0.4 + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    sns.heatmap(
        freq_matrix,
        ax=ax,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.0,
        cbar_kws={"label": "Selection frequency"},
        annot=False,
    )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("log₁₀(λ)")
    ax.set_ylabel("Feature")
    ax.set_title(f"Stability Selection Heatmap — {dataset_name}")

    fig.tight_layout()
    out_path = os.path.join(args.output_dir, f"lasso_stability_{dataset_name}.png")
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved: {out_path}")
    if args.show_plots:
        plt.show()
    plt.close(fig)


def save_csvs(
    path_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    dataset_name: str,
    args: argparse.Namespace,
    static_path_df: pd.DataFrame | None = None,
    static_stability_df: pd.DataFrame | None = None,
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    path_df.to_csv(
        os.path.join(args.output_dir, f"lasso_path_{dataset_name}.csv"), index=False
    )
    stability_df.to_csv(
        os.path.join(args.output_dir, f"lasso_stability_{dataset_name}.csv"),
        index=False,
    )
    if static_path_df is not None:
        static_path_df.to_csv(
            os.path.join(args.output_dir, f"lasso_static_path_{dataset_name}.csv"),
            index=False,
        )
    if static_stability_df is not None:
        static_stability_df.to_csv(
            os.path.join(args.output_dir, f"lasso_static_stability_{dataset_name}.csv"),
            index=False,
        )
    print(f"Saved CSVs to {args.output_dir}/")


def print_summary(
    path_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    split: DatasetSplit,
    args: argparse.Namespace,
    static_path_df: pd.DataFrame | None = None,
    static_stability_df: pd.DataFrame | None = None,
) -> None:
    features = split.available_features
    n_penalties = len(path_df)

    print("\n=== Summary ===\n")

    # --- Representative penalties table ---
    representative_indices = list(range(0, n_penalties, max(1, n_penalties // 10)))
    print(f"{'#':<5} {'λ':>10} {'R²':>8} {'MAE%':>8}  Active features")
    print("-" * 80)
    for idx in representative_indices:
        row = path_df.iloc[idx]
        active = [f for f in features if row.get(f, 0.0) > args.zero_threshold]
        r2_str = f"{row['r2']:.4f}" if not np.isnan(row["r2"]) else "  nan "
        mae_str = f"{row['mae_pct']:.2f}%" if not np.isnan(row["mae_pct"]) else "  nan "
        print(f"{idx:<5} {row['penalty']:>10.4f} {r2_str:>8} {mae_str:>8}  {active}")

    # --- Elbow penalty ---
    elbow_penalty = _find_elbow_penalty(path_df)
    elbow_row = path_df.iloc[(path_df["penalty"] - elbow_penalty).abs().argmin()]
    elbow_active = [f for f in features if elbow_row.get(f, 0.0) > args.zero_threshold]
    print(f"\nElbow penalty (last λ where R² ≥ 95% of max R²): λ={elbow_penalty:.4f}")
    print(f"  R²={elbow_row['r2']:.4f}  MAE%={elbow_row['mae_pct']:.2f}%")
    print(f"  Active features: {elbow_active}")

    # --- Stability ranking ---
    mean_freq = stability_df[features].mean(axis=0).sort_values(ascending=False)
    print("\nStability ranking (mean selection frequency across all penalties):")
    print(f"  {'Rank':<5} {'Feature':<35} {'Mean freq':>10}")
    print("  " + "-" * 55)
    for rank, (feature, freq) in enumerate(mean_freq.items(), start=1):
        print(f"  {rank:<5} {feature:<35} {freq:>10.3f}")

    if (
        static_stability_df is not None
        and split is not None
        and split.static_feature_names
    ):
        mean_static_freq = static_stability_df.drop(columns="penalty").mean()
        ranked_static = mean_static_freq.sort_values(ascending=False)
        print("\nStatic feature stability ranking (mean selection frequency):")
        print(f"  {'Rank':<6}{'Feature':<30}{'Mean freq':>10}")
        print("  " + "-" * 46)
        for rank, (feat, freq) in enumerate(ranked_static.items(), start=1):
            print(f"  {rank:<6}{feat:<30}{freq:>10.3f}")


def main() -> None:
    args = parse_args()
    split = load_and_split_data(args)
    path_df, stability_df, static_path_df, static_stability_df = run_path_sweep(
        args, split
    )

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = Path(args.data).parent.name or Path(args.data).stem

    plot_regularization_path(path_df, split.available_features, dataset_name, args)
    plot_metrics_curve(path_df, dataset_name, args)
    plot_stability_heatmap(stability_df, split.available_features, dataset_name, args)

    if static_path_df is not None:
        plot_static_path(static_path_df, split.static_feature_names, dataset_name, args)

    save_csvs(
        path_df, stability_df, dataset_name, args, static_path_df, static_stability_df
    )
    print_summary(
        path_df, stability_df, split, args, static_path_df, static_stability_df
    )
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
