"""Sequential Backward Selection (SBS) for energy model feature selection.

SBS starts with the full feature set and greedily removes one feature per step,
keeping the removal that costs the least in predictive power (highest R² after
removal). It stops when any further removal would drop R² by more than
``--max-loss``, or when ``--min-features`` is reached.

Why SBS catches things SFS misses
----------------------------------
SFS builds a model greedily from nothing, so it can miss features that only
contribute when *combined* with others — if feature A looks weak in isolation
it is never added, and feature B that depends on A is evaluated without it.
SBS starts from the full joint model, so synergistic interactions between
features are visible from the very first step. A feature pair that is jointly
informative but individually weak will be retained together by SBS, whereas
SFS might discard both.

Typical use: run SFS first to find a compact forward-selected set, then run SBS
on the full candidate set to check whether any features excluded by SFS would
have been retained in a backward pass (indicating interaction effects).
"""

import argparse
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
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
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    interval_energy_train: pd.Series
    interval_energy_test: pd.Series
    available_features: list[str]


@dataclass
class StepResult:
    step: int
    removed: str
    features: list[str]
    r2: float
    mae_pct: float
    weights: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential backward selection for energy model"
    )
    parser.add_argument(
        "--data", default="runs/benchmark-siena06-v6/process_interval_data.parquet"
    )
    parser.add_argument("--l1-penalty", type=float, default=0.1)
    parser.add_argument(
        "--max-loss",
        type=float,
        default=0.01,
        help="Maximum R² drop allowed when removing a feature",
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=1,
        help="Stop when the selected set reaches this size",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def load_and_split_data(args: argparse.Namespace) -> DatasetSplit:
    print(f"Loading {args.data} ...")
    df = pd.read_parquet(args.data)
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

    interval_energy_train = interval_energy_all.loc[train_times]
    interval_energy_test = interval_energy_all.loc[test_times]

    df_train = df[df["_time"].isin(train_times)].copy()
    df_test = df[df["_time"].isin(test_times)].copy()

    print(f"Train intervals: {len(train_times)}  |  Test intervals: {len(test_times)}")
    print(f"Candidate features: {available}\n")

    return DatasetSplit(
        df_train=df_train,
        df_test=df_test,
        interval_energy_train=interval_energy_train,
        interval_energy_test=interval_energy_test,
        available_features=available,
    )


def train_and_evaluate(
    features: list[str],
    split: DatasetSplit,
    l1_penalty: float,
) -> tuple[float | None, float | None, dict[str, float] | None]:
    df_tr = split.df_train.copy()
    df_tr[features] = df_tr[features].fillna(0)
    df_tr = df_tr[df_tr["_time"].isin(split.interval_energy_train.index)].sort_values(
        "_time"
    )
    ie_tr = split.interval_energy_train.sort_index()

    df_agg = df_tr.groupby("_time")[features].sum()
    df_agg = df_agg.reindex(ie_tr.index).fillna(0)

    scaler = MaxAbsScaler()
    x_matrix = scaler.fit_transform(df_agg.values)

    weights_var = cp.Variable(x_matrix.shape[1])
    static_var = cp.Variable()

    interval_preds = x_matrix @ weights_var + static_var
    loss = cp.sum_squares(interval_preds - ie_tr.values)
    reg = l1_penalty * cp.norm1(weights_var)
    problem = cp.Problem(
        cp.Minimize(loss + reg), constraints=[static_var >= 0, weights_var >= 0]
    )
    problem.solve(solver=cp.CLARABEL)

    if weights_var.value is None:
        return None, None, None

    weights = weights_var.value
    static_energy = static_var.value

    df_te = split.df_test.copy()
    df_te[features] = df_te[features].fillna(0)
    df_te[features] = scaler.transform(df_te[features])
    df_te["pred"] = df_te[features].values @ weights

    pred = df_te.groupby("_time")["pred"].sum() + static_energy
    actual = split.interval_energy_test

    aligned = pred.reindex(actual.index).dropna()
    actual_aligned = actual.loc[aligned.index]

    r2 = r2_score(actual_aligned, aligned)
    mae = mean_absolute_error(actual_aligned, aligned)
    return r2, mae, dict(zip(features, weights))


def run_sbs(args: argparse.Namespace, split: DatasetSplit) -> list[StepResult]:
    selected = list(split.available_features)
    results_log: list[StepResult] = []

    print("=== Sequential Backward Selection ===\n")

    # Evaluate baseline (all features)
    print(f"Evaluating baseline with all {len(selected)} features ...")
    baseline_r2, baseline_mae, baseline_weights = train_and_evaluate(
        features=selected,
        split=split,
        l1_penalty=args.l1_penalty,
    )

    if baseline_r2 is None:
        print("Baseline model failed to converge. Aborting.")
        return results_log

    mean_energy = split.interval_energy_test.mean()
    print(
        f"Baseline  R²={baseline_r2:.4f}  MAE%={100 * baseline_mae / mean_energy:.2f}%"
    )
    print(f"  Features: {selected}\n")

    best_r2 = baseline_r2

    while len(selected) > max(1, args.min_features):
        step_best_r2 = -np.inf
        step_best_feature = None
        step_best_mae = None
        step_best_weights = None

        for feature in selected:
            candidate_set = [f for f in selected if f != feature]
            r2, mae, weights = train_and_evaluate(
                features=candidate_set,
                split=split,
                l1_penalty=args.l1_penalty,
            )
            if r2 is None:
                continue

            print(
                f"  [remove {feature}]  R²={r2:.4f}  MAE%={100 * mae / mean_energy:.2f}%"
            )

            if r2 > step_best_r2:
                step_best_r2 = r2
                step_best_feature = feature
                step_best_mae = mae
                step_best_weights = weights

        loss = best_r2 - step_best_r2
        if step_best_feature is None or loss > args.max_loss:
            print(
                f"\nStopping: best R² after removal {step_best_r2:.4f} "
                f"(loss {loss:.4f} > max_loss {args.max_loss})"
            )
            break

        selected = [f for f in selected if f != step_best_feature]
        best_r2 = step_best_r2

        results_log.append(
            StepResult(
                step=len(results_log) + 1,
                removed=step_best_feature,
                features=list(selected),
                r2=step_best_r2,
                mae_pct=100 * step_best_mae / mean_energy,
                weights=step_best_weights,
            )
        )

        print(
            f"\n→ Step {len(results_log)}: removed '{step_best_feature}'  "
            f"R²={step_best_r2:.4f}  MAE%={100 * step_best_mae / mean_energy:.2f}%"
        )
        print(f"  Remaining ({len(selected)}): {selected}\n")

    return results_log


def print_summary(results_log: list[StepResult]) -> None:
    print("\n=== Summary ===")
    print(
        f"{'Step':<6} {'Removed Feature':<35} {'R²':>8} {'MAE%':>8}  {'Remaining':>9}"
    )
    print("-" * 80)

    for result in results_log:
        print(
            f"{result.step:<6} {result.removed:<35} {result.r2:>8.4f} "
            f"{result.mae_pct:>7.2f}%  {len(result.features):>9}"
        )

    if results_log:
        best = max(results_log, key=lambda item: item.r2)
        print(
            f"\nRecommended stopping point (R²={best.r2:.4f}, MAE%={best.mae_pct:.2f}%, "
            f"{len(best.features)} features):"
        )
        print(f"  {best.features}")
        print("\nWeights:")
        for feature, weight in best.weights.items():
            print(f"  {feature}: {weight:.4e}")


def main() -> None:
    args = parse_args()
    split = load_and_split_data(args)
    results_log = run_sbs(args, split)
    print_summary(results_log)


if __name__ == "__main__":
    main()
