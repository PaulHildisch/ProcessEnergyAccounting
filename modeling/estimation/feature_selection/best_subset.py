"""Exhaustive best-subset selection for the energy accounting model.

Exhaustive best-subset search is the gold standard for small feature sets: unlike
greedy approaches (forward/backward SFS) it incurs no approximation error and
returns the provably optimal subset at every size k.
For each size k the script fits a non-negative L1-regularised linear model via
CVXPY (identical to sfs.py) on a time-based train/test split, records R² and
MAE% for every combination, and reports the top-n subsets per size plus the
single overall best.  A full results CSV is written to the output directory so
results can be inspected offline.
"""

import argparse
import itertools
import os
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

CANDIDATE_FEATURES = [
    "delta_cpu_ns",
    "delta_io_bytes",
    "delta_net_send_bytes",
    "context_switches",
    "syscall_count",
    "delta_rss_memory",
    "delta_cpu_time_psutil",
    "delta_cpu_time_proc",
    "delta_instructions",
    "delta_cycles",
    "delta_branch_instructions",
    "delta_cache_misses",
    "delta_stalled_cycles_backend",
    "delta_llc_load_misses",
    "delta_llc_store_misses",
    "delta_cpu_migrations",
    "delta_page_faults_min",
    "delta_page_faults_maj",
    "delta_stalled_cycles_frontend",
    "delta_branch_misses",
    "delta_ref_cpu_cycles",
    "delta_l1d_load_misses",
    "delta_dtlb_load_misses",
    "delta_dtlb_store_misses",
    "delta_node_load_misses",
    "delta_disk_read_bytes",
    "delta_disk_write_bytes",
    "delta_net_recv_bytes",
    "delta_net_send_packets",
    "delta_net_recv_packets",
    "hw_numa_node_count",
    "hw_freq_ratio",
    "hw_core_count",
    "hw_ram_total_gb",
    "hw_ram_slot_count",
    "hw_fan_count",
    "hw_temperature_c",
    "hw_arch_x86_64",
    "hw_arch_arm64",
    "hw_arch_riscv64",
    "hw_arch_other",
    "hw_cpu_vendor_intel",
    "hw_cpu_vendor_amd",
    "hw_cpu_vendor_arm",
    "hw_cpu_vendor_apple",
    "hw_cpu_vendor_other",
    "hw_tdp_tier_low",
    "hw_tdp_tier_mid",
    "hw_tdp_tier_high",
    "hw_tdp_tier_unknown",
    "hw_cpu_governor_performance",
    "hw_cpu_governor_powersave",
    "hw_cpu_governor_schedutil",
    "hw_cpu_governor_ondemand",
    "hw_cpu_governor_unknown",
    "hw_cores_1_4",
    "hw_cores_5_8",
    "hw_cores_9_16",
    "hw_cores_17_32",
    "hw_cores_33_plus",
    "hw_cores_unknown",
    "hw_ram_lt16gb",
    "hw_ram_16_32gb",
    "hw_ram_33_64gb",
    "hw_ram_65_128gb",
    "hw_ram_129gb_plus",
    "hw_ram_unknown",
    "hw_ram_slots_single",
    "hw_ram_slots_dual",
    "hw_ram_slots_quad_or_more",
    "hw_ram_slots_unknown",
    "hw_fans_0",
    "hw_fans_1",
    "hw_fans_2_plus",
    "hw_fans_unknown",
    "hw_temp_cool",
    "hw_temp_normal",
    "hw_temp_hot",
    "hw_temp_unknown",
    "delta_fp_scalar",
    "delta_fp_128b_packed",
    "delta_fp_256b_packed",
    "delta_fp_512b_packed",
    "delta_fp_add_sub",
    "delta_fp_mult",
    "delta_fp_div",
    "delta_fp_mac",
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
class SubsetResult:
    size: int
    features: list[str]
    r2: float
    mae_pct: float
    weights: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustive best-subset selection for energy model"
    )
    parser.add_argument(
        "--data", default="runs/benchmark-siena06-v6/process_interval_data.parquet"
    )
    parser.add_argument("--l1-penalty", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--max-k", type=int, default=6, help="Maximum subset size to search"
    )
    parser.add_argument(
        "--min-size", type=int, default=1, help="Minimum subset size to search"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="How many top subsets per size to print in summary",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save results CSV",
    )
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


def run_best_subset(
    args: argparse.Namespace, split: DatasetSplit
) -> list[SubsetResult]:
    mean_energy = split.interval_energy_test.mean()
    max_k = min(args.max_k, len(split.available_features))
    all_results: list[SubsetResult] = []

    print("=== Exhaustive Best-Subset Selection ===\n")

    for k in range(args.min_size, max_k + 1):
        combos = list(itertools.combinations(split.available_features, k))
        print(f"\nSearching subsets of size {k} ({len(combos)} combinations)...")

        size_best_r2 = -np.inf
        size_results: list[SubsetResult] = []

        for features in combos:
            feature_list = list(features)
            r2, mae, weights = train_and_evaluate(
                features=feature_list,
                split=split,
                l1_penalty=args.l1_penalty,
            )
            if r2 is None:
                continue

            mae_pct = 100 * mae / mean_energy
            size_results.append(
                SubsetResult(
                    size=k,
                    features=feature_list,
                    r2=r2,
                    mae_pct=mae_pct,
                    weights=weights,
                )
            )

            if r2 > size_best_r2:
                size_best_r2 = r2
                print(f"  New best size-{k}: {feature_list}  R²={r2:.4f}")

        if not size_results:
            print(f"  No valid results for size {k}.")
            continue

        size_results.sort(key=lambda item: item.r2, reverse=True)
        all_results.extend(size_results)

        best = size_results[0]
        print(
            f"→ Best size-{k}: {best.features}  R²={best.r2:.4f}  MAE%={best.mae_pct:.2f}%"
        )

    return all_results


def save_results_csv(
    all_results: list[SubsetResult], data_path: str, output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    output_path = os.path.join(output_dir, f"best_subset_results_{dataset_name}.csv")

    # Rank within each size (1-based, sorted by descending R²)
    rows = []
    sizes_seen: dict[int, int] = {}
    for result in sorted(all_results, key=lambda item: (item.size, -item.r2)):
        size = result.size
        sizes_seen[size] = sizes_seen.get(size, 0) + 1
        rows.append(
            {
                "size": size,
                "rank_within_size": sizes_seen[size],
                "features": ",".join(result.features),
                "r2": result.r2,
                "mae_pct": result.mae_pct,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


def print_summary(all_results: list[SubsetResult], top_n: int) -> None:
    if not all_results:
        print("\nNo results to summarise.")
        return

    # Group by size
    by_size: dict[int, list[SubsetResult]] = {}
    for result in all_results:
        by_size.setdefault(result.size, []).append(result)

    print("\n=== Summary ===")
    print(f"{'Size':<6} {'R²':>8} {'MAE%':>8}  Best features")
    print("-" * 90)

    for k in sorted(by_size):
        size_results = sorted(by_size[k], key=lambda item: item.r2, reverse=True)
        best = size_results[0]
        print(f"{k:<6} {best.r2:>8.4f} {best.mae_pct:>7.2f}%  {best.features}")

    print(f"\n--- Top-{top_n} subsets per size ---")
    for k in sorted(by_size):
        size_results = sorted(by_size[k], key=lambda item: item.r2, reverse=True)
        print(f"\nSize {k}:")
        for rank, result in enumerate(size_results[:top_n], start=1):
            print(
                f"  #{rank}  R²={result.r2:.4f}  MAE%={result.mae_pct:.2f}%  {result.features}"
            )

    overall_best = max(all_results, key=lambda item: item.r2)
    print(
        f"\n=== Overall best  (size={overall_best.size}, R²={overall_best.r2:.4f},"
        f" MAE%={overall_best.mae_pct:.2f}%) ==="
    )
    print(f"  Features: {overall_best.features}")
    print("\n  Weights:")
    for feature, weight in overall_best.weights.items():
        print(f"    {feature}: {weight:.4e}")


def main() -> None:
    args = parse_args()
    split = load_and_split_data(args)
    all_results = run_best_subset(args, split)
    print_summary(all_results, top_n=args.top_n)
    save_results_csv(all_results, data_path=args.data, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
