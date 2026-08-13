#!/usr/bin/env python3
"""
Mutual Information analysis for energy modeling.

Computes mutual information between features and the target to detect
non-linear dependencies that Pearson/Spearman correlation can miss.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

DEFAULT_FEATURES = [
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mutual information analysis for energy modeling features"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the parquet data file",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="interval_energy",
        help="Target variable name (default: interval_energy)",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=None,
        help="List of features to analyze (default: use predefined list)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save plots (default: plots)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved plots (default: 300)",
    )
    parser.add_argument(
        "--filter-active",
        action="store_true",
        help="Filter to only intervals with target > 0",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate process-level data by time interval",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=3,
        help="Number of neighbors for MI estimation (default: 3)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--prune-redundant",
        action="store_true",
        help=(
            "Greedily drop features that are highly correlated with a "
            "higher-MI feature. Addresses MI's univariate blind spot: two "
            "features can each score highly while carrying the same "
            "information (e.g. delta_cycles vs delta_cpu_ns)."
        ),
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=0.9,
        help=(
            "Absolute correlation above which a lower-MI feature is treated "
            "as redundant and dropped (default: 0.9)"
        ),
    )
    parser.add_argument(
        "--redundancy-method",
        type=str,
        default="spearman",
        choices=["pearson", "spearman"],
        help="Correlation method used for redundancy pruning (default: spearman)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively (in addition to saving)",
    )
    return parser.parse_args()


def load_and_prepare_data(
    data_path: str,
    features: list[str],
    target: str,
    aggregate: bool = True,
) -> pd.DataFrame:
    """
    Load data and prepare it for mutual information analysis.

    Args:
        data_path: Path to parquet file
        features: List of feature column names
        target: Target variable name
        aggregate: If True, aggregate by _time interval

    Returns:
        DataFrame ready for analysis
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    if "_time" in df.columns:
        df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")

    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]

    if missing_features:
        print(f"Warning: Features not found in data: {missing_features}")

    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in data")

    print(f"Available features: {len(available_features)}")
    print(f"Data shape: {df.shape}")

    if aggregate and "_time" in df.columns:
        print("Aggregating data by time interval...")
        df_energy = df[["_time", target]].dropna().drop_duplicates("_time")
        df_agg = df.groupby("_time")[available_features].sum().reset_index()
        df_result = df_agg.merge(df_energy, on="_time", how="left")
        print(f"Aggregated shape: {df_result.shape}")
    else:
        df_result = df[available_features + [target]].copy()

    df_result[available_features] = df_result[available_features].fillna(0)
    df_result = df_result.dropna(subset=[target])

    print(f"Final data shape: {df_result.shape}")
    return df_result


def compute_mutual_information(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> pd.Series:
    """
    Compute mutual information between each feature and the target.

    Args:
        df: DataFrame with features and target
        features: List of feature names
        target: Target variable name
        n_neighbors: Number of neighbors for the KNN-based estimator
        random_state: Seed for reproducibility

    Returns:
        Series of mutual information scores indexed by feature
    """
    available_features = [f for f in features if f in df.columns]

    print(f"\nComputing mutual information (n_neighbors={n_neighbors})...")

    feature_matrix = df[available_features].to_numpy()
    target_values = df[target].to_numpy()

    mi_scores = mutual_info_regression(
        feature_matrix,
        target_values,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    mi_series = pd.Series(mi_scores, index=available_features)

    print("\n=== Mutual Information Scores ===")
    print(mi_series.sort_values(ascending=False))

    return mi_series


def select_non_redundant(
    mi_scores: pd.Series,
    df: pd.DataFrame,
    threshold: float = 0.9,
    method: str = "spearman",
) -> tuple[list[str], pd.DataFrame]:
    """
    Greedily select a minimal-redundancy subset of features.

    Mutual information is computed per feature in isolation, so two features
    can each score highly while encoding nearly the same signal (e.g.
    ``delta_cycles`` and ``delta_cpu_ns``). This walks features in descending
    MI order and keeps one only if its absolute correlation with every
    already-kept feature stays below ``threshold``; otherwise it is dropped as
    redundant with the higher-MI feature it duplicates.

    Args:
        mi_scores: Mutual information scores indexed by feature
        df: DataFrame containing the feature columns
        threshold: |correlation| at or above which a feature is redundant
        method: Correlation method ("pearson" or "spearman")

    Returns:
        Tuple of (selected feature names, table of dropped features)
    """
    ranked = mi_scores.sort_values(ascending=False)
    features = ranked.index.to_list()
    corr = df[features].corr(method=method).abs()

    selected: list[str] = []
    dropped: list[dict] = []

    for feat in features:
        # Find the already-kept feature this one correlates with most.
        best_kept = None
        best_corr = 0.0
        for kept in selected:
            c = float(corr.loc[feat, kept])
            if c > best_corr:
                best_corr = c
                best_kept = kept

        if best_kept is not None and best_corr >= threshold:
            dropped.append(
                {
                    "dropped_feature": feat,
                    "redundant_with": best_kept,
                    f"{method}_corr": round(best_corr, 4),
                    "mutual_info": round(float(ranked[feat]), 4),
                }
            )
        else:
            selected.append(feat)

    dropped_df = pd.DataFrame(dropped)

    print(f"\n=== Redundancy pruning ({method}, |corr| >= {threshold}) ===")
    print(f"Selected {len(selected)} of {len(features)} features (minimal redundancy):")
    for feat in selected:
        print(f"  keep   {feat}  (MI={ranked[feat]:.3f})")
    if len(dropped_df) > 0:
        print(f"\nDropped {len(dropped_df)} redundant feature(s):")
        print(dropped_df.to_string(index=False))
    else:
        print("\nNo redundant features above the threshold.")

    return selected, dropped_df


def plot_mutual_information(
    mi_scores: pd.Series,
    target: str,
    output_dir: Path,
    dpi: int = 300,
    show: bool = False,
    dataset_name: str = "",
    selected: list[str] | None = None,
) -> None:
    """Create and save mutual information bar plot.

    If ``selected`` is provided (redundancy pruning enabled), only the kept
    features are shown; redundant/pruned features are excluded from the chart.
    """
    fig, ax = plt.subplots(figsize=(8, 3.2))

    data = mi_scores.sort_values(ascending=False)
    if selected is not None:
        data = data[data.index.isin(selected)]
    labels = data.index.to_list()
    values = data.values
    x = np.arange(len(labels))
    top = max(values) if len(values) else 1.0

    bars = ax.bar(
        x,
        values,
        color="#34a853",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_ylabel("Mutual Information", fontsize=14, labelpad=2)
    ax.set_xlabel(None)
    ax.set_title("Mutual Information with Target", fontsize=13, pad=6)

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=11,
    )
    ax.tick_params(axis="y", labelsize=11)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + top * 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"mutual_information_{dataset_name}.png"
        if dataset_name
        else "mutual_information.png"
    )
    plt.savefig(output_dir / filename, bbox_inches="tight", dpi=dpi)
    print(f"\nSaved MI plot to {output_dir}/{filename}")

    if show:
        plt.show()
    plt.close()


def save_mi_table(
    mi_scores: pd.Series,
    df: pd.DataFrame,
    target: str,
    output_dir: Path,
    dataset_name: str = "",
) -> None:
    """
    Save MI scores alongside Pearson/Spearman correlations.

    Including the correlations makes it easy to spot non-linear features:
    high mutual information paired with low |Pearson| is the signature of a
    non-linear relationship.
    """
    features = mi_scores.index.to_list()
    pearson = df[features + [target]].corr(method="pearson")[target].drop(target)
    spearman = df[features + [target]].corr(method="spearman")[target].drop(target)

    table = pd.DataFrame(
        {
            "feature": features,
            "mutual_info": mi_scores.values,
            "pearson": pearson[features].values,
            "spearman": spearman[features].values,
        }
    )
    table = table.sort_values("mutual_info", ascending=False)

    filename = (
        f"mutual_information_{dataset_name}.csv"
        if dataset_name
        else "mutual_information.csv"
    )
    output_path = output_dir / filename
    table.to_csv(output_path, index=False)
    print(f"Saved MI table to {output_path}")

    # Flag potential non-linear features: high MI, weak linear correlation.
    nonlinear = table[
        (table["mutual_info"] > table["mutual_info"].median())
        & (table["pearson"].abs() < 0.5)
    ]
    print("\n=== Potentially non-linear features (high MI, low |Pearson|) ===")
    if len(nonlinear) > 0:
        print(nonlinear[["feature", "mutual_info", "pearson"]].to_string(index=False))
    else:
        print("None detected.")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    features = args.features if args.features else DEFAULT_FEATURES

    df = load_and_prepare_data(
        data_path=args.data,
        features=features,
        target=args.target,
        aggregate=args.aggregate,
    )

    if args.filter_active:
        print(f"\nFiltering to {args.target} > 0...")
        df = df[df[args.target] > 0]
        print(f"Filtered data shape: {df.shape}")

    mi_scores = compute_mutual_information(
        df=df,
        features=features,
        target=args.target,
        n_neighbors=args.n_neighbors,
        random_state=args.random_state,
    )

    output_dir = Path(args.output_dir)
    dataset_name = Path(args.data).stem

    # Address MI's redundancy blind spot: keep the highest-MI feature from
    # each correlated cluster and drop the rest.
    selected = None
    if args.prune_redundant:
        selected, dropped_df = select_non_redundant(
            mi_scores=mi_scores,
            df=df,
            threshold=args.redundancy_threshold,
            method=args.redundancy_method,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        sel_name = (
            f"mi_selected_features_{dataset_name}.csv"
            if dataset_name
            else "mi_selected_features.csv"
        )
        pd.DataFrame({"selected_feature": selected}).to_csv(
            output_dir / sel_name, index=False
        )
        print(f"\nSaved selected feature set to {output_dir}/{sel_name}")
        if len(dropped_df) > 0:
            drop_name = (
                f"mi_dropped_features_{dataset_name}.csv"
                if dataset_name
                else "mi_dropped_features.csv"
            )
            dropped_df.to_csv(output_dir / drop_name, index=False)
            print(f"Saved dropped feature table to {output_dir}/{drop_name}")

    plot_mutual_information(
        mi_scores=mi_scores,
        target=args.target,
        output_dir=output_dir,
        dpi=args.dpi,
        show=args.show_plots,
        dataset_name=dataset_name,
        selected=selected,
    )

    save_mi_table(
        mi_scores=mi_scores,
        df=df,
        target=args.target,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
