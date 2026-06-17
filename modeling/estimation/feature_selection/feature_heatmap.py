#!/usr/bin/env python3
"""
Feature correlation heatmap for energy modeling.

Visualizes correlation matrix to identify multicollinearity between features.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
    "delta_cache_misses",
    "delta_branch_instructions",
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
        description="Correlation heatmap for feature multicollinearity analysis"
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
        "--method",
        type=str,
        default="pearson",
        choices=["pearson", "spearman"],
        help="Correlation method (default: pearson)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively",
    )
    return parser.parse_args()


def load_and_prepare_data(
    data_path: str,
    features: list[str],
    target: str,
    aggregate: bool = True,
) -> pd.DataFrame:
    """Load and prepare data for analysis."""
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    if "_time" in df.columns:
        df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")

    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]

    if missing_features:
        print(f"Warning: Features not found: {missing_features}")

    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in data")

    print(f"Available features: {len(available_features)}")

    if aggregate and "_time" in df.columns:
        print("Aggregating data by time interval...")
        df_energy = df[["_time", target]].dropna().drop_duplicates("_time")
        df_agg = df.groupby("_time")[available_features].sum().reset_index()
        df_result = df_agg.merge(df_energy, on="_time", how="left")
    else:
        df_result = df[available_features + [target]].copy()

    df_result[available_features] = df_result[available_features].fillna(0)
    df_result = df_result.dropna(subset=[target])

    print(f"Final data shape: {df_result.shape}")
    return df_result


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    method: str,
    output_dir: Path,
    dataset_name: str,
    dpi: int = 300,
    show: bool = False,
) -> pd.DataFrame:
    """Create and save correlation heatmap."""
    available_features = [f for f in features if f in df.columns]

    # Include target in correlation matrix
    cols = available_features + [target]
    corr_matrix = df[cols].corr(method=method)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} Correlation"},
        ax=ax,
    )

    # Styling
    ax.set_title(
        f"Feature Correlation Heatmap ({method.capitalize()})",
        fontsize=14,
        pad=10,
    )

    # Rotate labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"correlation_heatmap_{method}_{dataset_name}.png"
    plt.savefig(output_dir / filename, bbox_inches="tight", dpi=dpi)
    print(f"\nSaved heatmap to {output_dir}/{filename}")

    if show:
        plt.show()
    plt.close()

    return corr_matrix


def find_high_correlations(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.9,
    exclude_target: str = None,
) -> pd.DataFrame:
    """Find pairs of features with high correlation."""
    high_corr = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col_i = corr_matrix.columns[i]
            col_j = corr_matrix.columns[j]

            # Skip if either is the target
            if exclude_target and (col_i == exclude_target or col_j == exclude_target):
                continue

            corr_val = corr_matrix.iloc[i, j]

            if abs(corr_val) >= threshold:
                high_corr.append(
                    {
                        "feature_1": col_i,
                        "feature_2": col_j,
                        "correlation": corr_val,
                    }
                )

    df_high_corr = pd.DataFrame(high_corr)

    if len(df_high_corr) > 0:
        df_high_corr = df_high_corr.sort_values("correlation", ascending=False, key=abs)

    return df_high_corr


def save_correlation_matrix(
    corr_matrix: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    method: str,
) -> None:
    """Save full correlation matrix to CSV."""
    filename = f"correlation_matrix_{method}_{dataset_name}.csv"
    output_path = output_dir / filename
    corr_matrix.to_csv(output_path)
    print(f"Saved correlation matrix to {output_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    features = args.features if args.features else DEFAULT_FEATURES

    # Load data
    df = load_and_prepare_data(
        data_path=args.data,
        features=features,
        target=args.target,
        aggregate=args.aggregate,
    )

    # Filter active intervals
    if args.filter_active:
        print(f"\nFiltering to {args.target} > 0...")
        df = df[df[args.target] > 0]
        print(f"Filtered data shape: {df.shape}")

    # Extract dataset name
    dataset_name = Path(args.data).stem
    output_dir = Path(args.output_dir)

    # Create heatmap
    corr_matrix = plot_correlation_heatmap(
        df=df,
        features=features,
        target=args.target,
        method=args.method,
        output_dir=output_dir,
        dataset_name=dataset_name,
        dpi=args.dpi,
        show=args.show_plots,
    )

    # Save correlation matrix
    save_correlation_matrix(
        corr_matrix=corr_matrix,
        output_dir=output_dir,
        dataset_name=dataset_name,
        method=args.method,
    )

    # Find and report high correlations
    print("\n=== High Feature-Feature Correlations (|r| >= 0.9) ===")
    high_corr = find_high_correlations(
        corr_matrix=corr_matrix,
        threshold=0.9,
        exclude_target=args.target,
    )

    if len(high_corr) > 0:
        print(high_corr.to_string(index=False))
        print("\n⚠️  Consider removing one feature from each highly correlated pair")

        # Save high correlations
        filename = f"high_correlations_{args.method}_{dataset_name}.csv"
        high_corr.to_csv(output_dir / filename, index=False)
        print(f"Saved high correlations to {output_dir}/{filename}")
    else:
        print("No feature pairs with |correlation| >= 0.9 found")

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
