#!/usr/bin/env python3
"""
Feature correlation analysis for energy modeling.

Computes Pearson and Spearman correlation coefficients between features
and the target variable, and generates publication-ready plots.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_FEATURES = [
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Correlation analysis for energy modeling features"
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
    Load data and prepare it for correlation analysis.

    Args:
        data_path: Path to parquet file
        features: List of feature column names
        target: Target variable name
        aggregate: If True, aggregate by _time interval

    Returns:
        DataFrame ready for correlation analysis
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Convert time column if it exists
    if "_time" in df.columns:
        df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")

    # Check which features are available
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

        # Extract interval energy (should be same for all processes in interval)
        df_energy = df[["_time", target]].dropna().drop_duplicates("_time")

        # Sum features per interval
        df_agg = df.groupby("_time")[available_features].sum().reset_index()

        # Merge with target
        df_result = df_agg.merge(df_energy, on="_time", how="left")

        print(f"Aggregated shape: {df_result.shape}")
    else:
        df_result = df[available_features + [target]].copy()

    # Fill NaN values
    df_result[available_features] = df_result[available_features].fillna(0)
    df_result = df_result.dropna(subset=[target])

    print(f"Final data shape: {df_result.shape}")
    print(f"\nTarget variable '{target}' statistics:")
    print(df_result[target].describe())

    return df_result


def compute_correlations(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute Pearson and Spearman correlations.

    Args:
        df: DataFrame with features and target
        features: List of feature names
        target: Target variable name

    Returns:
        Tuple of (pearson_correlations, spearman_correlations)
    """
    # Filter to available features
    available_features = [f for f in features if f in df.columns]

    print("\nComputing correlations...")

    # Pearson correlation
    pearson_corr = df[available_features + [target]].corr(method="pearson")[target]
    pearson_corr = pearson_corr.drop(target)

    # Spearman correlation
    spearman_corr = df[available_features + [target]].corr(method="spearman")[target]
    spearman_corr = spearman_corr.drop(target)

    print("\n=== Pearson Correlation ===")
    print(pearson_corr.sort_values(ascending=False))

    print("\n=== Spearman Correlation ===")
    print(spearman_corr.sort_values(ascending=False))

    return pearson_corr, spearman_corr


def plot_pearson_correlation(
    pearson_corr: pd.Series,
    target: str,
    output_dir: Path,
    dpi: int = 300,
    show: bool = False,
    dataset_name: str = "",
) -> None:
    """Create and save Pearson correlation bar plot."""
    fig, ax = plt.subplots(figsize=(8, 3.2))

    # Sort by correlation value
    data = pearson_corr.sort_values(ascending=False)
    labels = data.index.to_list()
    values = data.values
    x = np.arange(len(labels))

    # Create bars
    bars = ax.bar(
        x,
        values,
        color="#3477eb",
        edgecolor="black",
        linewidth=0.6,
    )

    # Labels and title
    ax.set_ylabel(r"Correlation $r$", fontsize=14, labelpad=2)
    ax.set_xlabel(None)
    ax.set_title("Pearson Correlation", fontsize=13, pad=6)

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=11,
    )
    ax.tick_params(axis="y", labelsize=11)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02 if val >= 0 else val - 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9.5,
        )

    # Grid and styling
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.5)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"pearson_correlation_{dataset_name}.png"
        if dataset_name
        else "pearson_correlation.png"
    )
    plt.savefig(output_dir / filename, bbox_inches="tight", dpi=dpi)
    print(f"\nSaved Pearson plot to {output_dir}/{filename}")

    if show:
        plt.show()
    plt.close()


def plot_spearman_correlation(
    spearman_corr: pd.Series,
    target: str,
    output_dir: Path,
    dpi: int = 300,
    show: bool = False,
    dataset_name: str = "",
) -> None:
    """Create and save Spearman correlation bar plot."""
    fig, ax = plt.subplots(figsize=(8, 3.2))

    # Sort by correlation value
    data = spearman_corr.sort_values(ascending=False)
    labels = data.index.to_list()
    values = data.values
    x = np.arange(len(labels))

    # Create bars
    bars = ax.bar(
        x,
        values,
        color="#eb9834",
        edgecolor="black",
        linewidth=0.6,
    )

    # Labels and title
    ax.set_ylabel(r"Correlation $\rho$", fontsize=14, labelpad=2)
    ax.set_xlabel(None)
    ax.set_title("Spearman Correlation", fontsize=13, pad=6)

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=11,
    )
    ax.tick_params(axis="y", labelsize=11)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02 if val >= 0 else val - 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9.5,
        )

    # Grid and styling
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.5)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"spearman_correlation_{dataset_name}.png"
        if dataset_name
        else "spearman_correlation.png"
    )
    plt.savefig(output_dir / filename, bbox_inches="tight", dpi=dpi)
    print(f"Saved Spearman plot to {output_dir}/{filename}")

    if show:
        plt.show()
    plt.close()


def save_correlation_table(
    pearson_corr: pd.Series,
    spearman_corr: pd.Series,
    output_dir: Path,
    dataset_name: str = "",
) -> None:
    """Save correlation values to CSV file."""
    df_corr = pd.DataFrame(
        {
            "feature": pearson_corr.index,
            "pearson": pearson_corr.values,
            "spearman": spearman_corr.values,
        }
    )
    df_corr = df_corr.sort_values("spearman", ascending=False)

    filename = (
        f"correlations_{dataset_name}.csv" if dataset_name else "correlations.csv"
    )
    output_path = output_dir / filename
    df_corr.to_csv(output_path, index=False)
    print(f"Saved correlation table to {output_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Use provided features or defaults
    features = args.features if args.features else DEFAULT_FEATURES

    # Load and prepare data
    df = load_and_prepare_data(
        data_path=args.data,
        features=features,
        target=args.target,
        aggregate=args.aggregate,
    )

    # Filter to active intervals if requested
    if args.filter_active:
        print(f"\nFiltering to {args.target} > 0...")
        df = df[df[args.target] > 0]
        print(f"Filtered data shape: {df.shape}")

    # Compute correlations
    pearson_corr, spearman_corr = compute_correlations(
        df=df,
        features=features,
        target=args.target,
    )

    # Create output directory
    output_dir = Path(args.output_dir)

    # Extract dataset name from data path
    dataset_name = Path(args.data).stem

    # Generate plots
    plot_pearson_correlation(
        pearson_corr=pearson_corr,
        target=args.target,
        output_dir=output_dir,
        dpi=args.dpi,
        show=args.show_plots,
        dataset_name=dataset_name,
    )

    plot_spearman_correlation(
        spearman_corr=spearman_corr,
        target=args.target,
        output_dir=output_dir,
        dpi=args.dpi,
        show=args.show_plots,
        dataset_name=dataset_name,
    )

    # Save correlation table
    save_correlation_table(
        pearson_corr=pearson_corr,
        spearman_corr=spearman_corr,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
