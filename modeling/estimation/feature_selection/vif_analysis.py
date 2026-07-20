#!/usr/bin/env python3
"""
Variance Inflation Factor (VIF) analysis for energy modeling.

Quantifies multicollinearity among features. For each feature, VIF measures
how much its variance is inflated by linear dependence on the other features:

    VIF_i = 1 / (1 - R_i^2)

where R_i^2 is the coefficient of determination from regressing feature i on
all remaining features. Rules of thumb: VIF < 5 is fine, 5-10 is moderate, and
VIF > 10 indicates problematic multicollinearity.

VIF is computed with scikit-learn's LinearRegression to avoid adding a
statsmodels dependency.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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
        description="VIF (multicollinearity) analysis for energy modeling features"
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
        help="Directory to save outputs (default: plots)",
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
        "--vif-threshold",
        type=float,
        default=10.0,
        help="VIF above which a feature is flagged as severe (default: 10.0)",
    )
    parser.add_argument(
        "--iterative",
        action="store_true",
        help=(
            "Iteratively drop the highest-VIF feature and recompute until all "
            "remaining features fall below the threshold"
        ),
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
    Load data and prepare it for VIF analysis.

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


def compute_vif(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Compute the Variance Inflation Factor for each feature.

    Each feature is regressed on all the others; VIF = 1 / (1 - R^2).
    Constant features (zero variance) get VIF = 1.0, and perfectly collinear
    features (R^2 ~ 1) get VIF = inf.

    Args:
        df: DataFrame with the feature columns
        features: List of feature names

    Returns:
        DataFrame with columns [feature, VIF], sorted high to low
    """
    available_features = [f for f in features if f in df.columns]
    feature_matrix = df[available_features].to_numpy(dtype=float)

    vif_values: list[float] = []
    for i in range(feature_matrix.shape[1]):
        y_i = feature_matrix[:, i]

        # A constant feature cannot be inflated by the others.
        if np.var(y_i) == 0.0:
            vif_values.append(1.0)
            continue

        x_others = np.delete(feature_matrix, i, axis=1)
        model = LinearRegression()
        model.fit(x_others, y_i)
        r_squared = model.score(x_others, y_i)

        if r_squared >= 1.0 - 1e-12:
            vif_values.append(float("inf"))
        else:
            vif_values.append(1.0 / (1.0 - r_squared))

    vif_df = pd.DataFrame({"feature": available_features, "VIF": vif_values})
    vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)
    return vif_df


def iterative_vif_elimination(
    df: pd.DataFrame,
    features: list[str],
    threshold: float = 10.0,
) -> tuple[list[str], pd.DataFrame]:
    """
    Repeatedly drop the highest-VIF feature until all fall below threshold.

    This is the standard practical way to resolve multicollinearity: removing
    the single worst offender often pulls several correlated features back into
    an acceptable range, so VIF is recomputed after each removal.

    Args:
        df: DataFrame with the feature columns
        features: Starting list of feature names
        threshold: Stop once the maximum VIF is at or below this value

    Returns:
        Tuple of (retained features, log of removed features)
    """
    remaining = [f for f in features if f in df.columns]
    removal_log: list[dict] = []

    while len(remaining) > 1:
        vif_df = compute_vif(df, remaining)
        worst = vif_df.iloc[0]
        worst_vif = float(worst["VIF"])

        if worst_vif <= threshold:
            break

        removal_log.append(
            {
                "removed_feature": worst["feature"],
                "VIF_at_removal": round(worst_vif, 4)
                if np.isfinite(worst_vif)
                else worst_vif,
                "features_remaining": len(remaining) - 1,
            }
        )
        remaining.remove(str(worst["feature"]))

    log_df = pd.DataFrame(removal_log)

    print(f"\n=== Iterative VIF elimination (threshold = {threshold}) ===")
    if len(log_df) > 0:
        print(f"Removed {len(log_df)} feature(s) in order:")
        print(log_df.to_string(index=False))
    else:
        print("No features exceeded the threshold; nothing removed.")
    print(f"\nRetained {len(remaining)} feature(s): {remaining}")

    return remaining, log_df


def report_vif(vif_df: pd.DataFrame, threshold: float) -> None:
    """Print a tiered interpretation of the VIF results."""
    print("\n=== VIF Scores ===")
    print(vif_df.to_string(index=False))

    severe = vif_df[vif_df["VIF"] > threshold]
    moderate = vif_df[(vif_df["VIF"] >= 5) & (vif_df["VIF"] <= threshold)]
    low = vif_df[vif_df["VIF"] < 5]

    print("\n--- Interpretation ---")
    print(f"  Severe   (VIF > {threshold:g}): {len(severe)}  -> drop / regularize")
    if len(severe) > 0:
        print(f"           {severe['feature'].to_list()}")
    print(f"  Moderate (5 <= VIF <= {threshold:g}): {len(moderate)}  -> monitor")
    if len(moderate) > 0:
        print(f"           {moderate['feature'].to_list()}")
    print(f"  Low      (VIF < 5): {len(low)}  -> acceptable")


def plot_vif(
    vif_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    threshold: float = 10.0,
    dpi: int = 300,
    show: bool = False,
) -> None:
    """Create and save a horizontal VIF bar chart with reference lines."""
    data = vif_df.sort_values("VIF", ascending=True)
    labels = data["feature"].to_list()
    values = data["VIF"].to_numpy(dtype=float)

    # Cap infinities for display so the chart stays readable.
    finite = values[np.isfinite(values)]
    cap = (finite.max() * 1.15) if finite.size else threshold * 2
    display_values = np.where(np.isfinite(values), values, cap)

    def tier_color(v: float) -> str:
        if v > threshold:
            return "#eb3434"  # severe
        if v >= 5:
            return "#eb9834"  # moderate
        return "#34a853"  # low

    colors = [tier_color(v) for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.32 * len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, display_values, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("VIF", fontsize=14, labelpad=2)
    ax.set_title("Variance Inflation Factor", fontsize=13, pad=6)

    # Reference lines at the conventional 5 and threshold cut-offs.
    ax.axvline(5, color="#eb9834", linestyle="--", linewidth=1, alpha=0.8)
    ax.axvline(threshold, color="#eb3434", linestyle="--", linewidth=1, alpha=0.8)

    for yi, (val, disp) in enumerate(zip(values, display_values)):
        label = "inf" if not np.isfinite(val) else f"{val:.1f}"
        ax.text(disp, yi, f" {label}", va="center", ha="left", fontsize=9)

    ax.margins(x=0.12)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"vif_{dataset_name}.png" if dataset_name else "vif.png"
    plt.savefig(output_dir / filename, bbox_inches="tight", dpi=dpi)
    print(f"\nSaved VIF plot to {output_dir}/{filename}")

    if show:
        plt.show()
    plt.close()


def save_vif_table(
    vif_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
) -> None:
    """Save the VIF scores to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"vif_{dataset_name}.csv" if dataset_name else "vif.csv"
    output_path = output_dir / filename
    vif_df.to_csv(output_path, index=False)
    print(f"Saved VIF table to {output_path}")


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

    output_dir = Path(args.output_dir)
    dataset_name = Path(args.data).stem

    # Full VIF on all features.
    vif_df = compute_vif(df, features)
    report_vif(vif_df, threshold=args.vif_threshold)

    plot_vif(
        vif_df=vif_df,
        output_dir=output_dir,
        dataset_name=dataset_name,
        threshold=args.vif_threshold,
        dpi=args.dpi,
        show=args.show_plots,
    )
    save_vif_table(vif_df, output_dir, dataset_name)

    # Optional iterative elimination to a minimal low-VIF feature set.
    if args.iterative:
        retained, log_df = iterative_vif_elimination(
            df=df,
            features=features,
            threshold=args.vif_threshold,
        )

        retained_name = (
            f"vif_retained_features_{dataset_name}.csv"
            if dataset_name
            else "vif_retained_features.csv"
        )
        pd.DataFrame({"retained_feature": retained}).to_csv(
            output_dir / retained_name, index=False
        )
        print(f"\nSaved retained feature set to {output_dir}/{retained_name}")

        if len(log_df) > 0:
            log_name = (
                f"vif_removed_features_{dataset_name}.csv"
                if dataset_name
                else "vif_removed_features.csv"
            )
            log_df.to_csv(output_dir / log_name, index=False)
            print(f"Saved removal log to {output_dir}/{log_name}")

        # Final VIF on the retained set, for confirmation + plot.
        final_vif = compute_vif(df, retained)
        report_vif(final_vif, threshold=args.vif_threshold)
        plot_vif(
            vif_df=final_vif,
            output_dir=output_dir,
            dataset_name=f"{dataset_name}_retained",
            threshold=args.vif_threshold,
            dpi=args.dpi,
            show=args.show_plots,
        )
        save_vif_table(final_vif, output_dir, f"{dataset_name}_retained")

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
