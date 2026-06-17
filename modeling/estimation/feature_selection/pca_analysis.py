#!/usr/bin/env python3
"""
PCA analysis for energy modeling.

Applies Principal Component Analysis to the (standardized) feature matrix to:

  1. Measure intrinsic dimensionality via the explained-variance scree plot.
  2. Reveal feature groupings via a PC1/PC2 loading biplot.
  3. Show how well a chosen feature subset spans the principal axes via a
     loading-coverage heatmap.
  4. Visualize intervals in PC space colored by interval_energy (workload map).
  5. Detect temporal drift by plotting PC scores over time.

All features are z-score standardized before PCA so that differences in
physical units do not dominate the decomposition.

Usage example
-------------
python pca_analysis.py \\
    --data ../../data/out.parquet \\
    --aggregate \\
    --filter-active \\
    --output-dir plots \\
    --highlight-features delta_instructions delta_cache_misses \\
                         delta_branch_instructions syscall_class_other
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

# Features currently used in the CVXPY estimator (shown highlighted in plots).
DEFAULT_HIGHLIGHT = [
    "delta_instructions",
    "delta_cache_misses",
    "delta_branch_instructions",
    "syscall_class_other",
]

# Colors
_COLOR_HIGHLIGHT = "#eb3434"
_COLOR_DEFAULT = "#3477eb"
_COLOR_SCATTER = "viridis"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCA analysis for energy modeling features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Target variable (used to color the workload-map scatter)",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=None,
        help="Feature columns to include (default: predefined list)",
    )
    parser.add_argument(
        "--highlight-features",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Subset of features to highlight in biplot and coverage heatmap "
            "(default: DEFAULT_HIGHLIGHT — the 4 CVXPY estimator features)"
        ),
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help=(
            "Number of PCs to retain. Defaults to min(n_features, n_samples). "
            "Plots always use the first 2 PCs for 2-D views."
        ),
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.90,
        help="Cumulative explained-variance target used as a reference line in the scree plot",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save all outputs",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved PNG plots",
    )
    parser.add_argument(
        "--filter-active",
        action="store_true",
        help="Keep only intervals where target > 0",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate process-level rows by time interval before fitting PCA",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display each plot interactively after saving",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading (mirrors pattern used by the other feature_selection scripts)
# ---------------------------------------------------------------------------


def load_and_prepare_data(
    data_path: str,
    features: list[str],
    target: str,
    aggregate: bool = True,
) -> pd.DataFrame:
    """Load the parquet dataset, optionally aggregate by time interval."""
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    if "_time" in df.columns:
        df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")

    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"Warning: features not found in data and will be skipped: {missing}")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data.")

    print(f"Using {len(available)} features. Data shape: {df.shape}")

    if aggregate and "_time" in df.columns:
        print("Aggregating by time interval...")
        df_energy = df[["_time", target]].dropna().drop_duplicates("_time")
        df_agg = df.groupby("_time")[available].sum().reset_index()
        df_result = df_agg.merge(df_energy, on="_time", how="left")
        print(f"Aggregated shape: {df_result.shape}")
    else:
        cols = list({*available, target, "_time"} & set(df.columns))
        df_result = df[cols].copy()

    df_result[available] = df_result[available].fillna(0)
    df_result = df_result.dropna(subset=[target])

    print(f"Final shape: {df_result.shape}")
    return df_result, available


# ---------------------------------------------------------------------------
# PCA fitting
# ---------------------------------------------------------------------------


def fit_pca(
    df: pd.DataFrame,
    features: list[str],
    n_components: int | None,
) -> tuple[PCA, StandardScaler, np.ndarray]:
    """
    Standardize features and fit PCA.

    Returns the fitted PCA object, the scaler, and the transformed score
    matrix (n_intervals x n_components).
    """
    X = df[features].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_comp = n_components or min(len(features), X_scaled.shape[0])
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X_scaled)

    print(f"\nFitted PCA: {pca.n_components_} components retained")
    ev = pca.explained_variance_ratio_
    cumev = np.cumsum(ev)
    for i, (e, c) in enumerate(zip(ev, cumev)):
        print(f"  PC{i + 1:>2}: {e * 100:5.1f}%  (cumulative {c * 100:5.1f}%)")

    return pca, scaler, scores


# ---------------------------------------------------------------------------
# Plot 1 — Scree / explained-variance plot
# ---------------------------------------------------------------------------


def plot_scree(
    pca: PCA,
    variance_threshold: float,
    output_dir: Path,
    dataset_name: str,
    dpi: int,
    show: bool,
) -> None:
    """Bar chart of per-PC explained variance with a cumulative line."""
    ev = pca.explained_variance_ratio_ * 100
    cumev = np.cumsum(ev)
    x = np.arange(1, len(ev) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    bars = ax1.bar(
        x,
        ev,
        color=_COLOR_DEFAULT,
        edgecolor="black",
        linewidth=0.6,
        label="Per-PC variance",
        zorder=3,
    )
    # Annotate each bar with its value
    for bar, val in zip(bars, ev):
        if val >= 2.0:  # skip tiny bars at the tail
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.3,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )

    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Explained Variance (%)", fontsize=12)
    ax1.set_xticks(x)
    ax1.tick_params(axis="both", labelsize=10)
    ax1.set_ylim(0, max(ev) * 1.45)  # extra headroom so bar labels don't clash
    ax1.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        cumev,
        color="#eb9834",
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="Cumulative",
        zorder=4,
    )
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=12)
    ax2.set_ylim(0, 115)  # push ceiling so the threshold line has breathing room
    ax2.tick_params(axis="y", labelsize=10)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # Draw the threshold line on ax2 only and label it directly on the right
    n_thresh = int(np.searchsorted(cumev / 100, variance_threshold)) + 1
    thresh_pct = variance_threshold * 100
    ax2.axhline(
        thresh_pct,
        color="#eb3434",
        linestyle="--",
        linewidth=1.4,
        zorder=2,
    )
    ax2.text(
        len(ev) + 0.4,
        thresh_pct + 1.5,
        f"{thresh_pct:.0f}% cumulative\n(PC{n_thresh})",
        color="#eb3434",
        fontsize=9,
        va="bottom",
        ha="left",
    )

    ax1.set_title(
        f"Explained Variance per PC  "
        f"({n_thresh} PCs needed for {variance_threshold * 100:.0f}% of variance)",
        fontsize=12,
        pad=8,
    )

    # Combined legend (bars + cumulative line)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        fontsize=9,
        loc="upper right",
        framealpha=0.9,
    )

    plt.tight_layout(pad=0.6)
    _save(fig, output_dir, f"pca_scree_{dataset_name}.png", dpi, show)


# ---------------------------------------------------------------------------
# Plot 2 — Loading biplot (PC1 vs PC2)
# ---------------------------------------------------------------------------


def plot_loading_biplot(
    pca: PCA,
    features: list[str],
    highlight: list[str],
    output_dir: Path,
    dataset_name: str,
    dpi: int,
    show: bool,
) -> None:
    """
    Arrow biplot of feature loadings on PC1 and PC2.

    Arrows are scaled so the longest one reaches 0.85 of the unit circle.
    Labels are fanned out evenly across the angular spread of the arrows and
    connected back to their arrow tip by a thin dotted leader line, preventing
    the pile-up that occurs when many features load on the same axis.

    Highlighted features (the current estimator feature set) are drawn in red.
    """
    from matplotlib.lines import Line2D

    loadings = pca.components_[:2].T  # shape: (n_features, 2)

    # Scale so the longest arrow reaches 0.85 (fills the circle without clipping)
    arrow_lengths = np.sqrt((loadings**2).sum(axis=1))
    scale = 0.85 / arrow_lengths.max() if arrow_lengths.max() > 0 else 1.0
    scaled = loadings * scale

    fig, ax = plt.subplots(figsize=(12, 8))

    # Unit circle for reference
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#dddddd", linewidth=1.0, zorder=0)

    # Draw arrows (no inline labels yet)
    for i, feat in enumerate(features):
        lx, ly = scaled[i, 0], scaled[i, 1]
        is_highlight = feat in highlight
        color = _COLOR_HIGHLIGHT if is_highlight else _COLOR_DEFAULT
        lw = 2.4 if is_highlight else 1.5
        zorder = 4 if is_highlight else 2
        ax.annotate(
            "",
            xy=(lx, ly),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                mutation_scale=12,
            ),
            zorder=zorder,
        )

    # Fan labels evenly across the actual angular spread at a fixed outer radius.
    # Sorting by angle then distributing evenly prevents any two labels from
    # sitting at the same position, even when many arrows point in the same direction.
    angles = np.arctan2(scaled[:, 1], scaled[:, 0])
    order = np.argsort(angles)

    a_min = angles.min() - 0.15
    a_max = angles.max() + 0.15
    label_angles = np.linspace(a_min, a_max, len(features))
    label_radius = 1.18

    for rank, orig_idx in enumerate(order):
        lx, ly = scaled[orig_idx]
        la = label_angles[rank]
        feat = features[orig_idx]
        is_highlight = feat in highlight
        color = _COLOR_HIGHLIGHT if is_highlight else _COLOR_DEFAULT

        tx = label_radius * np.cos(la)
        ty = label_radius * np.sin(la)

        # Dotted leader line from arrow tip to label anchor
        ax.plot(
            [lx, tx],
            [ly, ty],
            color=color,
            lw=0.7,
            alpha=0.5,
            linestyle=":",
            zorder=1,
        )

        ha = "left" if np.cos(la) >= 0 else "right"
        va = "bottom" if np.sin(la) >= 0 else "top"

        ax.text(
            tx,
            ty,
            feat,
            ha=ha,
            va=va,
            fontsize=9.5,
            color=color,
            fontweight="bold" if is_highlight else "normal",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
        )

    pct = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1  ({pct[0]:.1f}% variance)", fontsize=13)
    ax.set_ylabel(f"PC2  ({pct[1]:.1f}% variance)", fontsize=13)
    ax.set_title("PCA Loading Biplot  (PC1 vs PC2)", fontsize=13, pad=10)
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.axvline(0, color="#cccccc", linewidth=0.8)

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=_COLOR_HIGHLIGHT,
            lw=2.2,
            label="Highlighted (estimator) features",
        ),
        Line2D([0], [0], color=_COLOR_DEFAULT, lw=1.5, label="Other features"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower left", framealpha=0.9)

    # Wide x-range so right-side labels (ha='left') are not clipped
    ax.set_xlim(-1.5, 2.4)
    ax.set_ylim(-1.5, 1.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    _save(fig, output_dir, f"pca_biplot_{dataset_name}.png", dpi, show)


# ---------------------------------------------------------------------------
# Plot 3 — Loading heatmap (features × top-k PCs)
# ---------------------------------------------------------------------------


def plot_loading_heatmap(
    pca: PCA,
    features: list[str],
    highlight: list[str],
    output_dir: Path,
    dataset_name: str,
    dpi: int,
    show: bool,
    n_components_shown: int = 6,
) -> None:
    """
    Heatmap of |loading| for features × first k PCs.

    Features are sorted by their dominant PC, grouping naturally correlated
    features together. Highlighted features (estimator subset) are labelled in
    red so gaps in PC coverage are immediately visible.
    """
    n_show = min(n_components_shown, pca.n_components_)
    loadings = np.abs(pca.components_[:n_show].T)  # (n_features, n_show)

    # Sort features by dominant PC then by descending loading within that PC
    dominant_pc = np.argmax(loadings, axis=1)
    order = np.lexsort((-loadings[np.arange(len(features)), dominant_pc], dominant_pc))
    sorted_features = [features[i] for i in order]
    sorted_loadings = loadings[order]

    pct = pca.explained_variance_ratio_[:n_show] * 100
    col_labels = [f"PC{i + 1}\n({p:.1f}%)" for i, p in enumerate(pct)]

    row_h = 0.48  # height per row in inches
    col_w = 1.05  # width per column
    fig_w = max(7, n_show * col_w + 2.5)  # extra space for y-tick labels
    fig_h = max(5, len(features) * row_h + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(sorted_loadings, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(np.arange(n_show))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.xaxis.set_tick_params(length=0)
    ax.set_yticks(np.arange(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=10)

    # Colour and bold the highlighted feature rows
    for tick, feat in zip(ax.get_yticklabels(), sorted_features):
        if feat in highlight:
            tick.set_color(_COLOR_HIGHLIGHT)
            tick.set_fontweight("bold")

    # Draw a subtle separator between dominant-PC groups
    prev_pc = dominant_pc[order[0]]
    for row_i, orig_idx in enumerate(order[1:], start=1):
        curr_pc = dominant_pc[orig_idx]
        if curr_pc != prev_pc:
            ax.axhline(row_i - 0.5, color="white", lw=2)
        prev_pc = curr_pc

    # Annotate every cell with its loading value
    for row_i in range(len(sorted_features)):
        for col_j in range(n_show):
            val = sorted_loadings[row_i, col_j]
            text_color = "white" if val > 0.55 else "#333333"
            ax.text(
                col_j,
                row_i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
                fontweight="bold" if val == sorted_loadings[row_i].max() else "normal",
            )

    ax.set_title(
        "|Loading| heatmap  (features × PCs, sorted by dominant PC)",
        fontsize=12,
        pad=10,
    )
    cbar = plt.colorbar(im, ax=ax, label="|Loading|", shrink=0.55, pad=0.02)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout(pad=0.7)
    _save(fig, output_dir, f"pca_loading_heatmap_{dataset_name}.png", dpi, show)


# ---------------------------------------------------------------------------
# Plot 4 — Workload map: intervals in PC1/PC2 space, colored by energy
# ---------------------------------------------------------------------------


def plot_workload_map(
    scores: np.ndarray,
    energy: pd.Series,
    pca: PCA,
    output_dir: Path,
    dataset_name: str,
    target: str,
    dpi: int,
    show: bool,
    clip_percentile: float = 99.0,
) -> None:
    """
    Scatter plot of every time interval in (PC1, PC2) space.

    Extreme outliers (beyond clip_percentile on either axis) are excluded from
    the main view so the dense core of the distribution is visible. Excluded
    points are counted and annotated. The colour scale is also clipped to the
    same percentile of energy to prevent a handful of spikes from washing out
    the gradient across normal operating intervals.
    """
    pct = pca.explained_variance_ratio_ * 100
    energy_vals = energy.values.astype(float)

    # Determine axis and colour clip limits
    pc1_max = np.percentile(scores[:, 0], clip_percentile)
    pc1_min = np.percentile(scores[:, 0], 100 - clip_percentile)
    pc2_max = np.percentile(scores[:, 1], clip_percentile)
    pc2_min = np.percentile(scores[:, 1], 100 - clip_percentile)
    e_max = np.percentile(energy_vals, clip_percentile)
    e_min = energy_vals.min()

    # Split into main cloud and outliers
    in_view = (
        (scores[:, 0] >= pc1_min)
        & (scores[:, 0] <= pc1_max)
        & (scores[:, 1] >= pc2_min)
        & (scores[:, 1] <= pc2_max)
    )
    n_outliers = int((~in_view).sum())

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Main cloud
    sc = ax.scatter(
        scores[in_view, 0],
        scores[in_view, 1],
        c=np.clip(energy_vals[in_view], e_min, e_max),
        cmap=_COLOR_SCATTER,
        s=14,
        alpha=0.75,
        linewidths=0,
        vmin=e_min,
        vmax=e_max,
        zorder=2,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f"{target}  (clipped at {clip_percentile:.0f}th pct)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel(f"PC1  ({pct[0]:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2  ({pct[1]:.1f}% variance)", fontsize=12)
    ax.set_title("Workload Map  (intervals in PC space)", fontsize=12, pad=8)

    if n_outliers > 0:
        ax.text(
            0.98,
            0.98,
            f"{n_outliers} extreme outlier(s) outside\n"
            f"PC1 ∈ [{pc1_min:.1f}, {pc1_max:.1f}] / "
            f"PC2 ∈ [{pc2_min:.1f}, {pc2_max:.1f}]\nnot shown",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#888888",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85),
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle=":", alpha=0.3)

    plt.tight_layout(pad=0.6)
    _save(fig, output_dir, f"pca_workload_map_{dataset_name}.png", dpi, show)


# ---------------------------------------------------------------------------
# Plot 5 — PC scores over time (temporal drift)
# ---------------------------------------------------------------------------


def plot_temporal_drift(
    scores: np.ndarray,
    times: pd.Series,
    pca: PCA,
    output_dir: Path,
    dataset_name: str,
    dpi: int,
    show: bool,
    n_pcs: int = 3,
    clip_percentile: float = 98.0,
) -> None:
    """
    Line plot of the first n_pcs PC scores over time.

    Each panel is individually y-clipped to clip_percentile so that a single
    extreme spike does not flatten the rest of the signal. The actual peak
    value is annotated directly on the panel.

    Systematic drift or regime changes indicate workload non-stationarity —
    a key risk for a chronological train/test split.
    """
    n = min(n_pcs, scores.shape[1])
    pct = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(n, 1, figsize=(13, 3.0 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = ["#3477eb", "#eb9834", "#34a853"]

    for i, ax in enumerate(axes):
        s = scores[:, i]
        color = colors[i % len(colors)]

        ax.plot(times, s, linewidth=0.85, color=color, alpha=0.9)
        ax.axhline(0, color="#cccccc", linewidth=0.9)

        # Clip y-axis to show the bulk of the distribution clearly
        y_upper = np.percentile(s, clip_percentile)
        y_lower = np.percentile(s, 100 - clip_percentile)
        y_pad = max((y_upper - y_lower) * 0.12, 0.05)
        ax.set_ylim(
            y_lower - y_pad, y_upper + y_pad * 3
        )  # extra top room for annotation

        # Annotate the true peak if it was clipped
        s_max = float(s.max())
        s_min = float(s.min())
        if s_max > y_upper:
            t_peak = times.iloc[int(np.argmax(s))]
            ax.annotate(
                f"peak = {s_max:.1f}",
                xy=(t_peak, y_upper),
                xytext=(t_peak, y_upper + y_pad * 1.5),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.1),
                fontsize=8.5,
                color=color,
                ha="center",
            )
        if s_min < y_lower:
            t_trough = times.iloc[int(np.argmin(s))]
            ax.annotate(
                f"min = {s_min:.1f}",
                xy=(t_trough, y_lower),
                xytext=(t_trough, y_lower - y_pad * 1.5),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.1),
                fontsize=8.5,
                color=color,
                ha="center",
            )

        ax.set_ylabel(f"PC{i + 1}  ({pct[i]:.1f}%)", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.grid(axis="x", linestyle=":", alpha=0.2)

    axes[-1].set_xlabel("Time", fontsize=12)
    axes[0].set_title(
        f"PC Scores Over Time  (temporal drift, y-axis clipped to {clip_percentile:.0f}th percentile)",
        fontsize=12,
        pad=8,
    )

    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout(pad=0.6)
    _save(fig, output_dir, f"pca_temporal_drift_{dataset_name}.png", dpi, show)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def save_loadings_csv(
    pca: PCA,
    features: list[str],
    output_dir: Path,
    dataset_name: str,
) -> None:
    """Save the full loading matrix to CSV for inspection."""
    n = pca.n_components_
    cols = [f"PC{i + 1}" for i in range(n)]
    df_loadings = pd.DataFrame(pca.components_.T, index=features, columns=cols)
    df_loadings.index.name = "feature"
    df_loadings.insert(0, "dominant_pc", (df_loadings.abs().values.argmax(axis=1) + 1))

    filename = (
        f"pca_loadings_{dataset_name}.csv" if dataset_name else "pca_loadings.csv"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    df_loadings.to_csv(output_dir / filename)
    print(f"Saved loadings table to {output_dir / filename}")


def save_scores_csv(
    scores: np.ndarray,
    times: pd.Series | None,
    energy: pd.Series,
    output_dir: Path,
    dataset_name: str,
    target: str,
) -> None:
    """Save PC scores (one row per interval) together with time and energy."""
    cols = {f"PC{i + 1}": scores[:, i] for i in range(scores.shape[1])}
    df_scores = pd.DataFrame(cols)
    if times is not None:
        df_scores.insert(0, "_time", times.values)
    df_scores[target] = energy.values

    filename = f"pca_scores_{dataset_name}.csv" if dataset_name else "pca_scores.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_scores.to_csv(output_dir / filename, index=False)
    print(f"Saved scores table to {output_dir / filename}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(
    fig: plt.Figure, output_dir: Path, filename: str, dpi: int, show: bool
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"Saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def print_coverage_summary(
    pca: PCA,
    features: list[str],
    highlight: list[str],
    variance_threshold: float,
) -> None:
    """
    Print how well the highlighted feature subset spans the principal axes.

    For each of the top-k PCs (those together reaching variance_threshold),
    report the highest |loading| among the highlighted features vs. the global
    maximum loading. A large gap means a PC is poorly represented by the
    current feature subset.
    """
    ev = pca.explained_variance_ratio_
    cumev = np.cumsum(ev)
    n_thresh = int(np.searchsorted(cumev, variance_threshold)) + 1
    loadings = np.abs(pca.components_[:n_thresh])  # (n_thresh, n_features)

    feat_idx = {f: i for i, f in enumerate(features)}
    hi_idx = [feat_idx[f] for f in highlight if f in feat_idx]

    print(
        f"\n=== Coverage of highlighted features over top-{n_thresh} PCs "
        f"(cumulative ≥ {variance_threshold * 100:.0f}% variance) ==="
    )
    print(
        f"{'PC':<5} {'Var%':>6}  {'Max |load|':>11}  "
        f"{'Best highlighted':>18}  {'Best highlighted feat'}"
    )
    print("-" * 70)

    for i in range(n_thresh):
        row = loadings[i]
        global_max = float(row.max())
        if hi_idx:
            hi_max_idx = int(np.argmax(row[hi_idx]))
            hi_max_val = float(row[hi_idx[hi_max_idx]])
            hi_max_feat = highlight[hi_max_idx]
        else:
            hi_max_val, hi_max_feat = 0.0, "—"

        coverage_pct = 100 * hi_max_val / global_max if global_max > 0 else 0
        flag = "  ⚠" if coverage_pct < 50 else ""
        print(
            f"PC{i + 1:<3} {ev[i] * 100:6.1f}%  {global_max:11.3f}  "
            f"{hi_max_val:18.3f}  {hi_max_feat}{flag}"
        )

    print(
        "\nFeatures with ⚠ have low representation among the highlighted subset — "
        "consider adding a feature with high loading on that PC."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    features = args.features or DEFAULT_FEATURES
    highlight = args.highlight_features or DEFAULT_HIGHLIGHT

    df, available_features = load_and_prepare_data(
        data_path=args.data,
        features=features,
        target=args.target,
        aggregate=args.aggregate,
    )

    if args.filter_active:
        before = len(df)
        df = df[df[args.target] > 0].copy()
        print(f"Filtered to {args.target} > 0: {before} → {len(df)} rows")

    if len(df) < 3:
        raise ValueError("Not enough data rows to run PCA (need at least 3).")

    pca, scaler, scores = fit_pca(df, available_features, args.n_components)

    # Resolve highlight to available features only
    highlight_avail = [f for f in highlight if f in available_features]
    if len(highlight_avail) < len(highlight):
        skipped = [f for f in highlight if f not in available_features]
        print(f"Note: highlighted features not in data (skipped): {skipped}")

    output_dir = Path(args.output_dir)
    dataset_name = Path(args.data).stem
    energy = df[args.target].reset_index(drop=True)
    times = df["_time"].reset_index(drop=True) if "_time" in df.columns else None

    # --- Print summary ---
    print_coverage_summary(
        pca, available_features, highlight_avail, args.variance_threshold
    )

    # --- Plots ---
    plot_scree(
        pca=pca,
        variance_threshold=args.variance_threshold,
        output_dir=output_dir,
        dataset_name=dataset_name,
        dpi=args.dpi,
        show=args.show_plots,
    )

    if pca.n_components_ >= 2:
        plot_loading_biplot(
            pca=pca,
            features=available_features,
            highlight=highlight_avail,
            output_dir=output_dir,
            dataset_name=dataset_name,
            dpi=args.dpi,
            show=args.show_plots,
        )

        plot_workload_map(
            scores=scores,
            energy=energy,
            pca=pca,
            output_dir=output_dir,
            dataset_name=dataset_name,
            target=args.target,
            dpi=args.dpi,
            show=args.show_plots,
        )

    plot_loading_heatmap(
        pca=pca,
        features=available_features,
        highlight=highlight_avail,
        output_dir=output_dir,
        dataset_name=dataset_name,
        dpi=args.dpi,
        show=args.show_plots,
        n_components_shown=min(6, pca.n_components_),
    )

    if times is not None and pca.n_components_ >= 1:
        plot_temporal_drift(
            scores=scores,
            times=times,
            pca=pca,
            output_dir=output_dir,
            dataset_name=dataset_name,
            dpi=args.dpi,
            show=args.show_plots,
            n_pcs=min(3, pca.n_components_),
        )

    # --- CSV exports ---
    save_loadings_csv(pca, available_features, output_dir, dataset_name)
    save_scores_csv(scores, times, energy, output_dir, dataset_name, args.target)

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
