#!/usr/bin/env python3
"""Simple PCA analysis for feature selection.

Outputs:
- scree plot: decide how many principal components to keep
- loading heatmap: identify original features with the largest loadings per PC
- loadings CSV: full PCA loading matrix
- selected-features CSV: union of top-loading features across selected PCs
"""

import argparse
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    parser = argparse.ArgumentParser(
        description="Simple PCA analysis for feature selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to the parquet data file")
    parser.add_argument(
        "--target",
        default="interval_energy",
        help="Target column used for filtering valid rows",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Feature columns to include. Defaults to all known collected features.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Number of PCs to fit. Defaults to min(n_samples, n_features).",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.90,
        help="Cumulative variance threshold used to choose PCs for the heatmap/CSV.",
    )
    parser.add_argument(
        "--top-features-per-pc",
        type=int,
        default=5,
        help="How many highest-loading features to show/select per principal component.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to save plots and CSV outputs.",
    )
    parser.add_argument(
        "--hostname",
        default=None,
        help="Optional hostname subdirectory under output-dir.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved PNG plots.",
    )
    parser.add_argument(
        "--filter-active",
        action="store_true",
        help="Keep only rows where target > 0.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate process-level rows by _time before PCA.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively after saving.",
    )
    return parser.parse_args()


def load_data(
    data_path: str,
    features: list[str],
    target: str,
    aggregate: bool,
    filter_active: bool,
) -> tuple[pd.DataFrame, list[str]]:
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    if "_time" in df.columns:
        df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data.")

    available_features = [feature for feature in features if feature in df.columns]
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        print(f"Skipping missing features: {missing_features}")
    if not available_features:
        raise ValueError(
            "None of the requested feature columns were found in the data."
        )

    if aggregate and "_time" in df.columns:
        energy = df.loc[:, ["_time", target]].dropna().drop_duplicates(subset=["_time"])
        feature_sums = (
            df.loc[:, ["_time", *available_features]]
            .groupby("_time", as_index=False)
            .sum()
        )
        df = pd.merge(feature_sums, energy, on="_time", how="left")
    else:
        columns = available_features + [target]
        if "_time" in df.columns:
            columns.append("_time")
        df = df.loc[:, columns].copy()

    df.loc[:, available_features] = df.loc[:, available_features].fillna(0)
    df = df.dropna(subset=[target])

    if filter_active:
        before = len(df)
        df = df[df[target] > 0].copy()
        print(f"Filtered to {target} > 0: {before} -> {len(df)} rows")

    if len(df) < 3:
        raise ValueError("Not enough rows to run PCA; need at least 3 rows.")

    print(f"Using {len(available_features)} features and {len(df)} rows.")
    return cast(pd.DataFrame, df), available_features


def fit_pca(
    df: pd.DataFrame,
    features: list[str],
    n_components: int | None,
) -> PCA:
    X = df[features].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    max_components = min(X_scaled.shape[0], X_scaled.shape[1])
    requested_components = n_components or max_components
    if requested_components > max_components:
        print(
            f"Requested {requested_components} PCs, but only {max_components} are possible; "
            f"using {max_components}."
        )
    n_components = min(requested_components, max_components)

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    print("\nExplained variance:")
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for idx, (variance, cumulative_variance) in enumerate(
        zip(pca.explained_variance_ratio_, cumulative),
        start=1,
    ):
        print(
            f"  PC{idx:>2}: {variance * 100:5.1f}% "
            f"(cumulative {cumulative_variance * 100:5.1f}%)"
        )

    return pca


def n_components_for_threshold(pca: PCA, variance_threshold: float) -> int:
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return min(
        int(np.searchsorted(cumulative, variance_threshold)) + 1, pca.n_components_
    )


def plot_scree(
    pca: PCA,
    variance_threshold: float,
    output_dir: Path,
    dataset_name: str,
    dpi: int,
    show: bool,
) -> None:
    explained = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)
    x = np.arange(1, len(explained) + 1)
    threshold_pc = n_components_for_threshold(pca, variance_threshold)
    threshold_pct = variance_threshold * 100

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.bar(x, explained, color="#3477eb", edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_xticks(x)
    ax1.grid(axis="y", linestyle=":", alpha=0.35)
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color="#eb9834", marker="o", linewidth=2)
    ax2.axhline(threshold_pct, color="#eb3434", linestyle="--", linewidth=1.3)
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    ax1.set_title(
        f"PCA scree plot: {threshold_pc} PCs explain {threshold_pct:.0f}% of variance"
    )
    ax2.text(
        len(explained) + 0.25,
        threshold_pct,
        f"{threshold_pct:.0f}% at PC{threshold_pc}",
        color="#eb3434",
        va="bottom",
        ha="left",
        fontsize=9,
    )

    save_figure(fig, output_dir, f"pca_scree_{dataset_name}.png", dpi, show)


def plot_loading_heatmap(
    pca: PCA,
    features: list[str],
    output_dir: Path,
    dataset_name: str,
    n_components: int,
    top_features_per_pc: int,
    dpi: int,
    show: bool,
) -> list[str]:
    n_components = min(n_components, pca.n_components_)
    top_features_per_pc = min(top_features_per_pc, len(features))
    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        index=features,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )

    selected_features = select_top_loading_features(
        loadings=loadings,
        top_features_per_pc=top_features_per_pc,
    )
    heatmap = loadings.loc[selected_features]

    fig_width = max(10, n_components * 0.45)
    fig_height = max(6, len(selected_features) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(heatmap.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Top-loading original features")
    ax.set_title(
        f"Top {top_features_per_pc} loading features per PC (PC1-PC{n_components})"
    )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("PCA loading")

    plt.tight_layout()
    save_figure(fig, output_dir, f"pca_loading_heatmap_{dataset_name}.png", dpi, show)
    return selected_features


def top_loading_features_by_pc(
    loadings: pd.DataFrame,
    top_features_per_pc: int,
) -> dict[str, list[str]]:
    if top_features_per_pc < 1:
        raise ValueError("--top-features-per-pc must be at least 1.")

    top_count = min(top_features_per_pc, len(loadings.index))
    feature_names = [str(feature) for feature in loadings.index.to_list()]
    top_by_pc: dict[str, list[str]] = {}

    for pc in [str(column) for column in loadings.columns.to_list()]:
        abs_values = loadings.loc[:, pc].abs().to_numpy(dtype=float)
        top_indices = np.argsort(abs_values)[::-1][:top_count]
        top_by_pc[pc] = [feature_names[int(index)] for index in top_indices]

    return top_by_pc


def select_top_loading_features(
    loadings: pd.DataFrame,
    top_features_per_pc: int,
) -> list[str]:
    selected: list[str] = []
    for top_features in top_loading_features_by_pc(
        loadings, top_features_per_pc
    ).values():
        for feature in top_features:
            if feature not in selected:
                selected.append(feature)
    return selected


def save_loadings_csv(
    pca: PCA,
    features: list[str],
    output_dir: Path,
    dataset_name: str,
) -> None:
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )
    loadings.index.name = "feature"
    abs_loadings = loadings.abs()
    loadings.insert(0, "dominant_pc", abs_loadings.idxmax(axis=1))
    loadings.insert(1, "max_abs_loading", abs_loadings.max(axis=1))

    output_path = output_dir / f"pca_loadings_{dataset_name}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    loadings.to_csv(output_path)
    print(f"Saved {output_path}")


def save_selected_features_csv(
    pca: PCA,
    features: list[str],
    output_dir: Path,
    dataset_name: str,
    n_components: int,
    top_features_per_pc: int,
) -> None:
    n_components = min(n_components, pca.n_components_)
    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        index=features,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    top_by_pc = top_loading_features_by_pc(loadings, top_features_per_pc)
    selected_features = select_top_loading_features(loadings, top_features_per_pc)

    rows = []
    for feature in selected_features:
        feature_loadings = loadings.loc[feature]
        loading_values = feature_loadings.to_numpy(dtype=float)
        best_pc_index = int(np.argmax(np.abs(loading_values)))
        best_pc = str(loadings.columns[best_pc_index])
        loading = float(loading_values[best_pc_index])
        rows.append(
            {
                "feature": feature,
                "best_pc": best_pc,
                "best_pc_index": best_pc_index + 1,
                "loading": loading,
                "abs_loading": abs(loading),
                "selected_in_pcs": ",".join(
                    pc
                    for pc, pc_features in top_by_pc.items()
                    if feature in pc_features
                ),
            }
        )

    selected = pd.DataFrame(rows).sort_values(
        ["best_pc_index", "abs_loading"], ascending=[True, False]
    )
    output_path = output_dir / f"pca_selected_features_{dataset_name}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = selected.drop(columns=["best_pc_index"])
    selected.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
    print(f"Selected {len(selected)} PCA feature candidates:")
    print(" ".join(selected["feature"].tolist()))


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    filename: str,
    dpi: int,
    show: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    print(f"Saved {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def output_directory(base_dir: str, hostname: str | None) -> Path:
    output_dir = Path(base_dir)
    if hostname:
        output_dir = output_dir / hostname
    return output_dir


def main() -> None:
    args = parse_args()
    features = args.features or DEFAULT_FEATURES
    if not 0 < args.variance_threshold <= 1:
        raise ValueError("--variance-threshold must be in the interval (0, 1].")
    if args.top_features_per_pc < 1:
        raise ValueError("--top-features-per-pc must be at least 1.")

    output_dir = output_directory(args.output_dir, args.hostname)
    dataset_name = Path(args.data).stem

    df, available_features = load_data(
        data_path=args.data,
        features=features,
        target=args.target,
        aggregate=args.aggregate,
        filter_active=args.filter_active,
    )
    pca = fit_pca(df, available_features, args.n_components)
    threshold_components = n_components_for_threshold(pca, args.variance_threshold)

    print(
        f"\nUsing PC1-PC{threshold_components} for heatmap/feature candidates "
        f"({args.variance_threshold * 100:.0f}% variance threshold)."
    )

    plot_scree(
        pca=pca,
        variance_threshold=args.variance_threshold,
        output_dir=output_dir,
        dataset_name=dataset_name,
        dpi=args.dpi,
        show=args.show_plots,
    )
    plot_loading_heatmap(
        pca=pca,
        features=available_features,
        output_dir=output_dir,
        dataset_name=dataset_name,
        n_components=threshold_components,
        top_features_per_pc=args.top_features_per_pc,
        dpi=args.dpi,
        show=args.show_plots,
    )
    save_loadings_csv(pca, available_features, output_dir, dataset_name)
    save_selected_features_csv(
        pca=pca,
        features=available_features,
        output_dir=output_dir,
        dataset_name=dataset_name,
        n_components=threshold_components,
        top_features_per_pc=args.top_features_per_pc,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
