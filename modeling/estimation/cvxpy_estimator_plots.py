import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score


def _plot_actual_vs_predicted(
    evaluation_df: pd.DataFrame, time_column: str, target_column: str
):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(
        evaluation_df[time_column],
        evaluation_df[target_column],
        label="Actual",
        linewidth=1.0,
        alpha=0.8,
    )
    ax.plot(
        evaluation_df[time_column],
        evaluation_df["predicted_total_energy"],
        label="Predicted",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Interval Energy (J)")
    ax.legend(loc="upper right")
    ax.set_title("Actual vs Predicted Interval Energy")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def _plot_error_over_time(
    evaluation_df: pd.DataFrame, time_column: str, target_column: str
):
    """
    Prediction error over time. The faint line shows raw per-interval noise;
    the bold rolling mean and ±1σ band reveal any systematic temporal drift.
    """
    error = evaluation_df[target_column] - evaluation_df["predicted_total_energy"]
    times = evaluation_df[time_column]
    window = max(3, len(error) // 10)

    rolling_mean = error.rolling(window, center=True, min_periods=1).mean()
    rolling_std = error.rolling(window, center=True, min_periods=1).std().fillna(0)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(times, error, linewidth=0.6, alpha=0.2, color="steelblue", label="Error")
    ax.fill_between(
        times,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        alpha=0.25,
        label=f"±1σ (w={window})",
    )
    ax.plot(
        times,
        rolling_mean,
        linewidth=2.5,
        zorder=5,
        label=f"Rolling mean (w={window})",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Prediction Error (J)")
    ax.set_title("Prediction Error Over Time")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def _plot_error_histogram(evaluation_df: pd.DataFrame, target_column: str):
    """
    Distribution of per-interval prediction errors.
    Overlaid normal fit and ±1σ lines show how Gaussian the residuals are.
    """
    error = (
        evaluation_df[target_column] - evaluation_df["predicted_total_energy"]
    ).values
    mean, std = error.mean(), error.std()

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    n, bins, _ = ax.hist(
        error, bins=50, density=True, edgecolor="white", linewidth=0.4, alpha=0.7
    )

    # Normal fit overlay
    x = np.linspace(bins[0], bins[-1], 300)
    ax.plot(x, stats.norm.pdf(x, mean, std), linewidth=2.0, label="Normal fit")

    # Mean and ±1σ reference lines
    for val, ls, lbl in [
        (mean, "-", f"Mean = {mean:.1f} J"),
        (mean - std, "--", f"−1σ = {mean - std:.1f} J"),
        (mean + std, "--", f"+1σ = {mean + std:.1f} J"),
    ]:
        ax.axvline(val, linestyle=ls, linewidth=1.2, alpha=0.8, label=lbl)

    ax.set_xlabel("Prediction Error (J)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Prediction Errors")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_feature_weights(weights, features: list[str]):
    """
    Learned weights in the MaxAbsScaled input space — bars are directly comparable
    across features. A larger bar = more energy attributed to that counter.
    """
    sorted_pairs = sorted(zip(features, weights), key=lambda x: x[1])
    feats, wts = zip(*sorted_pairs) if sorted_pairs else ([], [])

    height = max(3.0, 0.55 * len(feats))
    fig, ax = plt.subplots(figsize=(7.2, height))
    colors = ["#2196F3" if w >= 0 else "#F44336" for w in wts]
    bars = ax.barh(feats, wts, color=colors)

    # Value labels at end of each bar
    x_pad = max(wts) * 0.02 if wts else 0
    for bar, val in zip(bars, wts):
        ax.text(
            val + x_pad if val >= 0 else val - x_pad,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2e}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Learned Weight (scaled input space)")
    ax.set_title("Feature Weights")
    # Give value labels room
    ax.set_xlim(right=max(wts) * 1.25 if wts else 1)
    fig.tight_layout()
    return fig


def _plot_scatter_actual_vs_predicted(evaluation_df: pd.DataFrame, target_column: str):
    """
    Actual vs predicted scatter. Colour encodes absolute error clipped at the
    95th percentile so the colour scale isn't dominated by outliers.
    """
    actual = evaluation_df[target_column].values
    predicted = evaluation_df["predicted_total_energy"].values
    abs_error = np.abs(actual - predicted)
    vmax = float(np.percentile(abs_error, 95))

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    sc = ax.scatter(
        actual,
        predicted,
        c=abs_error,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        alpha=0.6,
        s=15,
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Absolute Error (J)\n[clipped at 95th pct]", fontsize=8)

    lim_min = min(actual.min(), predicted.min())
    lim_max = max(actual.max(), predicted.max())
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        "r--",
        linewidth=1.5,
        label="Perfect fit",
    )

    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    ax.text(
        0.05,
        0.95,
        f"R² = {r2:.3f}\nMAE = {mae:.1f} J",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=8,
        bbox=dict(boxstyle="round", alpha=0.3),
    )
    ax.set_xlabel("Actual Energy (J)")
    ax.set_ylabel("Predicted Energy (J)")
    ax.set_title("Actual vs Predicted Scatter")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def _plot_residuals_vs_fitted(evaluation_df: pd.DataFrame, target_column: str):
    """
    Residuals vs fitted values, sorted by fitted value.
    The rolling mean (orange) exposes load-dependent bias; Spearman ρ between
    |residual| and fitted value quantifies heteroscedasticity.
    """
    predicted = evaluation_df["predicted_total_energy"].values
    residuals = (
        evaluation_df[target_column] - evaluation_df["predicted_total_energy"]
    ).values

    sorted_idx = np.argsort(predicted)
    pred_sorted = predicted[sorted_idx]
    resid_sorted = pd.Series(residuals[sorted_idx])

    window = max(3, len(pred_sorted) // 10)
    rolling_mean = resid_sorted.rolling(window, center=True, min_periods=1).mean()

    rho, pval = stats.spearmanr(np.abs(residuals), predicted)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.scatter(pred_sorted, resid_sorted, alpha=0.3, s=12, label="Residual")
    ax.plot(
        pred_sorted,
        rolling_mean,
        color="orange",
        linewidth=2.5,
        zorder=5,
        label=f"Rolling mean (w={window})",
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2)

    ax.text(
        0.02,
        0.97,
        f"Spearman ρ(|residual|, fitted) = {rho:.3f}  (p={pval:.2e})\n"
        "ρ ≈ 0: homoscedastic  |  |ρ| > 0.3: load-dependent error",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round", alpha=0.2),
    )
    ax.set_xlabel("Fitted Values (J)")
    ax.set_ylabel("Residuals (J)")
    ax.set_title("Residuals vs Fitted")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def _plot_cumulative_error(
    evaluation_df: pd.DataFrame, time_column: str, target_column: str
):
    """
    Cumulative prediction error over time.
    A flat line near zero means the model is globally unbiased;
    a monotone drift indicates systematic over- or under-prediction that
    compounds across the test window — critical for energy accounting.
    """
    actual = evaluation_df[target_column]
    predicted = evaluation_df["predicted_total_energy"]
    error = actual - predicted
    cumulative_abs = error.cumsum()
    cumulative_pct = 100.0 * cumulative_abs / actual.cumsum().replace(0, np.nan)

    final_abs = float(cumulative_abs.iloc[-1])
    final_pct = float(cumulative_pct.iloc[-1])

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)

    axes[0].plot(evaluation_df[time_column], cumulative_abs, linewidth=1.5)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Cumulative Error (J)")
    axes[0].set_title("Cumulative Prediction Error Over Time")
    axes[0].annotate(
        f"Final: {final_abs:+.1f} J",
        xy=(evaluation_df[time_column].iloc[-1], final_abs),
        xytext=(-60, 10),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )

    axes[1].plot(
        evaluation_df[time_column], cumulative_pct, linewidth=1.5, color="orange"
    )
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Cumulative Error (% of actual)")
    axes[1].set_xlabel("Time")
    axes[1].annotate(
        f"Final: {final_pct:+.2f}%",
        xy=(evaluation_df[time_column].iloc[-1], final_pct),
        xytext=(-60, 10),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def _plot_error_cdf(evaluation_df: pd.DataFrame, target_column: str):
    """
    CDF of absolute percentage error (APE) per interval.
    Read off: "X% of intervals are predicted within Y% of actual energy."
    X-axis is clipped at the 99th percentile so the main body isn't squished
    by rare large-error outliers.
    """
    actual = evaluation_df[target_column].replace(0, np.nan)
    ape = (
        (evaluation_df[target_column] - evaluation_df["predicted_total_energy"]).abs()
        / actual
        * 100
    ).dropna()

    sorted_ape = np.sort(ape.values)
    cdf = np.arange(1, len(sorted_ape) + 1) / len(sorted_ape)
    x_max = float(np.percentile(sorted_ape, 99))

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(sorted_ape, cdf * 100, linewidth=2.0)

    for threshold, ls in [(5, "--"), (10, ":")]:
        coverage = float(np.interp(threshold, sorted_ape, cdf) * 100)
        ax.axvline(threshold, color="gray", linestyle=ls, linewidth=1.0)
        ax.axhline(coverage, color="gray", linestyle=ls, linewidth=1.0)
        ax.annotate(
            f"{coverage:.1f}% within {threshold}%",
            xy=(threshold, coverage),
            xytext=(threshold + x_max * 0.04, coverage - 8),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )

    median_ape = float(np.interp(0.5, cdf, sorted_ape))
    ax.axvline(
        median_ape,
        color="C1",
        linestyle="-.",
        linewidth=1.2,
        label=f"Median APE = {median_ape:.1f}%",
    )

    ax.set_xlabel("Absolute Percentage Error (%)")
    ax.set_ylabel("Cumulative Intervals (%)")
    ax.set_title("CDF of Absolute Percentage Error")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def _plot_learning_curve(learning_curve_data: list[dict]):
    """
    MAE and R² on the fixed test set as training set size grows (time-ordered).
    Three panels (MAE in J, MAE as % of mean energy, R²) avoid dual-axis
    confusion and show each metric on its own natural scale.
    """
    if not learning_curve_data:
        return None

    sizes = [d["train_size"] for d in learning_curve_data]
    maes = [d["mae"] for d in learning_curve_data]
    mae_pcts = [d["mae_pct"] for d in learning_curve_data]
    r2s = [d["r2"] for d in learning_curve_data]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.5), sharex=True)
    fig.suptitle("Learning Curve (test set fixed)", fontsize=11, y=1.01)

    panels = [
        (axes[0], maes, "MAE (J)", "C0"),
        (axes[1], mae_pcts, "MAE (% of mean)", "C1"),
        (axes[2], r2s, "R²", "C2"),
    ]
    for ax, vals, ylabel, color in panels:
        ax.plot(sizes, vals, marker="o", linewidth=2.0, color=color)
        # Value label at each point
        for x, y in zip(sizes, vals):
            ax.annotate(
                f"{y:.2f}",
                xy=(x, y),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # R² panel: show full range including any negative values
    r2_min = min(r2s)
    axes[2].set_ylim(r2_min - abs(r2_min) * 0.1 - 0.05, 1.05)
    axes[2].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel("Training Intervals")

    fig.tight_layout()
    return fig


def _plot_cv_heatmap(cv_results: list[dict]):
    """
    Cell-annotated heatmap of cross-validated MAE over the
    (l1_penalty × static_penalty) grid.  Green = low error = good.
    The red star marks the best combination found by the search.
    """
    l1_vals = sorted({r["l1_penalty"] for r in cv_results})
    sp_vals = sorted({r["static_penalty"] for r in cv_results})

    matrix = np.full((len(l1_vals), len(sp_vals)), np.nan)
    for r in cv_results:
        i = l1_vals.index(r["l1_penalty"])
        j = sp_vals.index(r["static_penalty"])
        matrix[i, j] = r["mean_mae"]

    best_flat = int(np.nanargmin(matrix))
    best_i, best_j = np.unravel_index(best_flat, matrix.shape)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    im = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("CV MAE (J)")

    # Cell annotations
    vmid = float(np.nanmean(matrix))
    for i in range(len(l1_vals)):
        for j in range(len(sp_vals)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if val > vmid else "black",
                )

    # Best combination marker
    ax.plot(
        best_j,
        best_i,
        "r*",
        markersize=14,
        label=f"Best: {matrix[best_i, best_j]:.1f} J  "
        f"(l1={l1_vals[best_i]:.4g}, static={sp_vals[best_j]:.4g})",
    )

    ax.set_xticks(range(len(sp_vals)))
    ax.set_xticklabels([f"{v:.4g}" for v in sp_vals], rotation=30, ha="right")
    ax.set_yticks(range(len(l1_vals)))
    ax.set_yticklabels([f"{v:.4g}" for v in l1_vals])
    ax.set_xlabel("static_penalty")
    ax.set_ylabel("l1_penalty")
    ax.set_title("Hyperparameter Search: CV MAE (J)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _save_figure(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    return path


def _sanitize_path_part(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = normalized.strip("-._")
    return normalized or "unknown"


def _scoped_output_dir(
    output_dir: Path,
    hostname: str | list[str] | tuple[str, ...] | None,
    workload_name: str | None,
) -> Path:
    scoped_dir = output_dir
    if hostname:
        if isinstance(hostname, str):
            host_part = _sanitize_path_part(hostname)
        else:
            host_part = "_".join(
                _sanitize_path_part(host) for host in dict.fromkeys(hostname)
            )
        scoped_dir = scoped_dir / host_part
    if workload_name:
        scoped_dir = scoped_dir / _sanitize_path_part(workload_name)
    return scoped_dir


def save_estimator_plots(
    evaluation_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    time_column: str,
    target_column: str,
    model_weights=None,
    features: list[str] | None = None,
    learning_curve_data: list[dict] | None = None,
    cv_results: list[dict] | None = None,
    hostname: str | list[str] | tuple[str, ...] | None = None,
    workload_name: str | None = None,
) -> list[Path]:
    output_dir = _scoped_output_dir(output_dir, hostname, workload_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures: list[tuple[plt.Figure, str]] = []

    figures.append(
        (
            _plot_actual_vs_predicted(evaluation_df, time_column, target_column),
            "actual_vs_predicted",
        )
    )
    figures.append(
        (
            _plot_error_over_time(evaluation_df, time_column, target_column),
            "error_over_time",
        )
    )
    figures.append(
        (_plot_error_histogram(evaluation_df, target_column), "error_histogram")
    )
    figures.append(
        (
            _plot_scatter_actual_vs_predicted(evaluation_df, target_column),
            "scatter_actual_vs_predicted",
        )
    )
    figures.append(
        (_plot_residuals_vs_fitted(evaluation_df, target_column), "residuals_vs_fitted")
    )
    figures.append(
        (
            _plot_cumulative_error(evaluation_df, time_column, target_column),
            "cumulative_error",
        )
    )
    figures.append((_plot_error_cdf(evaluation_df, target_column), "error_cdf"))

    if model_weights is not None and features is not None:
        figures.append(
            (_plot_feature_weights(model_weights, features), "feature_weights")
        )

    lc_fig = _plot_learning_curve(learning_curve_data or [])
    if lc_fig is not None:
        figures.append((lc_fig, "learning_curve"))

    if cv_results:
        figures.append((_plot_cv_heatmap(cv_results), "cv_heatmap"))

    saved_paths: list[Path] = []
    try:
        for fig, name in figures:
            saved_paths.append(_save_figure(fig, output_dir / f"{prefix}_{name}.png"))
    finally:
        for fig, _ in figures:
            plt.close(fig)

    return saved_paths
