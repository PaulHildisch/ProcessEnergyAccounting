import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .linear.cvxpy_optimizer import train_cvxpy_model

FEATURES = [
    "delta_cpu_ns",
    "delta_io_bytes",
    "delta_net_send_bytes",
    "context_switches",
    "syscall_count",
    "delta_rss_memory",
    "delta_cpu_time_psutil",
    "delta_cpu_time_proc",
    "syscall_class_file",
    "syscall_class_network",
    "syscall_class_memory",
    "syscall_class_process",
    "syscall_class_other",
    "syscall_class_sched",
    "syscall_class_signal",
    "syscall_class_time",
]
TARGET = "interval_energy"
DATA_PATH = "../data/parallel_bench_replay_2.parquet"
MAX_LAG = 10

GOOD_FEATURES = [
    "delta_cpu_ns",
    "syscall_count",
    "syscall_class_file",
    "syscall_class_other",
]


def plot_interval_energy_and_metrics(
    df_interval: pd.DataFrame, features: list[str]
) -> None:
    plt.figure(figsize=(15, 5))
    plt.plot(
        df_interval["_time"], df_interval["interval_energy"], label="Interval Energy"
    )
    for metric in features:
        plt.plot(df_interval["_time"], df_interval[metric], label=metric, alpha=0.5)
    plt.legend()
    plt.title("Interval Energy and Summed Metrics Over Time")
    plt.show()


def plot_pearson_bar(pearson_corr: pd.DataFrame, target: str) -> None:
    plt.figure(figsize=(8, 3))
    pearson_corr_no_target = pearson_corr[target].drop(target)
    pearson_corr_no_target.sort_values(ascending=False).plot(
        kind="bar",
        color="#3477eb",
        edgecolor="black",
    )
    plt.ylabel("Pearson correlation", fontsize=13)
    plt.title("Pearson Correlation (Features vs. Interval Energy)", fontsize=14)
    plt.xticks(rotation=35, ha="right", fontsize=10)
    plt.tight_layout()
    plt.grid(axis="y", linestyle=":", alpha=0.4)
    plt.show()


def plot_spearman_bar(spearman_corr: pd.DataFrame, target: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    spearman_corr_no_target = (
        spearman_corr[target].drop(target).sort_values(ascending=False)
    )
    labels = spearman_corr_no_target.index.to_list()
    values = spearman_corr_no_target.values
    x = np.arange(len(labels))

    bars = ax.bar(
        x,
        values,
        color="#eb9834",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_ylabel(r"Correlation $\rho$", fontsize=14, labelpad=2)
    ax.set_xlabel(None)

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
            val + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title("Spearman Correlation", fontsize=13, pad=6)

    plt.tight_layout(pad=0.5)
    plt.savefig("spearman_correlation.pdf", bbox_inches="tight", dpi=400)
    plt.savefig("spearman_correlation.png", dpi=3000)
    plt.show()


def print_lag_correlations(
    df_interval: pd.DataFrame,
    features: list[str],
    target: str,
    max_lag: int,
    shift_sign: int,
) -> None:
    for lag in range(1, max_lag + 1):
        shifted = df_interval.copy()
        shifted["lagged_energy"] = shifted[target].shift(shift_sign * lag)
        corr = shifted[features + ["lagged_energy"]].corr()
        print(f"\n--- Lag {lag} ---")
        print(corr["lagged_energy"].sort_values(ascending=False).head(5))


def plot_prediction_results(df_pred: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(
        df_pred["_time"],
        df_pred["interval_energy"],
        label="Actual Energy",
        linewidth=4.5,
    )
    plt.plot(
        df_pred["_time"],
        df_pred["predicted_total_energy"],
        label="Predicted Energy",
        linestyle="--",
        linewidth=4.5,
    )
    plt.xlabel("Time")
    plt.ylabel("Interval Energy")
    plt.title("Actual vs Predicted Total Interval Energy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(
        df_pred["_time"],
        df_pred["interval_energy"] - df_pred["predicted_total_energy"],
        label="Error",
    )
    plt.axhline(0, color="gray", linestyle="--")
    plt.ylabel("Prediction Error")
    plt.xlabel("Time")
    plt.title("Prediction Error Over Time")
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")

    interval_energy = (
        df[["_time", "interval_energy"]]
        .dropna()
        .drop_duplicates("_time")
        .set_index("_time")["interval_energy"]
    )
    print(f"Number of intervals with energy: {len(interval_energy)}")

    df = df[df["_time"].isin(interval_energy.index)]
    print(f"Process-level rows after filtering: {len(df)}")

    print(f"Unique times in process data: {df['_time'].nunique()}")
    print(f"Unique times with energy: {interval_energy.index.nunique()}")

    active = df[df["interval_energy"] > 0]
    print(f"Intervall energy shape: {active.shape}")

    print(df.columns)
    print(df.shape)

    df_energy = df[["_time", "interval_energy"]].dropna().drop_duplicates("_time")
    df_avg_power = df[["_time", "avg_power"]].dropna().drop_duplicates("_time")

    df_interval = df.groupby("_time")[FEATURES].sum().reset_index()
    df_interval = df_interval.merge(df_energy, on="_time", how="left")
    df_interval = df_interval.merge(df_avg_power, on="_time", how="left")

    print("Sanity check of data:")
    print(df_interval["interval_energy"].describe())
    print(df_interval["interval_energy"].isna().mean())

    plot_interval_energy_and_metrics(df_interval, FEATURES)
    print(df_interval.shape)

    active = df_interval[df_interval[TARGET] > 0]
    print(f"Active shape: {active.shape}")

    print(active.describe())
    print(active[FEATURES + [TARGET]].corr()[TARGET])

    pearson_corr = df_interval[FEATURES + [TARGET]].corr(method="pearson")
    print(f"=== Pearson Correlation with {TARGET} ===")
    print(pearson_corr[TARGET].sort_values())

    spearman_corr = df_interval[FEATURES + [TARGET]].corr(method="spearman")
    print(f"\n=== Spearman Correlation with {TARGET} ===")
    print(spearman_corr[TARGET].sort_values())

    print("Unique values per feature in normal set:")
    print(df_interval[FEATURES].nunique().sort_values())

    print(f"\nDescribe of {TARGET} in normal set:")
    print(df_interval[TARGET].describe())

    plot_pearson_bar(pearson_corr, TARGET)
    plot_spearman_bar(spearman_corr, TARGET)

    print_lag_correlations(df_interval, FEATURES, TARGET, MAX_LAG, shift_sign=-1)
    print_lag_correlations(df_interval, FEATURES, TARGET, MAX_LAG, shift_sign=1)

    df["syscall_class_file"] = df["syscall_class_file"].fillna(0)
    df["syscall_class_other"] = df["syscall_class_other"].fillna(0)

    nan_report = df[GOOD_FEATURES].isna().sum()
    print("NaNs per feature:\n", nan_report[nan_report > 0])

    results = train_cvxpy_model(df, GOOD_FEATURES)

    weights = results["weights"]
    static_energy = results["static_energy"]

    print("Learned weights:", dict(zip(GOOD_FEATURES, weights)))
    print("Static energy component:", static_energy)

    df_scaled = df.copy()
    df_scaled[GOOD_FEATURES] = results["scaler"].transform(df_scaled[GOOD_FEATURES])
    df_scaled["predicted_process_energy"] = (
        df_scaled[GOOD_FEATURES].values @ results["weights"]
    )

    df_pred = df_scaled.groupby("_time")["predicted_process_energy"].sum().reset_index()
    df_pred = df_pred.merge(
        df[["_time", "interval_energy"]].drop_duplicates("_time"),
        on="_time",
    )
    df_pred["predicted_total_energy"] = (
        df_pred["predicted_process_energy"] + results["static_energy"]
    )

    r2 = r2_score(df_pred["interval_energy"], df_pred["predicted_total_energy"])
    mae = mean_absolute_error(
        df_pred["interval_energy"],
        df_pred["predicted_total_energy"],
    )
    print(f"R² (interval-level): {r2:.4f}")
    print(f"MAE (interval-level): {mae:.4f}")
    print(f"Static energy per interval: {results['static_energy']:.4f}")

    plot_prediction_results(df_pred)


if __name__ == "__main__":
    main()
