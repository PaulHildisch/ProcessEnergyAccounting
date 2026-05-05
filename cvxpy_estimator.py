import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

# from estimation.linear.cvxpy_optimizer import train_cvxpy_model as optimizer

# ==== CONFIGURATION ====
good_features = [
    "delta_instructions",
    "delta_cache_misses",
    "delta_branch_instructions",
    "syscall_class_other",
    # "delta_cycles",
    # "delta_cpu_ns",
    # "syscall_count",
    # "syscall_class_file",
]

target = "interval_energy"
l1_penalty = 0.1
static_penalty = 0.00

# ==== LOAD AND PREPARE DATA ====
df = pd.read_parquet("runs/benchmark-siena06-v6/process_interval_data.parquet")
df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")
df[good_features] = df[good_features].fillna(0)

# Build unique interval_energy source of truth
interval_energy_all = (
    df[["_time", "interval_energy"]]
    .dropna()
    .drop_duplicates("_time")
    .set_index("_time")["interval_energy"]
)
print(f"Number of intervals with energy: {len(interval_energy_all)}")

# Filter to rows matching those intervals only
df = df[df["_time"].isin(interval_energy_all.index)]
print(f"Process-level rows after filtering: {len(df)}")
print(f"Unique times in process data: {df['_time'].nunique()}")
print(f"Unique times with energy: {interval_energy_all.index.nunique()}")

# ==== TRAIN-TEST SPLIT ====
time_values = interval_energy_all.index.sort_values()
train_times, test_times = train_test_split(time_values, test_size=0.2, shuffle=False)

interval_energy_train = interval_energy_all.loc[train_times]
interval_energy_test = interval_energy_all.loc[test_times]

df_train = df[df["_time"].isin(train_times)].copy()
df_test = df[df["_time"].isin(test_times)].copy()


# ==== TRAINING FUNCTION ====
def train_cvxpy_model(
    df: pd.DataFrame,
    features: list,
    interval_energy: pd.Series,
    l1_penalty=1.0,
    static_penalty=0.0,
):
    df = df.copy()
    df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")
    df[features] = df[features].fillna(0)
    df = df[df["_time"].isin(interval_energy.index)]
    interval_energy = interval_energy.sort_index()

    # Pre-aggregate features to interval level (sum across all processes per interval).
    # Mathematically equivalent to building a dense (n_intervals × n_samples) matrix
    # but uses O(n_intervals) memory instead of O(n_intervals × n_samples).
    df_agg = df.groupby("_time")[features].sum()
    df_agg = df_agg.reindex(interval_energy.index).fillna(0)

    scaler = MaxAbsScaler()
    X = scaler.fit_transform(df_agg.values)

    # Variables
    w = cp.Variable(X.shape[1])
    s = cp.Variable()  # static component

    # Model: X @ w gives per-interval predicted variable energy
    interval_preds = X @ w + s
    loss = cp.sum_squares(interval_preds - interval_energy.values)
    reg = l1_penalty * cp.norm1(w) + static_penalty * cp.abs(s)
    prob = cp.Problem(cp.Minimize(loss + reg), constraints=[s >= 0, w >= 0])
    prob.solve()

    return {"weights": w.value, "static_energy": s.value, "scaler": scaler}


# ==== TRAIN ====
results = train_cvxpy_model(
    df_train, good_features, interval_energy_train, l1_penalty, static_penalty
)
# results = optimizer(df_train, good_features, interval_energy_train)
weights = results["weights"]
static_energy = results["static_energy"]
scaler = results["scaler"]

print("Learned weights:")
for f, w in zip(good_features, weights):
    print(f"  {f}: {w:.4e}")
print(f"Static energy component: {static_energy:.4f}")


# ==== PREDICT AND EVALUATE ====
def predict_per_interval(df, weights, scaler, good_features, static_energy):
    df_scaled = df.copy()
    df_scaled[good_features] = scaler.transform(df_scaled[good_features])
    df_scaled["predicted_process_energy"] = df_scaled[good_features].values @ weights
    pred = df_scaled.groupby("_time")["predicted_process_energy"].sum().reset_index()
    return pred


df_pred = predict_per_interval(df_test, weights, scaler, good_features, static_energy)
df_pred = df_pred.merge(
    interval_energy_test.rename("interval_energy"), left_on="_time", right_index=True
)
df_pred["predicted_total_energy"] = df_pred["predicted_process_energy"] + static_energy

r2 = r2_score(df_pred["interval_energy"], df_pred["predicted_total_energy"])
mae = mean_absolute_error(df_pred["interval_energy"], df_pred["predicted_total_energy"])
mean_energy = df_pred["interval_energy"].mean()
print(f"\nR² (interval-level): {r2:.4f}")
print(f"MAE (interval-level): {mae:.4f}")
print(f"Mean interval energy: {mean_energy:.4f}")
print(f"MAE (% of mean): {100 * mae / mean_energy:.2f}%")

start_time = df_pred["_time"].min()
end_time = df_pred["_time"].max()

mask = (df_pred["_time"] >= start_time) & (df_pred["_time"] <= end_time)


# ==== VISUALIZE ====
# Actual vs Predicted Total Interval Energy (IEEE-style)
fig, ax = plt.subplots(figsize=(7.2, 3.4))  # IEEE double-column width

ax.plot(
    df_pred.loc[mask, "_time"],
    df_pred.loc[mask, "interval_energy"],
    label="Actual Energy",
    linewidth=2.0,
)
ax.plot(
    df_pred.loc[mask, "_time"],
    df_pred.loc[mask, "predicted_total_energy"],
    label="Predicted Energy",
    linestyle="--",
    linewidth=2.0,
)

# Axis labels and ticks
ax.set_xlabel("Time", fontsize=12, labelpad=4)
ax.set_ylabel("Interval Energy (Wh)", fontsize=12, labelpad=4)
ax.tick_params(axis="both", labelsize=12)

# No title – use LaTeX \caption{} in the paper instead
# ax.set_title("Actual vs Predicted Total Interval Energy (after 2 min)", fontsize=12)

# Legend inside, similar style as stacked area plot
ax.legend(
    loc="upper right",
    bbox_to_anchor=(0.97, 0.97),
    fontsize=10.5,
    frameon=True,
    framealpha=0.9,
    handlelength=1.8,
    labelspacing=0.4,
)

ax.set_title("Actual vs. Predicted Interval energy", fontsize=13, pad=6)

plt.tight_layout(pad=0.5)
plt.savefig("actual_vs_predicted_interval_energy.pdf", bbox_inches="tight")
plt.savefig("actual_vs_predicted_interval_energy.png", bbox_inches="tight", dpi=3000)
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

plt.figure(figsize=(10, 4))
plt.hist(df_pred["interval_energy"] - df_pred["predicted_total_energy"], bins=40)
plt.title("Histogram of Prediction Errors")
plt.xlabel("Error")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 1. Estimate per-process energy for each row in df_test
df_test_plot = df_test.copy()
df_test_plot[good_features] = scaler.transform(df_test_plot[good_features])
df_test_plot["estimated_process_energy"] = df_test_plot[good_features].values @ weights

# 2. Group by process base name (merge all instances of the same process, e.g. wrk_99, wrk_246 -> wrk)
import re

df_test_plot["base_name"] = (
    df_test_plot["process_name"].str.replace(r"_\d+$", "", regex=True).str.strip()
)
df_test_plot.loc[df_test_plot["base_name"] == "", "base_name"] = "unknown"

# 3. Sum energy per interval per base process name
agg = (
    df_test_plot.groupby(["_time", "base_name"])["estimated_process_energy"]
    .sum()
    .reset_index()
)

# Pivot to intervals × process_name matrix
pivot = agg.pivot(
    index="_time", columns="base_name", values="estimated_process_energy"
).fillna(0)

# 4. Select top N processes by peak contribution (so processes active in any window are represented)
N = 8
top_processes = pivot.max().sort_values(ascending=False).head(N).index
pivot_top = pivot[top_processes].copy()

# Add "Other" for the rest
if len(pivot.columns) > N:
    pivot_top["Other"] = pivot.drop(columns=top_processes).sum(axis=1)

print((pivot_top < 0).sum())
print("Fraction of intervals with negative values per process:")
print((pivot_top < 0).mean())

# 5. Plot stacked area plot (IEEE two-column, improved readability)
pivot_top_clipped = pivot_top.clip(lower=0)

pivot_mask = (pivot_top_clipped.index >= start_time) & (
    pivot_top_clipped.index <= end_time
)


fig, ax = plt.subplots(figsize=(7.2, 3.4))  # IEEE double-column width

# Plot stacked areas (no legend labels yet)
pivot_top_clipped.loc[pivot_mask].plot.area(ax=ax, alpha=0.8, linewidth=0, legend=False)

# Axis labels and ticks
ax.set_xlabel("Time", fontsize=12, labelpad=4)
ax.set_ylabel("Estimated Process Energy (Wh)", fontsize=12, labelpad=4)
ax.tick_params(axis="both", labelsize=12)

other_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][
    len(pivot_top_clipped.columns) % 10 - 1
]

from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle

# --- Custom legend (Top 8 colors + 'Other') ---
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
top_colors = colors[:8]  # colors for the top eight processes


class HandlerMultiColor(HandlerBase):
    def create_artists(
        self, legend, orig_handle, x0, y0, width, height, fontsize, trans
    ):
        """Create multiple colored rectangles for one legend entry."""
        colors = getattr(orig_handle, "_legend_colors", [other_color])
        n = len(colors)
        artists = []
        for i, c in enumerate(colors):
            rect = Rectangle(
                (x0 + i * width / n, y0),
                width / n,
                height,
                facecolor=c,
                transform=trans,
                lw=0,
            )
            artists.append(rect)
        return artists


# Define dummy patch handles that only carry color info
top_handle = Rectangle((0, 0), 1, 1)
top_handle._legend_colors = top_colors  # 8 colors

other_handle = Rectangle((0, 0), 1, 1)
other_handle._legend_colors = [other_color]  # single color

handles = [top_handle, other_handle]
labels = ["Top eight processes", "Other"]

ax.legend(
    handles=handles,
    labels=labels,
    handler_map={top_handle: HandlerMultiColor(), other_handle: HandlerMultiColor()},
    loc="upper right",
    bbox_to_anchor=(0.97, 0.97),
    fontsize=10.5,
    frameon=True,
    framealpha=0.9,
    ncol=1,
    handlelength=2.5,
    labelspacing=0.5,
)

ax.set_title("Per-Process Energy Contribution Over Time", fontsize=13, pad=6)

plt.tight_layout(pad=0.5)
plt.savefig("per_process_energy_contribution.pdf", bbox_inches="tight")
plt.savefig("per_process_energy_contribution.png", bbox_inches="tight", dpi=3000)
plt.show()

# ==== ADDITIONAL PLOT: Top 10 PIDs by peak, with process names, no merging ====
N2 = 20

# Build pid+name label for each unique PID
df_test_plot["pid_label"] = (
    df_test_plot["process_name"] + " (" + df_test_plot["pid"].astype(str) + ")"
)
agg2 = (
    df_test_plot.groupby(["_time", "pid", "pid_label"])["estimated_process_energy"]
    .sum()
    .reset_index()
)
pivot2 = agg2.pivot(
    index="_time", columns="pid_label", values="estimated_process_energy"
).fillna(0)

top_pids = pivot2.max().sort_values(ascending=False).head(N2).index
pivot2_top = pivot2[top_pids].copy()
pivot2_top["Other"] = pivot2.drop(columns=top_pids).sum(axis=1)
pivot2_top_clipped = pivot2_top.clip(lower=0)

fig, ax = plt.subplots(figsize=(12, 5))
pivot2_top_clipped.loc[pivot_mask].plot.area(ax=ax, alpha=0.8, linewidth=0)
ax.set_xlabel("Time", fontsize=12, labelpad=4)
ax.set_ylabel("Estimated Process Energy (Wh)", fontsize=12, labelpad=4)
ax.set_title(
    "Per-PID Energy Contribution Over Time (Top 10 by Peak)", fontsize=13, pad=6
)
ax.tick_params(axis="both", labelsize=11)
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9, frameon=True)
plt.tight_layout()
plt.savefig("per_pid_energy_contribution.png", bbox_inches="tight", dpi=200)
plt.show()

# ==== SAVE MODEL ====
import os
import pickle

os.makedirs("estimation/models", exist_ok=True)
model = {
    "weights": weights,
    "static_energy": static_energy,
    "scaler": scaler,
    "features": good_features,
}
model_path = "estimation/models/model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {model_path}")
