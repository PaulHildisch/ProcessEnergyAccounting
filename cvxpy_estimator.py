import os
import pickle
import re

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from model_testing.clean_impl.preprocessing import Preprocessor

# from estimation.linear.cvxpy_optimizer import train_cvxpy_model as optimizer

# ==== CONFIGURATION ====
# good_features = [
#     "delta_instructions",
#     "delta_cache_misses",
#     "delta_branch_instructions",
#     "syscall_class_other",
#     # "delta_cycles",
#     # "delta_cpu_ns",
#     # "syscall_count",
#     # "syscall_class_file",
# ]

target = "interval_energy"
l1_penalty = 0.1
static_penalty = 0.00


# ==== LOAD AND PREPARE DATA ====


good_features =  ['delta_io_bytes', 'context_switches', 'delta_cpu_ns', 'delta_net_send_bytes', 'syscall_count']
train_ampliseq = [
        pd.read_parquet("runs/nfcore-20260703T215123Z/datasets/ampliseq_1_0607.parquet"),
        pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet"),
        #pd.read_parquet("runs/nfcore-20260708T125031Z/datasets/ampliseq_triple_run.parquet")

]

test_ampliseq = pd.read_parquet("runs/nfcore-20260706T112716Z/datasets/ampliseq_3_0707.parquet")

training_data = pd.concat(train_ampliseq, ignore_index=True)
training_data = training_data
test_data = test_ampliseq

preprocessor_train = Preprocessor(training_data, good_features, target=target)
X_train_agg, interval_energy_train, t_train, unagg_train = preprocessor_train.preprocess_no_split()
df_train = unagg_train.reset_index() # Bring _time back as a column for later functions

# Preprocess Test Data
preprocessor_test = Preprocessor(test_data, good_features, target=target)
X_test_agg, interval_energy_test, t_test, unagg_test = preprocessor_test.preprocess_no_split()
df_test = unagg_test.reset_index() # Bring _time back as a column for later functions



#---------------------------------------------------------------
# df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")
# df[good_features] = df[good_features].fillna(0)

# # Use one interval_energy value per timestamp.
# interval_energy_all = (
#     df[["_time", "interval_energy"]]
#     .dropna()
#     .drop_duplicates("_time")
#     .set_index("_time")["interval_energy"]
# )
# print(f"Number of intervals with energy: {len(interval_energy_all)}")

# # Keep only rows that belong to intervals with measured energy.
# df = df[df["_time"].isin(interval_energy_all.index)]
# print(f"Process-level rows after filtering: {len(df)}")
# print(f"Unique times in process data: {df['_time'].nunique()}")
# print(f"Unique times with energy: {interval_energy_all.index.nunique()}")


# # # ==== TRAIN / TEST SPLIT ====
# time_values = interval_energy_all.index.sort_values()
# train_times, test_times = train_test_split(time_values, test_size=0.2, shuffle=False)

# interval_energy_train = interval_energy_all.loc[train_times]
# interval_energy_test = interval_energy_all.loc[test_times]

# df_train = df[df["_time"].isin(train_times)].copy()
# df_test = df[df["_time"].isin(test_times)].copy()


# ==== TRAINING HELPERS ====
def train_cvxpy_model(
    X_train_agg: pd.DataFrame,
    features: list,
    interval_energy: pd.Series,
    l1_penalty=1.0,
    static_penalty=0.0,
):
    # df = df.copy()
    # df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")
    # df[features] = df[features].fillna(0)
    # df = df[df["_time"].isin(interval_energy.index)]
    # interval_energy = interval_energy.sort_index()

    # # Aggregate process features per interval.
    # df_agg = df.groupby("_time")[features].sum()
    # df_agg = df_agg.reindex(interval_energy.index).fillna(0)

    scaler = MaxAbsScaler()
    X = scaler.fit_transform(X_train_agg.values)

    w = cp.Variable(X.shape[1])
    s = cp.Variable()

    interval_preds = X @ w + s
    loss = cp.sum_squares(interval_preds - interval_energy.values)
    reg = l1_penalty * cp.norm1(w) + static_penalty * cp.abs(s)
    prob = cp.Problem(cp.Minimize(loss + reg), constraints=[s >= 0, w >= 0])
    prob.solve()

    return {"weights": w.value, "static_energy": s.value, "scaler": scaler}


def predict_per_interval(df, weights, scaler, good_features, static_energy):
    df_scaled = df.copy()
    df_scaled[good_features] = scaler.transform(df_scaled[good_features])
    df_scaled["predicted_process_energy"] = df_scaled[good_features].values @ weights
    pred = df_scaled.groupby("_time")["predicted_process_energy"].sum().reset_index()
    return pred


# ==== TRAIN ====
results = train_cvxpy_model(
    X_train_agg,
    good_features,
    interval_energy_train,
    l1_penalty,
    static_penalty,
)
# results = optimizer(df_train, good_features, interval_energy_train)

weights = results["weights"]
static_energy = results["static_energy"]
scaler = results["scaler"]

print("Learned weights:")
for feature_name, weight in zip(good_features, weights):
    print(f"  {feature_name}: {weight:.4e}")
print(f"Static energy component: {static_energy:.4f}")


# ==== PREDICT AND EVALUATE ====
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


# ==== PLOTS: INTERVAL PREDICTIONS ====
fig, ax = plt.subplots(figsize=(7.2, 3.4))

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

ax.set_xlabel("Time", fontsize=12, labelpad=4)
ax.set_ylabel("Interval Energy (Ws)", fontsize=12, labelpad=4)
ax.tick_params(axis="both", labelsize=12)
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
plt.savefig("actual_vs_predicted_interval_energy.png", bbox_inches="tight", dpi=300)
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


# ==== PREPARE PROCESS CONTRIBUTIONS ====
df_test_plot = df_test.copy()
df_test_plot[good_features] = scaler.transform(df_test_plot[good_features])
df_test_plot["estimated_process_energy"] = df_test_plot[good_features].values @ weights

# Merge process instances like wrk_99 and wrk_246 into one base name.
df_test_plot["base_name"] = (
    df_test_plot["process_name"].str.replace(r"_\d+$", "", regex=True).str.strip()
)
df_test_plot.loc[df_test_plot["base_name"] == "", "base_name"] = "unknown"

agg = (
    df_test_plot.groupby(["_time", "base_name"])["estimated_process_energy"]
    .sum()
    .reset_index()
)

pivot = agg.pivot(
    index="_time", columns="base_name", values="estimated_process_energy"
).fillna(0)

N = 8
top_processes = pivot.max().sort_values(ascending=False).head(N).index
pivot_top = pivot[top_processes].copy()

if len(pivot.columns) > N:
    pivot_top["Other"] = pivot.drop(columns=top_processes).sum(axis=1)

print((pivot_top < 0).sum())
print("Fraction of intervals with negative values per process:")
print((pivot_top < 0).mean())

pivot_top_clipped = pivot_top.clip(lower=0)
pivot_mask = (pivot_top_clipped.index >= start_time) & (
    pivot_top_clipped.index <= end_time
)


# ==== PLOTS: TOP BASE PROCESSES ====
fig, ax = plt.subplots(figsize=(7.2, 3.4))
pivot_top_clipped.loc[pivot_mask].plot.area(ax=ax, alpha=0.8, linewidth=0, legend=False)

ax.set_xlabel("Time", fontsize=12, labelpad=4)
ax.set_ylabel("Estimated Process Energy (Ws)", fontsize=12, labelpad=4)
ax.tick_params(axis="both", labelsize=12)

other_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][
    len(pivot_top_clipped.columns) % 10 - 1
]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
top_colors = colors[:8]


class HandlerMultiColor(HandlerBase):
    def create_artists(
        self, legend, orig_handle, x0, y0, width, height, fontsize, trans
    ):
        colors = getattr(orig_handle, "_legend_colors", [other_color])
        n = len(colors)
        artists = []
        for i, color in enumerate(colors):
            rect = Rectangle(
                (x0 + i * width / n, y0),
                width / n,
                height,
                facecolor=color,
                transform=trans,
                lw=0,
            )
            artists.append(rect)
        return artists


top_handle = Rectangle((0, 0), 1, 1)
top_handle._legend_colors = top_colors

other_handle = Rectangle((0, 0), 1, 1)
other_handle._legend_colors = [other_color]

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
plt.savefig("per_process_energy_contribution.png", bbox_inches="tight", dpi=300)
plt.show()


# ==== PLOTS: TOP PIDS ====
N2 = 8

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
#top_pids = pivot2.sum().sort_values(ascending=False).head(N2).index
pivot2_top = pivot2[top_pids].copy()
pivot2_top["Other"] = pivot2.drop(columns=top_pids).sum(axis=1)
pivot2_top_clipped = pivot2_top.clip(lower=0)

fig, ax = plt.subplots(figsize=(12, 5))
pivot2_top_clipped.loc[pivot_mask].plot.area(ax=ax, alpha=0.8, linewidth=0)
ax.set_xlabel("Time", fontsize=12, labelpad=4)
ax.set_ylabel("Estimated Process Energy (Ws)", fontsize=12, labelpad=4)
ax.set_title(
    "Per-PID Energy Contribution Over Time ", fontsize=13, pad=6
)
ax.tick_params(axis="both", labelsize=11)
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9, frameon=True)
plt.tight_layout()
plt.savefig("per_pid_energy_contribution.png", bbox_inches="tight", dpi=200)
plt.show()


# ==== SAVE MODEL ====
os.makedirs("estimation/models", exist_ok=True)
model = {
    "weights": weights,
    "static_energy": static_energy,
    "scaler": scaler,
    "features": good_features,
}
model_path = "estimation/models/model.pkl"
with open(model_path, "wb") as handle:
    pickle.dump(model, handle)
print(f"Model saved to {model_path}")
