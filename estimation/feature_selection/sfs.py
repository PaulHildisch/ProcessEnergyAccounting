"""
Sequential Forward Selection (SFS) for CVXPY energy estimator.

Greedily adds the feature that improves test R² most at each step.
Stops when no feature improves R² by more than --min-gain.
O(n²) evaluations instead of O(2^n).
"""

import argparse
import sys

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

CANDIDATE_FEATURES = [
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

parser = argparse.ArgumentParser(description="Sequential forward selection for energy model")
parser.add_argument("--data", default="runs/benchmark-siena06-v6/process_interval_data.parquet")
parser.add_argument("--l1-penalty", type=float, default=0.1)
parser.add_argument("--min-gain", type=float, default=0.01, help="Minimum R² improvement to keep adding features")
parser.add_argument("--test-size", type=float, default=0.2)
args = parser.parse_args()

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"Loading {args.data} ...")
df = pd.read_parquet(args.data)
df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")

# Filter to only features that exist in the dataset
available = [f for f in CANDIDATE_FEATURES if f in df.columns]
missing = [f for f in CANDIDATE_FEATURES if f not in df.columns]
if missing:
    print(f"Skipping (not in dataset): {missing}")

df[available] = df[available].fillna(0)

interval_energy_all = (
    df[["_time", "interval_energy"]]
    .dropna()
    .drop_duplicates("_time")
    .set_index("_time")["interval_energy"]
)
df = df[df["_time"].isin(interval_energy_all.index)]

time_values = interval_energy_all.index.sort_values()
train_times, test_times = train_test_split(time_values, test_size=args.test_size, shuffle=False)
interval_energy_train = interval_energy_all.loc[train_times]
interval_energy_test = interval_energy_all.loc[test_times]
df_train = df[df["_time"].isin(train_times)].copy()
df_test = df[df["_time"].isin(test_times)].copy()

print(f"Train intervals: {len(train_times)}  |  Test intervals: {len(test_times)}")
print(f"Candidate features: {available}\n")


# ── CVXPY training ─────────────────────────────────────────────────────────────
def train_and_evaluate(features):
    df_tr = df_train.copy()
    df_tr[features] = df_tr[features].fillna(0)
    df_tr = df_tr[df_tr["_time"].isin(interval_energy_train.index)].sort_values("_time")
    ie_tr = interval_energy_train.sort_index()

    df_agg = df_tr.groupby("_time")[features].sum()
    df_agg = df_agg.reindex(ie_tr.index).fillna(0)

    scaler = MaxAbsScaler()
    X = scaler.fit_transform(df_agg.values)

    w = cp.Variable(X.shape[1])
    s = cp.Variable()
    interval_preds = X @ w + s
    loss = cp.sum_squares(interval_preds - ie_tr.values)
    reg = args.l1_penalty * cp.norm1(w)
    prob = cp.Problem(cp.Minimize(loss + reg), constraints=[s >= 0, w >= 0])
    prob.solve(solver=cp.CLARABEL)

    if w.value is None:
        return None, None, None

    weights = w.value
    static_energy = s.value

    # Evaluate on test set
    df_te = df_test.copy()
    df_te[features] = df_te[features].fillna(0)
    df_te[features] = scaler.transform(df_te[features])
    df_te["pred"] = df_te[features].values @ weights
    pred = df_te.groupby("_time")["pred"].sum() + static_energy
    actual = interval_energy_test

    aligned = pred.reindex(actual.index).dropna()
    actual_aligned = actual.loc[aligned.index]

    r2 = r2_score(actual_aligned, aligned)
    mae = mean_absolute_error(actual_aligned, aligned)
    return r2, mae, dict(zip(features, weights))


# ── Sequential Forward Selection ───────────────────────────────────────────────
selected = []
remaining = list(available)
best_r2 = -np.inf
results_log = []

print("=== Sequential Forward Selection ===\n")

while remaining:
    step_best_r2 = -np.inf
    step_best_feature = None
    step_best_mae = None
    step_best_weights = None

    for feature in remaining:
        candidate_set = selected + [feature]
        r2, mae, weights = train_and_evaluate(candidate_set)
        if r2 is None:
            continue
        print(f"  [{'+'.join(candidate_set)}]  R²={r2:.4f}  MAE%={100*mae/interval_energy_test.mean():.2f}%")
        if r2 > step_best_r2:
            step_best_r2 = r2
            step_best_feature = feature
            step_best_mae = mae
            step_best_weights = weights

    gain = step_best_r2 - best_r2
    if step_best_feature is None or gain < args.min_gain:
        print(f"\nStopping: best gain {gain:.4f} < min_gain {args.min_gain}")
        break

    selected.append(step_best_feature)
    remaining.remove(step_best_feature)
    best_r2 = step_best_r2

    mean_energy = interval_energy_test.mean()
    results_log.append({
        "step": len(selected),
        "added": step_best_feature,
        "features": list(selected),
        "r2": step_best_r2,
        "mae_pct": 100 * step_best_mae / mean_energy,
        "weights": step_best_weights,
    })

    print(f"\n→ Step {len(selected)}: added '{step_best_feature}'  R²={step_best_r2:.4f}  MAE%={100*step_best_mae/mean_energy:.2f}%")
    print(f"  Selected so far: {selected}\n")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"{'Step':<6} {'Added Feature':<35} {'R²':>8} {'MAE%':>8}  Features")
print("-" * 90)
for r in results_log:
    print(f"{r['step']:<6} {r['added']:<35} {r['r2']:>8.4f} {r['mae_pct']:>7.2f}%  {r['features']}")

if results_log:
    best = max(results_log, key=lambda x: x["r2"])
    print(f"\nBest combination (R²={best['r2']:.4f}, MAE%={best['mae_pct']:.2f}%):")
    print(f"  {best['features']}")
    print(f"\nWeights:")
    for f, w in best["weights"].items():
        print(f"  {f}: {w:.4e}")
