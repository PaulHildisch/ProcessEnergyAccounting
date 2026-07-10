import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


#for wf -> how did i search these up? -> if all are available

#These seem to generalize well to other benchmarks as well
#good_features = ['delta_io_bytes', 'delta_cycles', 'delta_cpu_ns', 'delta_branch_instructions', 'syscall_class_signal']# R² 98% and 3,15% for the workflow data which is crazy
#for wf -> rf search, if delta io bytes are eliminated
good_features = ['context_switches', 'syscall_class_network', 'syscall_class_sched', 'delta_cycles', 'delta_branch_instructions', 'syscall_class_signal']
#for single bench random forest
#good_features =   ['delta_io_bytes', 'syscall_class_file'] #99,31 2,30%
#good_features = ['delta_io_bytes', 'syscall_class_network', 'delta_net_send_bytes', 'delta_cpu_ns']
#for linear
#good_features = ['delta_io_bytes', 'syscall_class_network']

#good_features = ['context_switches', 'delta_cycles', 'syscall_class_network']

#good_features = ['delta_io_bytes', 'delta_cache_misses', 'delta_net_send_bytes']

#good_features = ['context_switches', 'delta_cache_misses']

target = "interval_energy"


print("Loading data...")
df = pd.read_parquet("data/single_benchmarks/benchmark_primesieve.parquet")
#df = pd.read_parquet("data/phoronix/single_bench_data.parquet")
#df = pd.read_parquet("data/single_benchmarks/clean_benchmark_dbench.parquet")
#df = pd.read_parquet("data/process_interval_data_wf.parquet")



print("Processing timestamps and cleaning missing values...")
df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")


for feature in good_features:
    if feature not in df.columns:
        df[feature] = 0.0

df[good_features] = df[good_features].fillna(0)

# extract interval energy
print("Aligning energy measurements with process intervals...")
interval_energy_all = (
    df[["_time", target]]
    .dropna()
    .drop_duplicates("_time")
    .set_index("_time")[target]
)

# keep only rows that belong to intervals with measured energy
df = df[df["_time"].isin(interval_energy_all.index)]
interval_energy_all = interval_energy_all.sort_index()
print(f"Total valid intervals: {len(interval_energy_all)}")

#aggregation
df_agg = df.groupby("_time")[good_features].sum()
df_agg = df_agg.reindex(interval_energy_all.index).fillna(0)

#train test split
X = df_agg.values
y = interval_energy_all.values
times = interval_energy_all.index.values # capture timestamps for the X-axis of the plot

#split data sequentially (past predicts future), including the timestamps
X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
    X, y, times, test_size=0.2, shuffle=False
)

#scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Define models -> could be other models as well
models = {
    #"Linear (Positive Lasso)": Lasso(alpha=1.0, positive=True, max_iter=10000),
    #"Ridge Regression": KernelRidge(alpha=1.0, kernel='rbf'),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    #"Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

#train and evaluate
print("\n--- MODEL COMPARISON BENCHMARK ---")
mean_energy = y_test.mean()
print(f"Mean Interval Energy (Test Set): {mean_energy:.2f} Wh\n")


model_predictions = {}

for name, model in models.items():

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    model_predictions[name] = y_pred 
    
    #evaluate
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mae_pct = (mae / mean_energy) * 100
    
    print(f"[{name}]")
    print(f"  R² Score:  {r2:.4f}")
    print(f"  MAE:       {mae:.2f} Wh ({mae_pct:.2f}% of mean)")
    print("-" * 34)

#Plot -> [50: window] only used for large data sets
print("\nGenerating Actual vs. Predicted plot...")

fig, ax = plt.subplots(figsize=(7.2, 3.4))
window = 200
ax.plot(
    t_test,#[50:window],
    y_test,#[50:window],
    label="Actual Energy",
    linewidth=2.0,
)
ax.plot(
    t_test,#[50:window],
    y_pred,#[50:window],
    label="Predicted (Random Forest)",
    linestyle="--",
    linewidth=2.0,
)

ax.set_xlabel("Time", fontsize=12, labelpad=4)
ax.set_ylabel("Interval Energy (Wh)", fontsize=12, labelpad=4)
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