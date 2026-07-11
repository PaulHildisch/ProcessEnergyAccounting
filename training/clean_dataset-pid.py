import pandas as pd
import argparse
from sklearn.preprocessing import (StandardScaler)
from sklearn.model_selection import (train_test_split)

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filepath")
parser.add_argument("--features")
parser.add_argument("--pid-split", action="store_true", default=False)

args = parser.parse_args()

filename = args.filepath.split('.')[0]

df = pd.read_parquet(args.filepath)

# To make this dynamic we have to save the features used to train the model and read them here.
# features = ["delta_cpu_ns", "delta_cycles", "delta_instructions", "delta_cache_misses", "delta_branch_instructions", "delta_io_bytes", "delta_net_send_bytes", "context_switches", "syscall_count", "delta_rss_memory", "syscall_class_file", "syscall_class_network", "syscall_class_memory", "syscall_class_process", "syscall_class_other", "syscall_class_sched", "syscall_class_signal", "syscall_class_time",]
# features = ['context_switches', 'syscall_class_network', 'delta_branch_instructions', 'syscall_class_time']

# RF
# features = ["delta_cpu_ns", "delta_io_bytes", "delta_net_send_bytes", "context_switches", "syscall_count", "delta_rss_memory", "delta_cpu_time_proc", "syscall_class_file", "syscall_class_network", "syscall_class_memory", "syscall_class_other", "syscall_class_signal"]

# L2 Ridge - automatic feature selection
# features = ['delta_cpu_ns', 'delta_net_send_bytes', 'syscall_count', 'syscall_class_file', 'syscall_class_network', 'syscall_class_memory', 'syscall_class_process', 'syscall_class_other', 'syscall_class_sched']

# L2 Ridge - generalized features
features = ['delta_io_bytes', 'context_switches', 'delta_cpu_ns', 'delta_net_send_bytes', 'syscall_count']

if args.features:
    print(f"--features is not implemented. Using hardcoded values: ({features})")

df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")

for feature in features:
    if feature not in df.columns:
        print(f"Feature {feature} was selected but is not present in dataset. Removing from selection.")
        features.remove(feature)

df[features] = df[features].fillna(0)

# OLD
# interval_energy_all = (
#     df[["_time", "interval_energy"]]
#     .dropna()
#     .drop_duplicates("_time")
#     .set_index("_time")["interval_energy"]
# )
# df = df[df["_time"].isin(interval_energy_all.index)]

# interval_energy_all = interval_energy_all.sort_index()

# #aggregation
# df_agg = df.groupby("_time")[features].sum()

# out = pd.concat([interval_energy_all, df_agg], axis=1)
# out.to_parquet(f"{filename}-cleaned.parquet")
# print(f"Saving unscaled dataset with features \n{features}\nto \"{filename}-cleaned.parquet\"")


# Store actual values for evaluation
interval_energy_all = (
    df[["_time", "interval_energy"]]
    .dropna()
    .drop_duplicates("_time")
    .set_index(["_time"])
)
df = df[df["_time"].isin(interval_energy_all.index)]


interval_energy_all = interval_energy_all.sort_index()
interval_energy_all.to_parquet(f"{filename}-cleaned-targets.parquet")

# store aggregated counters
df_agg = df.groupby("_time")[features].sum()

df_agg = pd.concat([interval_energy_all, df_agg], axis=1)
df_agg.to_parquet(f"{filename}-cleaned-aggregated.parquet")

# store rows where any recorded value > 0 (meaning something was recorded for the given pid at the given time)
df = df.set_index(["_time","pid"])[features]
df_pid = df[(df > 0).any(axis=1)]

df_pid.to_parquet(f"{filename}-cleaned-pid.parquet")
print(f"Saving unscaled dataset with features \n{features}\nto \"{filename}-cleaned.parquet\"")