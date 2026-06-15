import pandas as pd
import pickle
import argparse
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filepath")

args = parser.parse_args()

filename = args.filepath.split('.')[0]

df = pd.read_parquet(args.filepath)

# features = ["delta_cpu_ns", "delta_cycles", "delta_instructions", "delta_cache_misses", "delta_branch_instructions", "delta_io_bytes", "delta_net_send_bytes", "context_switches", "syscall_count", "delta_rss_memory", "syscall_class_file", "syscall_class_network", "syscall_class_memory", "syscall_class_process", "syscall_class_other", "syscall_class_sched", "syscall_class_signal", "syscall_class_time",]
features = ['context_switches', 'syscall_class_network', 'delta_branch_instructions', 'syscall_class_time']

df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")

for feature in features:
    if feature not in df.columns:
        print(f"Feature {feature} was selected but is not present in dataset. Removing from selection.")
        features.remove(feature)

df[features] = df[features].fillna(0)

interval_energy_all = (
    df[["_time", "interval_energy"]]
    .dropna()
    .drop_duplicates("_time")
    .set_index("_time")["interval_energy"]
)
df = df[df["_time"].isin(interval_energy_all.index)]

interval_energy_all = interval_energy_all.sort_index()

#aggregation
df_agg = df.groupby("_time")[features].sum()
df_agg = df_agg.reindex(interval_energy_all.index).fillna(0)

#scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_agg)

interval_energy_all.to_json(f"{filename}-actual.json")
with open(f"{filename}-cleaned.npy", 'wb') as outfile:
    outfile.write(pickle.dumps(df_scaled))

