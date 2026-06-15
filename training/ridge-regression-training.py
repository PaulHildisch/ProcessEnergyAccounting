#! /usr/bin/python3

import pandas as pd
import numpy as np
import joblib
import time
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_absolute_error

def clean_dataset(df, selected_features: list[str]):
    df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")

    for feature in selected_features:
        if feature not in df.columns:
            print(f"Feature {feature} was selected but is not present in dataset. Removing from selection.")
            selected_features.remove(feature)

    df[selected_features] = df[selected_features].fillna(0)

    interval_energy_all = (
        df[["_time", "interval_energy"]]
        .dropna()
        .drop_duplicates("_time")
        .set_index("_time")["interval_energy"]
    )
    df = df[df["_time"].isin(interval_energy_all.index)]

    interval_energy_all = interval_energy_all.sort_index()

    #aggregation
    df_agg = df.groupby("_time")[selected_features].sum()
    df_agg = df_agg.reindex(interval_energy_all.index).fillna(0)

    df_train, df_test, interval_energy_train, interval_energy_test = train_test_split(
        df_agg, interval_energy_all, test_size=0.2, shuffle=False
    )

    #scaling
    scaler = StandardScaler()
    df_train_scaled = scaler.fit_transform(df_train)
    df_test_scaled = scaler.transform(df_test)

    return {"x": {"train": df_train_scaled, "test": df_test_scaled}, "y": {"train": interval_energy_train, "test": interval_energy_test}}

def evaluate_model(model, training_data):
    #evaluate
    prediction = model.predict(training_data["x"])
    
    r2 = r2_score(training_data["y"], prediction)
    mae = mean_absolute_error(training_data["y"], prediction)

    return r2, mae


def feature_selection(df, model, feature_candidates, min_gain=0.01):
    # ==== SEQUENTIAL FORWARD SELECTION ====
    selected = []
    remaining = list(feature_candidates)
    best_r2 = -np.inf
    results_log = []

    print("=== Sequential Forward Selection ===\n")

    while remaining:
        step_best_r2 = -np.inf
        step_best_feature = None
        step_best_mae = None

        for feature in remaining:
            candidate_set = selected + [feature]
            training_data = clean_dataset(df, candidate_set)

            model.fit(training_data["x"]["train"], training_data["y"]["train"])
            r2, mae = evaluate_model(model=model, training_data=training_data)
            
            if r2 is None:
                continue

            print(
                f"  [{'+'.join(candidate_set)}]  "
                f"R²={r2:.4f}  "
                f"MAE%={100 * mae / training_data["y"]["test"].mean():.2f}%"
            )

            if r2 > step_best_r2:
                step_best_r2 = r2
                step_best_feature = feature
                step_best_mae = mae

        gain = step_best_r2 - best_r2
        if step_best_feature is None or gain < min_gain:
            print(f"\nStopping: best gain {gain:.4f} < min_gain {min_gain}")
            break

        selected.append(step_best_feature)
        remaining.remove(step_best_feature)
        best_r2 = step_best_r2

        mean_energy = training_data["y"]["test"].mean()
        results_log.append(
            {
                "step": len(selected),
                "added": step_best_feature,
                "features": list(selected),
                "r2": step_best_r2,
                "mae_pct": 100 * step_best_mae / mean_energy,
            }
        )

        print(
            f"\n→ Step {len(selected)}: added '{step_best_feature}'  "
            f"R²={step_best_r2:.4f}  "
            f"MAE%={100 * step_best_mae / mean_energy:.2f}%"
        )
        print(f"  Selected so far: {selected}\n")

    return selected

def main(args):
    """
    Available Features:
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
    """
    all_features = ["delta_cpu_ns", "delta_cycles", "delta_instructions", "delta_cache_misses", "delta_branch_instructions", "delta_io_bytes", "delta_net_send_bytes", "context_switches", "syscall_count", "delta_rss_memory", "syscall_class_file", "syscall_class_network", "syscall_class_memory", "syscall_class_process", "syscall_class_other", "syscall_class_sched", "syscall_class_signal", "syscall_class_time",]
    selected_features = ['context_switches', 'syscall_class_network', 'delta_branch_instructions', 'syscall_class_time']
    
    print("Loading data...")
    df = pd.read_parquet(args.dataSource)
    

    if args.modelFile:
        print(f"Loading model from file {args.modelFile}...")
        model = joblib.load(args.modelFile)
        test_data = df
        print("Preparing Dataset...")
        data_set = clean_dataset(df, selected_features)

    else:
        print("Training new Model...")
        model=KernelRidge(alpha=1.0, kernel='rbf')

        if args.sfs:
            selected_features = feature_selection(df, model, all_features)    

        print("Preparing Dataset...")
        data_set = clean_dataset(df, selected_features)

        print(f"Training model with features {selected_features}")
        model.fit(data_set["x"]["train"], data_set["y"]["train"])

        print("Saving Model...")
        outpath = joblib.dump(model, f"models/l2-regression-{time.strftime("%m%d%H%M%S")}", compress=3)
        print(f"Model saved to {outpath}")


    print("Evaluating Model...")
    test_data = {"x": data_set["x"]["test"], "y": data_set["y"]["test"]}
    r2, mae = evaluate_model(model, test_data) 
    mae_percent = 100 * mae / test_data["y"].mean()

    print("-" * 34)
    print(f"  R² Score:  {r2:.4f}")
    print(f"  MAE:       {mae:.2f} Wh ({mae_percent:.2f}%)")
    print("-" * 34)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--modelFile")
    parser.add_argument("--dataSource", default="data/nf_core_test-full_0530-4.parquet")
    parser.add_argument("--sfs", action="store_true", default=False)

    args = parser.parse_args()

    main(args)