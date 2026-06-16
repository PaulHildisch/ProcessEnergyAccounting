#! /usr/bin/python3

import pandas as pd
import numpy as np
import joblib
import time
import pickle
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_absolute_error

def prepare_dataset(df):
    x = df[df.columns[1:]]
    y = df[df.columns[0]]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=False
    )

    #scaling
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return {"x": {"train": x_train_scaled, "test": x_test_scaled}, "y": {"train": y_train, "test": y_test}}, scaler

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
    
    print("Loading data...")
    df = pd.read_parquet(args.dataSource)
    selected_features = df.columns[1:]
    
    print("Training new Model...")
    model=KernelRidge(alpha=1.0, kernel='rbf')

    print("Preparing Dataset...")
    data_set, scaler = prepare_dataset(df)

    print(f"Training model with features {selected_features.values}")
    model.fit(data_set["x"]["train"], data_set["y"]["train"])

    print("Saving Model...")
    timestamp = time.strftime("%m%d%H%M%S")
    outpath = joblib.dump(model, f"models/l2-regression-{timestamp}", compress=3)
    print(f"Model saved to {outpath}")

    print("Saving scaler used for model...")
    with open(f"models/l2-regression-{timestamp}-scaler.npy", "w+b") as scaler_out:
        scaler_out.write(pickle.dumps(scaler))
        print(f"Saved scaler to {scaler_out.name}")

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

    parser.add_argument("--dataSource", default="data/nf_core_test-full_0530-4-cleaned.parquet")

    args = parser.parse_args()

    main(args)