#!/usr/bin/env python3
"""Evaluate a saved CVXPY energy model against a new dataset (e.g. from a different node)."""

import argparse
import sys
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_model_artifact(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def prepare_dataset(df, features):
    df = df.copy()
    df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")
    df[features] = df[features].fillna(0)

    interval_energy = (
        df[["_time", "interval_energy"]]
        .dropna()
        .drop_duplicates("_time")
        .set_index("_time")["interval_energy"]
    )
    df = df[df["_time"].isin(interval_energy.index)]
    return df, interval_energy


def predict(df, features, weights, scaler, static_energy):
    df_s = df.copy()
    df_s[features] = scaler.transform(df_s[features])
    df_s["estimated_process_energy"] = df_s[features].values @ weights
    preds = df_s.groupby("_time")["estimated_process_energy"].sum().reset_index()
    preds["predicted_total_energy"] = preds["estimated_process_energy"] + static_energy
    return preds


def evaluate(preds, interval_energy):
    df = preds.merge(
        interval_energy.rename("interval_energy"), left_on="_time", right_index=True
    )
    actual = df["interval_energy"].values
    predicted = df["predicted_total_energy"].values

    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    median_ae = np.median(np.abs(actual - predicted))
    mean_e = actual.mean()

    metrics = {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "median_ae": median_ae,
        "mean_energy": mean_e,
        "mae_pct": 100 * mae / mean_e if mean_e else 0.0,
        "rmse_pct": 100 * rmse / mean_e if mean_e else 0.0,
    }
    return df, metrics


def print_metrics(metrics):
    print("\n── Evaluation Results ──────────────────────────")
    print(f"  R²                   : {metrics['r2']:.4f}")
    print(f"  MAE                  : {metrics['mae']:.4f} Wh")
    print(f"  RMSE                 : {metrics['rmse']:.4f} Wh")
    print(f"  Median AE            : {metrics['median_ae']:.4f} Wh")
    print(f"  Mean interval energy : {metrics['mean_energy']:.4f} Wh")
    print(f"  MAE  (% of mean)     : {metrics['mae_pct']:.2f}%")
    print(f"  RMSE (% of mean)     : {metrics['rmse_pct']:.2f}%")
    print("────────────────────────────────────────────────\n")


def plot_results(df_eval, output_prefix):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_eval["_time"], df_eval["interval_energy"], label="Actual", linewidth=1.5)
    ax.plot(
        df_eval["_time"],
        df_eval["predicted_total_energy"],
        label="Predicted",
        linestyle="--",
        linewidth=1.5,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Interval Energy (Wh)")
    ax.set_title("Actual vs. Predicted Interval Energy (cross-node)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_actual_vs_predicted.png", dpi=150)
    print(f"Saved plot: {output_prefix}_actual_vs_predicted.png")

    fig, ax = plt.subplots(figsize=(10, 3))
    errors = df_eval["interval_energy"] - df_eval["predicted_total_energy"]
    ax.plot(df_eval["_time"], errors, linewidth=1.0)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Error (Wh)")
    ax.set_title("Prediction Error Over Time")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_error_over_time.png", dpi=150)
    print(f"Saved plot: {output_prefix}_error_over_time.png")
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved CVXPY energy model on a new dataset."
    )
    parser.add_argument("--model", required=True, help="Path to model .pkl file")
    parser.add_argument("--data", required=True, help="Path to dataset .parquet file")
    parser.add_argument(
        "--plot-prefix",
        default=None,
        help="If set, save evaluation plots with this filename prefix.",
    )
    args = parser.parse_args()

    artifact = load_model_artifact(args.model)
    features = artifact["features"]
    weights = artifact["weights"]
    scaler = artifact["scaler"]
    static_energy = artifact["static_energy"]

    print(f"Model loaded from : {args.model}")
    print(f"Features          : {features}")
    print(f"Static energy     : {static_energy:.4f} Wh")
    print(f"Weights           : { {f: f'{w:.4e}' for f, w in zip(features, weights)} }")

    df_raw = pd.read_parquet(args.data)
    print(f"\nDataset loaded from : {args.data}")
    print(f"Raw rows            : {len(df_raw)}")

    df, interval_energy = prepare_dataset(df_raw, features)
    print(f"Intervals with energy : {len(interval_energy)}")
    print(f"Process rows used     : {len(df)}")

    preds = predict(df, features, weights, scaler, static_energy)
    df_eval, metrics = evaluate(preds, interval_energy)
    print_metrics(metrics)

    if args.plot_prefix:
        plot_results(df_eval, args.plot_prefix)


if __name__ == "__main__":
    main()
