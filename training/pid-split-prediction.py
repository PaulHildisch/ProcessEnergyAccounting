import math

import numpy as np
import pandas as pd
import time
import argparse
import joblib
import warnings
import progressbar
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from matplotlib import dates as dates
from sklearn.metrics import r2_score, mean_absolute_error

def evaluate_model(prediction, actual, zero_prediction):
    prediction_agg = prediction.groupby(['_time'])['interval_energy'].sum().reset_index()
    prediction_agg = prediction_agg['interval_energy'] + zero_prediction
    r2 = r2_score(actual, prediction_agg)
    mae = mean_absolute_error(actual, prediction_agg)

    print("-" * 34)
    print(f"  R² Score:  {r2:.4f}")
    print(f"  MAE:       {mae:.2f} Wh ({100 * mae / actual.mean().interval_energy:.2f}%)")
    print("-" * 34)

    return r2, mae

def plot(zero_prediction, prediction, actual, range = None, title="Individual PID Attributions"):
    top_n = 8
    if range == None:
        start = 0
        end = actual.shape[0]
    else: 
        start = int(actual.shape[0] / 2 - range / 2)
        end = int(actual.shape[0] / 2 + range / 2)

    _ , ax = plt.subplots(figsize=(10, 5))

    actual.plot.line(ax=ax, linewidth=1, linestyle="solid")

    agg = prediction.groupby(['_time', "pid_label"])['interval_energy'].sum().reset_index()
    pivot = agg.pivot(index='_time', columns="pid_label", values='interval_energy').fillna(0)

    # top_pids = pivot.max().sort_values(ascending=False).head(top_n).index
    top_pids = pivot.sum().sort_values(ascending=False).head(top_n).index
    pivot_top = pivot[top_pids].copy()
    
    if len(pivot.columns) > top_n:
        pivot_top["Other"] = pivot.drop(columns=top_pids).sum(axis=1)
        
    pivot_top_clipped = pivot_top.clip(lower=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot_top_clipped.plot.area(ax=ax, alpha=0.8, linewidth=0)
    

    ax.set_xlabel("Time", fontsize=10, labelpad=4)
    ax.xaxis.set_major_locator(dates.SecondLocator(interval=int((end-start)/3)))
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
    ax.set_ylabel("Interval Energy (Ws)", fontsize=12, labelpad=4)
    ax.tick_params(axis="both", labelsize=10)
    
    # Standard legend for PID view (too many for a single grouped color block)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig(f"plots/actual_vs_predicted_interval_energy-{time.strftime("%m%d%H%M%S")}.png", bbox_inches="tight", dpi=300)

def predictions(data, model, zero_prediction) -> pd.DataFrame:
    #data = data.reset_index().set_index("_time")
    preds = pd.DataFrame(data['pid'], columns=['pid', 'interval_energy']).astype({"pid":"int64","interval_energy":"float64"})
    bar = progressbar.ProgressBar(max_value=len(data.index.to_series().unique()), widgets=['Making predictions:', ' ', progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ', progressbar.Timer()], redirect_stdout=True)
    for timestamp in data.index.to_series().unique():
        predictions = model.predict(data.loc[timestamp, data.columns != 'pid']) - zero_prediction
        preds.loc[timestamp, 'interval_energy'] = predictions
        bar.update(bar.value+1)
    bar.finish('\n')
    return preds

def read_data(measurementsPath, targetsPath, scaler):
    x = pd.read_parquet(measurementsPath)
    y = pd.read_parquet(targetsPath)

    x_labels = x[['pid_label', 'base_name']]
    x = x.drop(['pid_label', 'base_name'], axis=1)

    if scaler:
        bar = progressbar.ProgressBar(max_value=len(x.groupby(level=0)), widgets=['Scaling data:', ' ', progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ', progressbar.Timer()], redirect_stdout=True)

        for timestamp, interval_df in x.groupby(level=0):
            interval_df = interval_df.droplevel(0)
            #interval_df[["delta_cpu_ns", "delta_io_bytes", "delta_net_send_bytes", "context_switches", "syscall_count", "delta_rss_memory", "delta_cpu_time_proc", "syscall_class_file", "syscall_class_network", "syscall_class_memory", "syscall_class_other", "syscall_class_signal"]] = scaler.transform(interval_df.drop(['pid_label', 'base_name'], axis=1))
            # print(interval_df)
            x.loc[timestamp].iloc[:,:] = scaler.transform(interval_df)
            bar.update(bar.value+1)
        bar.finish('\n')
    else: 
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    x = x.reset_index().set_index('_time')
    x_labels = x_labels.reset_index().set_index('_time')
    return x, y, x_labels

def main(args):
    modelFile = joblib.load(args.modelFile)
    model = modelFile["model"]
    scaler = modelFile["scaler"]

    zero_df = pd.DataFrame(scaler.transform(np.zeros((1, model.n_features_in_))))
    zero_prediction = model.predict(zero_df)
    print(f"Predicted idle energy: {zero_prediction}")
    data, actual, labels = read_data(args.pidDataSource, args.targetDataSource, scaler)

    prediction = predictions(data, model, zero_prediction)
    prediction[['pid_label', 'base_name']] = labels[['pid_label', 'base_name']] 
    evaluate_model(prediction, actual, zero_prediction)
    
    modelname = "Random Forest"
    if args.full:
        plot(zero_prediction[0], prediction, actual, title=f"Individual PID Attributions - {modelname}")
    else:
        plot(zero_prediction[0], prediction, actual, range=600, title=f"Individual PID Attributions - {modelname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--modelFile")
    parser.add_argument("--targetDataSource")
    parser.add_argument("--pidDataSource")
    parser.add_argument("--full", action="store_true", default=False)

    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args)