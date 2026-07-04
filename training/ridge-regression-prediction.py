import numpy as np
import pandas as pd
import time
import argparse
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from matplotlib import dates as dates
from sklearn.metrics import r2_score, mean_absolute_error


def evaluate_model(prediction, actual):
    r2 = r2_score(actual, prediction)
    mae = mean_absolute_error(actual, prediction)

    print("-" * 34)
    print(f"  R² Score:  {r2:.4f}")
    print(f"  MAE:       {mae:.2f} Wh ({100 * mae / actual.mean():.2f}%)")
    print("-" * 34)

    return r2, mae

def plot(prediction, actual, range=300, title="L2 Regression - Actual & Predicted Energy Consumption"):
    time_frame_np = np.arange(actual.index.values[0], actual.index.values[-1], 1, dtype='datetime64[s]')

    start = int(time_frame_np.size / 2 - range / 2)
    end = int(time_frame_np.size / 2 + range / 2)

    _ , ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(
        time_frame_np[start:end],
        actual[start:end],
        label="Actual Energy",
        linewidth=1.0,
    )
    ax.plot(
        time_frame_np[start:end],
        prediction[start:end],
        label="Predicted Energy",
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel("Time", fontsize=10, labelpad=4)
    ax.xaxis.set_major_locator(dates.SecondLocator(interval=30))
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
    ax.set_ylabel("Interval Energy (Wh)", fontsize=12, labelpad=4)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(
        loc="upper right",
        fontsize=10,
        frameon=True
    )
    ax.set_title(title, fontsize=14)

    plt.savefig(f"plots/actual_vs_predicted_interval_energy-{time.strftime("%m%d%H%M%S")}.png", bbox_inches="tight", dpi=300)
    plt.close()

def read_data(path, scaler):
    df = pd.read_parquet(path)
    df = df.set_index('_time')
    x = df[df.columns[1:]]
    y = df[df.columns[0]]

    if scaler:
        x = scaler.transform(x)
    else: 
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    return x,y

def main(args):
    model = joblib.load(args.modelFile)

    if args.scalerFile:
        with open(args.scalerFile, "r+b") as scaler_in:
            scaler = pickle.loads(scaler_in.read())
    else:
        scaler = None

    data, actual = read_data(args.dataSource, scaler)

    prediction = model.predict(data)
    evaluate_model(prediction, actual)
    
    plot(prediction, actual, range=120)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--modelFile")
    parser.add_argument("--dataSource", default="data/nf_core_test-full_0530-4-cleaned.parquet")
    parser.add_argument("--scalerFile", default=None)

    args = parser.parse_args()
    main(args)