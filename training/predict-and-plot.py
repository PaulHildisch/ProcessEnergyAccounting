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

def evaluate_model(prediction, actual):
    r2 = r2_score(actual, prediction)
    mae = mean_absolute_error(actual, prediction)

    print("-" * 34)
    print(f"  R² Score:  {r2:.4f}")
    print(f"  MAE:       {mae:.2f} Wh ({100 * mae / actual.mean().interval_energy:.2f}%)")
    print("-" * 34)

    return r2, mae

def plot(prediction, actual, range = None, title="Individual PID Attributions"):
    if range == None:
        start = 0
        end = actual.shape[0]
    else: 
        start = int(actual.shape[0] / 2 - range / 2)
        end = int(actual.shape[0] / 2 + range / 2)

    _ , ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        actual.index.values[start:end],
        actual[start:end],
        label="Actual Energy",
        linewidth=1.0
    )
    ax.plot(
        actual.index.values[start:end],
        prediction[start:end],
        label="Predicted Energy",
        linestyle="--",
        linewidth=1.0
    )

    ax.set_xlabel("Time", fontsize=10, labelpad=4)
    ax.xaxis.set_major_locator(dates.SecondLocator(interval=int((end-start)/3)))
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
    ax.set_ylabel("Interval Energy (Ws)", fontsize=12, labelpad=4)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(
        loc="upper right",
        fontsize=10,
        frameon=True
    )
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(f"plots/actual_vs_predicted_interval_energy-{time.strftime("%m%d%H%M%S")}.png", bbox_inches="tight", dpi=300)
    plt.close()

def predictions(data, model):
    preds = model.predict(data)
    print(preds[:5])
    return preds

def read_data(measurementsPath, targetsPath, scaler):
    x = pd.read_parquet(measurementsPath)
    y = pd.read_parquet(targetsPath)

    if scaler:
        x = scaler.transform(x)
    else: 
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    return x,y

def main(args):
    modelFile = joblib.load(args.modelFile)
    model = modelFile["model"]
    scaler = modelFile["scaler"]

    data, actual = read_data(args.pidDataSource, args.targetDataSource, scaler)

    prediction = predictions(data, model)
    # evaluate_model(prediction, actual)
    
    modelname = "Random Forest"
    if args.full:
        plot(prediction, actual, title=f"Individual PID Attributions - {modelname}")
    else:
        plot(prediction, actual, range=600, title=f"Individual PID Attributions - {modelname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--modelFile")
    parser.add_argument("--targetDataSource")
    parser.add_argument("--pidDataSource")
    parser.add_argument("--full", action="store_true", default=False)

    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args)