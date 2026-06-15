import numpy as np
import pandas as pd
import time
import argparse
import joblib
import pickle
from matplotlib import pyplot as plt
from matplotlib import dates as dates


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


def main(args):
    model = joblib.load(args.modelFile)
    data = None
    with open(args.dataSource, 'rb') as dataFile:
        data = pickle.loads(dataFile.read())
    actual = pd.read_json(args.actualData, typ="series")
    prediction = model.predict(data)
    
    plot(prediction, actual, range=120)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--modelFile")
    parser.add_argument("--dataSource", default="data/nf_core_test-full_0530-4-cleaned.npy")
    parser.add_argument("--actualData", default="data/nf_core_test-full_0530-4-actual.json")

    args = parser.parse_args()
    main(args)