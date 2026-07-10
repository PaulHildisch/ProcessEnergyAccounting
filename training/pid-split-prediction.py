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
import warnings

def evaluate_model(prediction, actual):
    r2 = r2_score(actual, prediction)
    mae = mean_absolute_error(actual, prediction)

    print("-" * 34)
    print(f"  R² Score:  {r2:.4f}")
    print(f"  MAE:       {mae:.2f} Wh ({100 * mae / actual.mean().interval_energy:.2f}%)")
    print("-" * 34)

    return r2, mae

def plot(prediction, actual, range = None, title="L2 Regression - Actual & Predicted Energy Consumption"):
    time_frame_np = np.arange(actual.index.values[0], actual.index.values[-1], 1, dtype='datetime64[s]')
    print(time_frame_np.size)
    print(actual.shape[0])
    print(prediction.shape[0])

    if range == None:
        start = 0
        end = -1
    else: 
        start = int(actual.shape[0] / 2 - range / 2)
        end = int(actual.shape[0] / 2 + range / 2)

    _ , ax = plt.subplots(figsize=(10, 5))
    
    ax.scatter(
        actual.index.values[start:end],
        actual[start:end],
        label="Actual Energy",
        s=0.2,
        c='g'
    )
    ax.plot(
        prediction.index.values[start:end],
        prediction[start:end],
        label="Predicted Energy",
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel("Time", fontsize=10, labelpad=4)
    ax.xaxis.set_major_locator(dates.SecondLocator(interval=int(range/10)))
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

def predictions(data, model, zero_prediction, actual):
    print("making predictions")
    preds = pd.DataFrame(0, index=actual.index, columns=actual.columns) 
    for timestamp, interval_df in data.groupby(level=0):
        interval_df = interval_df.droplevel(0)
        interval_pred = 0
        for pid, pid_df in interval_df.groupby(level=0):
            prediction = model.predict(pid_df)
            interval_pred += prediction - zero_prediction
        #print(f"{timestamp} - Interval Prediction: {interval_pred}\n{timestamp} - Interval + Zero: {zero_prediction +  interval_pred}\n{timestamp} - Recorded Value: {actual.loc[timestamp, "interval_energy"]}")
        interval_pred += zero_prediction
        preds.loc[timestamp] = interval_pred
    return preds

def read_data(measurementsPath, targetsPath, scaler):
    x = pd.read_parquet(measurementsPath)
    y = pd.read_parquet(targetsPath)
    print(y.mean().interval_energy)

    print("Scaling data")
    if scaler:
        for timestamp, interval_df in x.groupby(level=0):
            interval_df = interval_df.droplevel(0)
            transformed = scaler.transform(interval_df)
            transformed_df = pd.DataFrame(data=transformed,
                                           index=interval_df.index,
                                           columns=interval_df.columns)
            x.loc[timestamp].update(transformed_df)
    else: 
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    return x,y

def main(args):
    modelFile = joblib.load(args.modelFile)
    model = modelFile["model"]
    scaler = modelFile["scaler"]

    zero_df = pd.DataFrame(scaler.transform(np.zeros((1, 12))))
    zero_prediction = model.predict(zero_df)
    print(zero_prediction)
    data, actual = read_data(args.pidDataSource, args.targetDataSource, scaler)

    prediction = predictions(data, model, zero_prediction, actual)
    evaluate_model(prediction, actual)
    
    if args.full:
        plot(prediction, actual)
    else:
        plot(prediction, actual, range=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--modelFile")
    parser.add_argument("--targetDataSource", default="data/sarek_2_0207-cleaned-targets.parquet")
    parser.add_argument("--pidDataSource", default="data/sarek_2_0207-cleaned-pid.parquet")
    parser.add_argument("--full", action="store_true", default=False)

    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args)