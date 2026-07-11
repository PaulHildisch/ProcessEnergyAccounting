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
    
    ax.plot(
        actual.index.values[start:end],
        actual[start:end],
        label="Actual Energy",
        linewidth=1.0
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
    preds = pd.DataFrame(0, index=actual.index, columns=actual.columns) 
    bar = progressbar.ProgressBar(max_value=len(data.groupby(level=0)), widgets=['Making predictions:', ' ', progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ', progressbar.Timer()], redirect_stdout=True)
    for timestamp, interval_df in data.groupby(level=0):
        interval_df = interval_df.droplevel(0)
        interval_pred = 0
        predictions = model.predict(interval_df)
        interval_pred = sum([prediction - zero_prediction for prediction in predictions])
        # print(interval_pred)
        # for pid, pid_df in interval_df.groupby(level=0):
        #     prediction = model.predict(pid_df)
        #     interval_pred += prediction - zero_prediction
        # #print(f"{timestamp} - Interval Prediction: {interval_pred}\n{timestamp} - Interval + Zero: {zero_prediction +  interval_pred}\n{timestamp} - Recorded Value: {actual.loc[timestamp, "interval_energy"]}")
        preds.loc[timestamp] = interval_pred + zero_prediction
        bar.update(bar.value+1)
    bar.finish('\n')
    return preds

def read_data(measurementsPath, targetsPath, scaler):
    x = pd.read_parquet(measurementsPath)
    y = pd.read_parquet(targetsPath)

    if scaler:
        bar = progressbar.ProgressBar(max_value=len(x.groupby(level=0)), widgets=['Scaling data:', ' ', progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ', progressbar.Timer()], redirect_stdout=True)

        for timestamp, interval_df in x.groupby(level=0):
            interval_df = interval_df.droplevel(0)
            x.loc[timestamp].iloc[:,:] = scaler.transform(interval_df)
            bar.update(bar.value+1)
        bar.finish('\n')
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