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

def plot(zero_prediction, prediction, actual, range = None, title="L2 Regression - Actual & Predicted Energy Consumption"):
    predictions_to_plot = prediction.swaplevel().sort_index()
    empty_df = pd.DataFrame(0, index=actual.index, columns=actual.columns)
    zero_prediction_df = pd.DataFrame(zero_prediction, index=actual.index, columns=actual.columns)
    datap = dict({'0':zero_prediction_df})
    bar = progressbar.ProgressBar(max_value=len(predictions_to_plot.groupby(level=0)), widgets=['Filling missing datapoint:', ' ', progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ', progressbar.Timer()], redirect_stdout=True)
    for pid, pid_df in predictions_to_plot.groupby(level=0):
        pid_df = pid_df.droplevel(0)
        padded_pid_df = pid_df.combine_first(empty_df)
        datap.update({pid: padded_pid_df})
        bar.update(bar.value + 1)
        # print(padded_pid_df.shape[0])
        # # padded_pid_df = padded_pid_df[padded_pid_df.columns[0]].tolist()
        # # predictions_to_plot = pd.concat([predictions_to_plot, padded_pid_df])
        # predictions_to_plot = predictions_to_plot.loc[pid, :].reindex(empty_df.index) 
        # print(predictions_to_plot)
        # print(predictions_to_plot.loc[pid].shape[0])
    bar.finish('\n')
    # [print(f"{pid}: {pid_prediction_df.loc[pid].shape[0]}") for pid, pid_prediction_df in predictions_to_plot.groupby(level=0)]
    #time_frame_np = np.arange(actual.index.values[0], actual.index.values[-1], 1, dtype='datetime64[s]')
    #print(time_frame_np.size)
    print(f"Actual has: {actual.shape[0]}")
    print(f"Empty DF has: {empty_df.shape[0]}")
    print(f"Preds for pid 1 has: {predictions_to_plot.loc['1'].shape[0]}")

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
    ax.stackplot(
        actual.index.values[start:end],
        [df['interval_energy'][start:end] for df in datap.values()],
        baseline="zero",
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel("Time", fontsize=10, labelpad=4)
    ax.xaxis.set_major_locator(dates.SecondLocator(interval=int(range/3)))
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
    ax.set_ylabel("Interval Energy (Ws)", fontsize=12, labelpad=4)
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
    preds = pd.DataFrame(0, index=data.index, columns=actual.columns).astype(np.float64)
    bar = progressbar.ProgressBar(max_value=len(data.groupby(level=0)), widgets=['Making predictions:', ' ', progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ', progressbar.Timer()], redirect_stdout=True)
    for timestamp, interval_df in data.groupby(level=0):
        interval_df = interval_df.droplevel(0)
        predictions = model.predict(interval_df)
        preds.loc[timestamp].iloc[:,:] = [prediction - zero_prediction for prediction in predictions]
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

    zero_df = pd.DataFrame(scaler.transform(np.zeros((1, model.n_features_in_))))
    zero_prediction = model.predict(zero_df)
    data, actual = read_data(args.pidDataSource, args.targetDataSource, scaler)

    prediction = predictions(data, model, zero_prediction, actual)
    # evaluate_model(prediction, actual)
    
    if args.full:
        plot(zero_prediction[0], prediction, actual)
    else:
        plot(zero_prediction[0], prediction, actual, range=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--modelFile")
    parser.add_argument("--targetDataSource")
    parser.add_argument("--pidDataSource")
    parser.add_argument("--full", action="store_true", default=False)

    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args)