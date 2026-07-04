#! /usr/bin/python3

import pandas as pd
import numpy as np
import joblib
import time
import pickle
import argparse
import progressbar

from pyarrow import parquet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def prepare_dataset(df, scaler):
    x = df[df.columns[1:]]
    y = df[df.columns[0]]

    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.2, shuffle=False
    # )

    x = scaler.transform(x)
    # x_test_scaled = scaler.transform(x_test)

    return {"x": x, "y": y}

def evaluate_model(model, training_data):
    #evaluate
    prediction = model.predict(training_data["x"])
    
    r2 = r2_score(training_data["y"], prediction)
    mae = mean_absolute_error(training_data["y"], prediction)

    return r2, mae

def main(args):
    print(f"opening {args.dataSource}")
    data_stream = parquet.ParquetFile(args.dataSource)
    total_rows = data_stream.metadata.num_rows
    train_rows = int(total_rows * 0.8)
    seen_rows = 0
    test_rows = total_rows - train_rows

    # TODO: This potentially results in one more batch than intended via the iterations parameter
    batch_size = int(data_stream.metadata.num_rows / args.iterations)
    print(f"Batchsize: {batch_size} for {args.iterations} iterations")
    data_it = data_stream.iter_batches(batch_size=batch_size)
    selected_features = data_stream.schema_arrow.names 
    
    print("Training new Model...")
    # model=KernelRidge(alpha=1.0, kernel='rbf')
     
    # Maybe play around with alpha?
    sgd_model = SGDRegressor(loss="squared_error", penalty="l2", shuffle=False)

    print("Preparing Dataset...")
    scaler = StandardScaler()
    for batch in progressbar.progressbar(data_it, max_value=args.iterations, prefix="Fitting scaler: "):
        if seen_rows > train_rows:
            break
        df = batch.to_pandas().set_index('_time')
        x = df[df.columns[1:]]
        scaler = scaler.partial_fit(x)
        seen_rows += batch.num_rows

    print("Saving scaler used for model...")
    with open(f"models/l2-regression-{timestamp}-scaler.npy", "w+b") as scaler_out:
        scaler_out.write(pickle.dumps(scaler))
        print(f"Saved scaler to {scaler_out.name}")

    # Reset iterator after scaling
    data_it = data_stream.iter_batches(batch_size=batch_size)
    seen_rows = 0

    print(f"Training model with features {selected_features}")
    for batch in data_it:
        if seen_rows >  train_rows:
            break
        df = batch.to_pandas().set_index('_time')
        data_set = prepare_dataset(df, scaler)
        sgd_model.partial_fit(data_set["x"], data_set["y"])
        seen_rows += batch.num_rows

    print("Saving Model...")
    timestamp = time.strftime("%m%d%H%M%S")
    outpath = joblib.dump(sgd_model, f"models/l2-regression-{timestamp}", compress=3)
    print(f"Model saved to {outpath}")


    print("Evaluating Model...")
    # TODO: Find a way to evaluate performance over many iterations
    # predictions = pd.DataFrame()
    # for batch in data_it:
    #     df = batch.to_pandas().set_index('_time')
    #     data_set = prepare_dataset(df, scaler)
    #     prediction = sgd_model.predict(data_set["x"])
    #     predictions.concat([predictions, prediction], axis=1)
    
    # r2 = r2_score(training_data["y"], prediction)
    # mae = mean_absolute_error(training_data["y"], prediction) 
    # mae_percent = 100 * mae / data_set["y"].mean()

    # print("-" * 34)
    # print(f"  R² Score:  {r2:.4f}")
    # print(f"  MAE:       {mae:.2f} Wh ({mae_percent:.2f}%)")
    # print("-" * 34)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataSource", required=True)
    parser.add_argument("--iterations", default=5000, type=int)

    args = parser.parse_args()

    main(args)