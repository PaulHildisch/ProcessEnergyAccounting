import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def train_cvxpy_model(df: pd.DataFrame, features: list, l1_penalty=1.0):
    import cvxpy as cp
    from sklearn.preprocessing import StandardScaler

    df = df.copy()
    df["_time"] = pd.to_datetime(df["_time"])

    # Drop NaNs
    df[features] = df[features].fillna(0)
    df = df.dropna(subset=["interval_energy"])

    # Aggregate system-wide interval energy
    interval_energy = df.groupby("_time")["interval_energy"].first().reset_index()
    interval_energy.columns = ["_time", "interval_energy"]

    # Only keep process rows with matching interval energy
    df = df.merge(interval_energy, on="_time", suffixes=("", "_y"))
    df = df.rename(columns={"interval_energy_y": "interval_energy"})

    n_matched = len(df)
    n_intervals = len(interval_energy)
    print(f"Matching process intervals: {n_matched} across {n_intervals} time intervals "
          f"({n_matched / n_intervals:.2f} per interval on average, "
          f"{(n_matched / df.shape[0]) * 100:.2f}% of total process records)")

    # Build feature matrix
    X_matrix = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_matrix)
    time = df["_time"].values
    y = interval_energy["interval_energy"].values

    # Build mapping from time -> index
    time_to_idx = {t: i for i, t in enumerate(interval_energy["_time"].values)}
    interval_idx = np.array([time_to_idx[t] for t in df["_time"].values])

    # Aggregation matrix
    n_intervals = len(interval_energy)
    n_samples = len(df)
    A = np.zeros((n_intervals, n_samples))
    for i, idx in enumerate(interval_idx):
        A[idx, i] = 1

    # Variables
    w = cp.Variable(X_scaled.shape[1])
    static_energy = cp.Variable()
    preds = X_scaled @ w
    interval_preds = A @ preds + static_energy

    # Loss and problem
    loss = cp.sum_squares(interval_preds - y)
    reg = l1_penalty * cp.norm1(w)
    prob = cp.Problem(cp.Minimize(loss + reg))
    prob.solve()

    return {
        "weights": w.value,
        "scaler": scaler,
        "static_energy": static_energy.value,
    }

