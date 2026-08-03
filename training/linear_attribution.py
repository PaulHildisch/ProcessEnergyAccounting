import numpy as np
import pandas as pd
import joblib
import argparse
import warnings
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score

def preprocess(df_original, features):
    df_original["_time"] = pd.to_datetime(df_original["_time"]).dt.round("1ms")

    for feature in features:
        if feature not in df_original.columns:
            print(f"Feature {feature} was selected but is not present in dataset. Removing from selection.")
            features.remove(feature)

    df_original[features] = df_original[features].fillna(0)

    # Store actual values for evaluation
    interval_energy_all = (
        df_original[["_time", "interval_energy"]]
        .dropna()
        .drop_duplicates("_time")
        .set_index(["_time"])
    )
    df_original = df_original[df_original["_time"].isin(interval_energy_all.index)]

    interval_energy_all = interval_energy_all.sort_index()

    # store aggregated counters
    df_agg = df_original.groupby("_time")[features].sum()

    df_agg = pd.concat([interval_energy_all, df_agg], axis=1)
    return df_agg

def attribute(df_org, df_clean, features, weights, intercept, scaler):
    print(weights)
    weights = np.reshape(weights, [1,-1])
    # df_org[features] = scaler.transform(df_org[features])
    df_budgets = pd.DataFrame(list(weights), columns=df_clean.columns[1:], index=df_clean.index)
    df_org = df_org.set_index("_time")
    print(df_org[:5][features])

    df_ratios = df_org.loc[:, features].truediv(df_clean.iloc[:, 1:], axis=0).fillna(0)
    # print(df_ratios[(df_ratios > 0).any(axis=1)])
    print(df_ratios[:5])

    df_result = df_org.copy()
    df_ratio_mult = df_ratios.mul(df_budgets, axis=0)
    print(df_ratio_mult)
    df_result["attributed_dynamic_Wh"] = df_ratio_mult.sum(axis=1)
    print(df_result)

    df_test_plot = df_result.copy()

   # print(df_test_plot["attributed_dynamic_Wh"])

    # Merge process instances like wrk_99 and wrk_246 into one base name.
    df_test_plot["base_name"] = (
        df_test_plot["process_name"].str.replace(r"_\d+$", "", regex=True).str.strip()
    )
    df_test_plot.loc[df_test_plot["base_name"] == "", "base_name"] = "unknown"

    agg = (
        df_test_plot.groupby(["_time", "base_name"])["attributed_dynamic_Wh"]
        .sum()
        .reset_index()
    )

    # print(agg.groupby("_time")["attributed_dynamic_Wh"].sum())
    pivot = agg.pivot(
        index="_time", columns="base_name", values="attributed_dynamic_Wh"
    ).fillna(0)

    N = 8
    top_processes = pivot.max().sort_values(ascending=False).head(N).index
    pivot_top = pivot[top_processes].copy()

    if len(pivot.columns) > N:
        pivot_top["Other"] = pivot.drop(columns=top_processes).sum(axis=1)

    print((pivot_top < 0).sum())
    print("Fraction of intervals with negative values per process:")
    print((pivot_top < 0).mean())

    start_time = df_clean.index.min()
    end_time = df_clean.index.max()

    pivot_top_clipped = pivot_top.clip(lower=0)
    pivot_mask = (pivot_top_clipped.index >= start_time) & (
        pivot_top_clipped.index <= end_time
    )


    # ==== PLOTS: TOP BASE PROCESSES ====
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    pivot_top_clipped.loc[pivot_mask].plot.area(ax=ax, alpha=0.8, linewidth=0, legend=False)

    ax.set_xlabel("Time", fontsize=12, labelpad=4)
    ax.set_ylabel("Estimated Process Energy (Wh)", fontsize=12, labelpad=4)
    ax.tick_params(axis="both", labelsize=12)

    ax.set_title("Per-Process Energy Contribution Over Time", fontsize=13, pad=6)

    plt.tight_layout(pad=0.5)
    plt.savefig("per_process_energy_contribution.pdf", bbox_inches="tight")
    plt.savefig("per_process_energy_contribution.png", bbox_inches="tight", dpi=300)
    plt.show()
    

def split_and_scale(df, scaler):
    x = scaler.transform(df.iloc[:,1:])
    y = df.iloc[:,0].to_numpy()
    return x, y

def predict(actual, data, model):
    predictions = model.predict(data)
    r2 = r2_score(actual, predictions)
    mae = mean_absolute_error(actual, predictions)

    print("-" * 34)
    print(f"  R² Score:  {r2:.4f}")
    print(f"  MAE:       {mae:.2f} Wh ({100 * mae / actual.mean():.2f}%)")
    print("-" * 34)

    return predictions

def get_feature_weights(model):
    try:
        weights = model.coef_
        features = model.feature_names_in_
    except AttributeError as e:
        print(f"Could not find feature names in model. Falling back to numbered index")
        features = range(len(weights))
    return dict(zip(features, weights))

def main(args):
    modelFile = joblib.load(args.modelFile)
    df_org = pd.read_parquet(args.dataSource)
    features = ['delta_io_bytes', 'context_switches', 'delta_cpu_ns', 'delta_net_send_bytes', 'syscall_count']
    model = modelFile["model"]
    scaler = modelFile["scaler"]

    df_cleaned = preprocess(df_org, features=features)
    data, targets = split_and_scale(df_cleaned, scaler)

    predictions = predict(targets, data, model)

    attribute(df_org, df_cleaned, features, model.coef_, model.intercept_, scaler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--modelFile")
    parser.add_argument("--dataSource")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args)