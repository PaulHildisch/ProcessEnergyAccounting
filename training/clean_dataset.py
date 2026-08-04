import pandas as pd
import argparse

# Use actual module provided by Johannes 
from preprocessor import Preprocessor

def main(args: tuple) -> None:
    filename = args.filepath.split('.')[0]
    features = args.features

    df = pd.read_parquet(args.filepath)
    preprocessor = Preprocessor(df, features)

    df_agg, interval_energy_all, t, df_unagg = preprocessor.preprocess_no_split()
    interval_energy_all = pd.DataFrame(interval_energy_all, index=t)

    if args.pid_split:
        df_unagg = df_unagg.reset_index().set_index(["_time","pid"])[features + ['pid_label', 'base_name']]
        df_unagg = df_unagg[(df_unagg[features] > 0).any(axis=1)]
        df_unagg.to_parquet(f"{filename}-preprocessed-pid.parquet")
        print(f"Saved unscaled, per pid data to \"{filename}-preprocessed-pid.parquet\"")
    else:
        df_agg.to_parquet(f"{filename}-preprocessed-aggregated.parquet")
        print(f"Saved unscaled, aggregated data to \"{filename}-preprocessed-aggregated.parquet\"")

    interval_energy_all.to_parquet(f"{filename}-preprocessed-targets.parquet")
    print(f"Saved unscaled target values (interval energy) to \"{filename}-preprocessed-targets.parquet\"")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", required=True)
    parser.add_argument("--features", 
                        nargs='+', 
                        type=str,
                        default=['context_switches', 'syscall_class_network', 'delta_branch_instructions', 'syscall_class_time'])
    parser.add_argument("--pid-split", action="store_true", default=False)


    args = parser.parse_args()
    main(args)