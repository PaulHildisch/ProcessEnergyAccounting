import pandas as pd
import argparse
import gc
from sklearn.preprocessing import (StandardScaler)
from sklearn.model_selection import (train_test_split)
from pyarrow import parquet
from pyarrow import RecordBatch
import pyarrow as pa
import progressbar

def main(args):
    filename = args.filepath.split('.')[0]

    inFile = parquet.ParquetFile(args.filepath)
    in_schema = inFile.schema_arrow
    out_schema = pa.schema([in_schema.field('_time'), in_schema.field('interval_energy')])

    rows_per_batch = 1024
    total_iterations = int(inFile.metadata.num_rows / rows_per_batch)
    inFile_it = inFile.iter_batches(batch_size=rows_per_batch)
    
    # To make this dynamic we have to save the features used to train the model and read them here.
    # features = ["delta_cpu_ns", "delta_cycles", "delta_instructions", "delta_cache_misses", "delta_branch_instructions", "delta_io_bytes", "delta_net_send_bytes", "context_switches", "syscall_count", "delta_rss_memory", "syscall_class_file", "syscall_class_network", "syscall_class_memory", "syscall_class_process", "syscall_class_other", "syscall_class_sched", "syscall_class_signal", "syscall_class_time",]
    features = ['context_switches', 'syscall_class_network', 'delta_branch_instructions', 'syscall_class_time']
    if args.features:
        print(f"--features is not implemented. Using hardcoded values: ({features})")
    for feature in features:
        if feature not in in_schema.names:
            print(f"Feature {feature} is not present in dataset. Removing from selection.")
            features.remove(feature)
            continue
        out_schema = out_schema.append(in_schema.field(feature))
        
    with parquet.ParquetWriter(f'{filename}-batch-cleaned.parquet', schema=out_schema) as writer:
        bar = progressbar.ProgressBar(max_value=total_iterations, widgets=[progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ', progressbar.Timer()], redirect_stdout=True)
        for batch in inFile_it:

            df = pd.DataFrame(batch.to_pandas())
            #print(df.shape)

            df["_time"] = pd.to_datetime(df["_time"]).dt.round("1ms")

            df[features] = df[features].fillna(0)

            interval_energy_all = (
                df[["_time", "interval_energy"]]
                .dropna()
                .drop_duplicates("_time")
                .set_index("_time")["interval_energy"]
            )
            df = df[df["_time"].isin(interval_energy_all.index)]

            interval_energy_all = interval_energy_all.sort_index()

            #aggregation
            df_agg = df.groupby("_time")[features].sum()
            out = pd.concat([interval_energy_all, df_agg], axis=1)
            batch_out = RecordBatch.from_pandas(out, schema=out_schema, preserve_index=True)
            writer.write_batch(batch_out)
            bar.update(min(bar.value+1, bar.max_value))
        bar.finish('\n')

    print(f"Saving unscaled dataset with features {features} to \"{filename}-batch-cleaned.parquet\"")
    print("[!] Remember to use .set_index(\"_time\") when using the dataset to use time as the index")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath")
    parser.add_argument("--features")
    parser.add_argument("--pid-split", action="store_true", default=False)

    args = parser.parse_args()
    main(args)