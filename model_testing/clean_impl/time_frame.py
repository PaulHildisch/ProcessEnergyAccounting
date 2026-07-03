import polars as pl
import pandas as pd

#file_path = "data/nfcore/process_interval_data_1tag.parquet"
#file_path = "data/gpu08/gpu08_dataset.parquet"
file_path = "data/siena/siena06_dataset.parquet"

start_offset = 0
end_offset = 180
#name = "data/gpu08_short_"+str(start_offset)+"_"+str(end_offset)+"_.parquet"
name = "data/siena/siena06_short_"+str(start_offset)+"_"+str(end_offset)+"_.parquet"
# 1. Peek at the file to find the very first timestamp (without loading the file!)
min_time = pl.scan_parquet(file_path).select(pl.col("_time").min()).collect().item()
print(f"Dataset starts at: {min_time}")

if isinstance(min_time, str):
    start_time = pd.to_datetime(min_time)
else:
    start_time = min_time

start_time = start_time+ pd.Timedelta(minutes=start_offset)
end_time = start_time + pd.Timedelta(minutes=end_offset)

# 2. Lazily scan, filter, and collect ONLY the first 30 minutes, 
# then convert it straight back to a Pandas DataFrame for your existing pipeline.
data = (
    pl.scan_parquet(file_path)
    .filter(
        (pl.col("_time") >= start_time) & 
        (pl.col("_time") <= end_time)
    )
    .collect()       # This actually executes the read and pulls it into memory
    .to_pandas()     # Hand it back to Pandas
)

data.to_parquet(name)

print(f"Saved {len(data)} rows successfully!")