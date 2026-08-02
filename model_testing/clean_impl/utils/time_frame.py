import polars as pl
import pandas as pd
#This is a utils file that allows to extract a smaller subset from a large dataset for testing purposes 

#Path to the larger set
file_path =  "runs/stressng-custom-1782744477/datasets/process_interval_data.parquet"

#Choose start and end offsets e.g extract minute 3-10 from the larger dataset
start_offset = 3
end_offset = 10

#Choose name/path of the output file
name = "stressng_test"+str(start_offset)+"_"+str(end_offset)+"_.parquet"

min_time = pl.scan_parquet(file_path).select(pl.col("_time").min()).collect().item()
print(f"Dataset starts at: {min_time}")

if isinstance(min_time, str):
    start_time = pd.to_datetime(min_time)
else:
    start_time = min_time

start_time = start_time+ pd.Timedelta(minutes=start_offset)
end_time = start_time + pd.Timedelta(minutes=end_offset)

data = (
    pl.scan_parquet(file_path)
    .filter(
        (pl.col("_time") >= start_time) & 
        (pl.col("_time") <= end_time)
    )
    .collect()       
    .to_pandas()     
)

data.to_parquet(name)

print(f"Saved {len(data)} rows successfully!")