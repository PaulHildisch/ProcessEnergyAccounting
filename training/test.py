import pandas as pd
import pyarrow as pa
from pyarrow import parquet


# for i in range(1,5):
#     filename = f"data/process_interval_data-{i}-batch-cleaned.parquet"
#     print(f"opening {filename}")
#     batched = parquet.ParquetFile(filename)
#     print(f"Found {batched.metadata.num_rows / 60 / 60} Hours in dataset")

filename = f"data/process_interval_data-4-cleaned.parquet"
print(f"opening {filename}")
batched = parquet.ParquetFile(filename)
print(f"Found {batched.metadata.num_rows / 60 / 60} Hours in dataset")
batched_it = batched.iter_batches()

# print('-'*10, " Batch Processed ", '-'*10)
# for batch in batched_it:
#     df = batch.to_pandas().set_index('_time')
#     print(df[:1])
#     print(df[-1:])
# raw = pd.read_parquet('data/process_interval_data-4-cleaned.parquet')
# print('-'*10, " In Memory ", '-'*10)
# print(raw[:5])

