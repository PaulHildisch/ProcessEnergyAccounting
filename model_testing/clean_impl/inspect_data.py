from plotting import plot_dataset
from preprocessing import Preprocessor
import pandas as pd



data = {

    "rnaseq_1_0207" : pd.read_parquet("runs/nfcore-20260701T114734Z/datasets/rnaseq_1_02027.parquet"),
    "sarek_1_0207": pd.read_parquet("runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"),
    "sarek_2_0207" : pd.read_parquet("runs/nfcore-20260702T193504Z/datasets/sarek_2_0207.parquet"),
    "chipseq_1_0207": pd.read_parquet("runs/nfcore-20260702T072031Z/datasets/chip_seq_0207.parquet"),
    "chipseq_2_0607" : pd.read_parquet("runs/nfcore-20260704T110043Z/datasets/chipseq_2_0607.parquet"),
    "ampliseq_1_0607" : pd.read_parquet("runs/nfcore-20260703T215123Z/datasets/ampliseq_1_0607.parquet"),
    "ampliseq_2_0607" : pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet"),

}

features = [
    "delta_cpu_ns",
    "delta_io_bytes",
    "delta_net_send_bytes",
    "context_switches",
    "syscall_count",
    "delta_rss_memory",
    "delta_cpu_time_psutil",
    "delta_cpu_time_proc",
    "syscall_class_file",
    "syscall_class_network",
    "syscall_class_memory",
    "syscall_class_process",
    "syscall_class_other",
    "syscall_class_sched",
    "syscall_class_signal",
    "syscall_class_time",
    "delta_cycles",
    "delta_cache_misses",
    "delta_instructions",
    "delta_branch_instructions",
]





for name , df in data.items():
    print("process ", name)
    print(df.head(3))
    preprocessor_train = Preprocessor(df,features )
    _, y_train, t_train = preprocessor_train.preprocess_no_split()
    plot_dataset(t_train, y_train, "data_inspection/" +name +".png")