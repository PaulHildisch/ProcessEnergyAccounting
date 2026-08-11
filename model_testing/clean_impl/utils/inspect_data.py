from model_testing.clean_impl.plotting.plotting import plot_dataset
from model_testing.clean_impl.pipeline.preprocessing import Preprocessor
import pandas as pd
#Utils file in order to plot datasets without making a prediction
#This allows to visualize a recorded dataset wihtout having to train a model
#Can be used for visual inspection to identify obvious outliers or errors during the data collection process

#Choose datasets that should be visualized
# data = {

#     "rnaseq_1_0207" : pd.read_parquet("runs/nfcore-20260701T114734Z/datasets/rnaseq_1_02027.parquet"),
#     "sarek_1_0207": pd.read_parquet("runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"),
#     "sarek_2_0207" : pd.read_parquet("runs/nfcore-20260702T193504Z/datasets/sarek_2_0207.parquet"),
#     "chipseq_1_0207": pd.read_parquet("runs/nfcore-20260702T072031Z/datasets/chip_seq_0207.parquet"),
#     "chipseq_2_0607" : pd.read_parquet("runs/nfcore-20260704T110043Z/datasets/chipseq_2_0607.parquet"),
#     "ampliseq_1_0607" : pd.read_parquet("runs/nfcore-20260703T215123Z/datasets/ampliseq_1_0607.parquet"),
#     "ampliseq_2_0607" : pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet"),

# }
data =  {
    # "amp1" : pd.read_parquet("runs/cpu06 daten/altes monitoring/cpu06_ampliseq1_old_mon.parquet"),
    # "amp2" : pd.read_parquet("runs/cpu06 daten/altes monitoring/cpu06_ampliseq2_old_mon.parquet"),
    # "amp3" : pd.read_parquet("runs/cpu06 daten/altes monitoring/cpu06_ampliseq3_old_mon.parquet"),
    # "amp4" : pd.read_parquet("runs/cpu06 daten/altes monitoring/cpu06_ampliseq4_old_mon.parquet"),
    # "amp5" : pd.read_parquet("runs/cpu06 daten/altes monitoring/cpu06_ampliseq5_old_mon.parquet"),
    # "amp6" : pd.read_parquet("runs/cpu06 daten/altes monitoring/cpu06_ampliseq6_old_mon.parquet"),
    # "chip" : pd.read_parquet("runs/cpu06 daten/altes monitoring/cpu06_chiseq_old_mon.parquet"),
    # "rnaseq" : pd.read_parquet("runs/cpu06 daten/altes monitoring/cpu06_rnaseq_old_mon.parquet"),
    # "sarek1" :pd.read_parquet("runs/cpu06 daten/altes monitoring/cpu06_sarek1_old_mon.parquet"),
    # "sarek2" : pd.read_parquet("runs/cpu06 daten/altes monitoring/cpu06_sarek2_old_mon.parquet"),
    # "sarek3" : pd.read_parquet("runs/cpu06 daten/altes monitoring/cpu06_sarek3_old_mon.parquet"),
    # "sarek_new_feat" : pd.read_parquet("runs/cpu06 daten/new monitoring/cpu06_sarek1_new_mon.parquet")
    "sarek_new_feat" : pd.read_parquet("sarek1.parquet")
}
#Not important for the actual visualization but we make use of our Preprocessing class which expects features
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
    print("Dataset: ", name)
    print(df.head(3))
    preprocessor_train = Preprocessor(df,features)
    #Preprocess entire dataset
    _, y_train, t_train, _ = preprocessor_train.preprocess_no_split()
    plot_dataset(t_train, y_train, "data_inspection/cpu06/" +name +".png")