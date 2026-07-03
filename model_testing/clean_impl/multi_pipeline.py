from model_builder import ModelBuilder
from preprocessing import Preprocessor
from plotting import Plotter
from plotting import plot_dataset
from shapley import ProcessAttributor
import pandas as pd


#data1 = pd.read_parquet("data/siena12/rnaseq_siena12.parquet")
#data2 = pd.read_parquet("data/siena12/chip_seq_1.parquet")
# data3 = pd.read_parquet("data/siena12/methylseq_1.parquet")
# data1 = pd.read_parquet("data/nfcore_gpu06/nf_rna_1.parquet")
# data2 = pd.read_parquet("data/nfcore_gpu06/nf_rna_2.parquet")

#data = pd.concat([data1,data2], ignore_index=True)
#data = pd.read_parquet("data\gpu08\gpu08_dataset.parquet")
data = pd.read_parquet("data/siena12/rnaseq_1_02027.parquet")

#chipseq -> very goog
#['delta_cpu_ns', 'syscall_class_other', 'delta_io_bytes', 'delta_net_send_bytes']

#methylseq1 -> not so high r² -> maybe wrong export?
#['delta_net_send_bytes', 'syscall_class_signal', 'context_switches', 'syscall_class_memory', 'syscall_class_sched', 'syscall_class_time']
#methlyseq2 -> not so high r²
#good_features =['syscall_class_signal', 'syscall_class_network', 'syscall_class_time']

#sarek1
#['syscall_class_file', 'delta_io_bytes', 'syscall_class_network', 'context_switches', 'syscall_class_time', 'syscall_class_sched']

#sarek2 -> check export here?
#good_features =['syscall_class_memory']

#seems to work well for many things
good_features =['syscall_class_other', 'syscall_class_signal', 'context_switches', 'syscall_class_process'] \
    + ['delta_cpu_ns', 'syscall_class_time', 'delta_rss_memory', 'delta_cycles']



preprocessor_train = Preprocessor(data, good_features)
X_train, y_train, t_train = preprocessor_train.preprocess_no_split()

plot_dataset(t_train, y_train)


test_data = pd.read_parquet("data/siena12/sarek_1.parquet")
test_data  = pd.read_parquet("data/siena12/sarek_2_0207.parquet")
preprocessor_test = Preprocessor(test_data, good_features)
X_test, y_test, t_test = preprocessor_test.preprocess_no_split()

plot_dataset(t_test, y_test)


#idle_power_isactually idle interval energy
builder = ModelBuilder(X_train, X_test, y_train, y_test, t_train, t_test)
y_pred, learned_idle_power = builder.run_and_save_model(".")


plotter = Plotter(y_pred,y_test, t_test)#, window_start =50, window_end=200)
plotter.plot_and_save("", "multi__pred")


df_original_test = preprocessor_test.df[preprocessor_test.df["_time"].isin(t_test)].copy()
df_original_test = df_original_test.set_index("_time")
attributor = ProcessAttributor(builder.model, X_test, learned_idle_power)
#pretty sure this df_original is wrong 
attributor.attribute(y_pred,df_original_test,good_features,t_test.values)



