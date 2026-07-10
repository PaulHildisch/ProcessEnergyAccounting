from model_builder import ModelBuilder
from preprocessing import Preprocessor
from plotting import Plotter
from plotting import plot_dataset
from shapley import ProcessAttributor
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn import linear_model

#TODO idle prediction is heavily dependent of features

#data  = pd.read_parquet("data/single_benchmarks/benchmark_primesieve.parquet")
#data = pd.read_parquet("data/new_node/stressng_siena12.parquet")
#data = pd.read_parquet("data/gpu06/stressng_gpu06.parquet")
#data = pd.read_parquet("data/gpu06/stressng_gpu06_L1.parquet")

#data = pd.read_parquet("data/nfcore/nf_rna_1.parquet")

#data  = pd.read_parquet("data/single_benchmarks/clean_benchmark_dbench.parquet")
#data  = pd.read_parquet("data/process_interval_data_wf.parquet")
#data  = pd.read_parquet("data/gpu08/gpu08_short_60_30_.parquet")
#data = pd.read_parquet("data/nfcore/process_interval_data_1tag.parquet")
#data  = pd.read_parquet("data/single_benchmarks/benchmark_coremark.parquet")
#data  = pd.read_parquet("data/siena/siena_short_40_min.parquet")

#data  = pd.read_parquet("data/new_node/sien06_stressng.parquet")
#siena 60
#good_features = ['delta_cycles', 'syscall_class_file', 'delta_cpu_ns']
#good_features = ['syscall_class_other', 'syscall_class_process']
#good_features = ['delta_cpu_ns', 'syscall_class_network', 'syscall_class_sched']

#rna
#good_features =   ['context_switches', 'syscall_class_network']
#siena short
#good_features = ['delta_instructions', 'delta_cycles', 'delta_cache_misses', 'delta_net_send_bytes', 'delta_io_bytes']
#good_features = ['context_switches', 'syscall_class_network', 'syscall_class_sched', 'delta_cycles', 'delta_branch_instructions', 'syscall_class_signal']

data = pd.read_parquet("runs/nfcore-20260702T193504Z/datasets/sarek_2_0207.parquet")
good_features = ['delta_cpu_ns', 'delta_io_bytes', 'delta_net_send_bytes', 'context_switches', 'syscall_count', 'delta_rss_memory', 'delta_cpu_time_proc', 'syscall_class_file', 'syscall_class_network', 'syscall_class_memory', 'syscall_class_other', 'syscall_class_signal']

#primesieve
#good_features =   ['context_switches', 'delta_cache_misses']
#gpu 60 -> 30
#good_features = ['delta_instructions', 'syscall_class_memory', 'delta_net_send_bytes', 'context_switches']

#model = RandomForestRegressor(n_estimators=100,  n_jobs=-1, random_state=42)
model = Lasso(alpha=0.1)
preprocessor = Preprocessor(data, good_features)
#as data frames with columns
X_train, X_test, y_train, y_test, t_train, t_test = preprocessor.preprocess()

plot_dataset(preprocessor.interval_energy_all.index, preprocessor.interval_energy_all)

#print(X_train.columns)
df_original_test = preprocessor.df[preprocessor.df["_time"].isin(t_test)].copy()
df_original_test = df_original_test.set_index("_time")


#Seems to work just fine without using the explicit vlaues like in the ProcessAccountig Example
#builder = ModelBuilder(X_train.values, X_test.values, y_train.values, y_test.values, t_train.values, t_test.values)
builder = ModelBuilder(X_train, X_test, y_train, y_test, model, StandardScaler())
y_pred, learned_idle_power = builder.run_and_save_model("model_testing/clean_impl")


plotter = Plotter(y_pred,y_test, t_test)#, window_start =50, window_end=200)
plotter.plot_and_save("", "single_pipe__pred")


# attributor = ProcessAttributor(builder.model,builder.X_test_scaled, learned_idle_power)
# #pretty sure this df_original is wrong 
# attributor.attribute(y_pred,df_original_test,good_features,t_test.values)


