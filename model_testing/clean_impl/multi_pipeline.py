from model_builder import ModelBuilder
from preprocessing import Preprocessor
from plotting import Plotter
from plotting import plot_dataset
from shapley import ProcessAttributor
from universal_filtering import CustomSpearmanFilter
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

#This combiantion predicts sarek2 better than just sarek1 alone

data = [
    pd.read_parquet("data/siena12/test/rnaseq_siena12.parquet"),
    pd.read_parquet("data/siena12/test/chip_seq_1.parquet"),
    pd.read_parquet("data/siena12/test/methylseq_1.parquet"),
    pd.read_parquet("data/siena12/test/sarek_2.parquet")
]



data = pd.concat(data, ignore_index=True)
#data = pd.read_parquet("data\gpu08\gpu08_dataset.parquet")
#data = pd.read_parquet("data/siena12/rnaseq_1_02027.parquet")

#data = pd.read_parquet("data/siena12/test/sarek_2.parquet")

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
#good_features =['syscall_class_other', 'syscall_class_signal', 'context_switches', 'syscall_class_process'] \
#    + ['delta_cpu_ns', 'syscall_class_time', 'delta_rss_memory', 'delta_cycles']

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



preprocessor_train = Preprocessor(data, features)
X_train, y_train, t_train = preprocessor_train.preprocess_no_split()

#Params could be tuned as well
model = RandomForestRegressor(n_estimators=100,  n_jobs=-1, random_state=42)
#model = Ridge(alpha=1.0)

#These thresholds could be fine tuned
automatic_feature_selection = Pipeline(steps=[
    ('variance', VarianceThreshold(threshold=0.01)),

    ('decorrelate', CustomSpearmanFilter(threshold=0.90)),
 
    ('select_features', SelectFromModel(model, threshold='0.5*median'))
])

automatic_feature_selection.set_output(transform="pandas")
X_train = automatic_feature_selection.fit_transform(X_train, y_train)
good_features = X_train.columns.tolist()
print("Selected columns:")
print(good_features)

plot_dataset(t_train, y_train)


#test_data = pd.read_parquet("data/siena12/sarek_1.parquet")
#test_data  = pd.read_parquet("data/siena12/sarek_2_0207.parquet")
test_data = pd.read_parquet("data/siena12/test/sarek_1.parquet")
preprocessor_test = Preprocessor(test_data, good_features)
X_test, y_test, t_test = preprocessor_test.preprocess_no_split()

plot_dataset(t_test, y_test)


#idle_power_isactually idle interval energy
builder = ModelBuilder(X_train, X_test, y_train, y_test, model, StandardScaler())
y_pred, learned_idle_power = builder.run_and_save_model(".")


plotter = Plotter(y_pred,y_test, t_test)#, window_start =50, window_end=200)
plotter.plot_and_save("", "multi__pred")


# df_original_test = preprocessor_test.df[preprocessor_test.df["_time"].isin(t_test)].copy()
# df_original_test = df_original_test.set_index("_time")
# attributor = ProcessAttributor(builder.model, X_test, learned_idle_power)
# #pretty sure this df_original is wrong 
# attributor.attribute(y_pred,df_original_test,good_features,t_test.values)



