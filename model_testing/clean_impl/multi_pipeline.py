from model_builder import ModelBuilder
from preprocessing import Preprocessor
from plotting import Plotter
from plotting import plot_dataset
#from shapley import ProcessAttributor
from shapley_improved import ProcessAttributorSHAP
from shapley_improved import ProcessAttributorLinear
from universal_filtering import CustomSpearmanFilter
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


# from xgboost import XGBRFRegressor
# from interpret.glassbox import ExplainableBoostingRegressor

short_data = [
    # pd.read_parquet("runs/nfcore-20260630T142308Z/datasets/rnaseq1_shorttest.parquet"),
    # pd.read_parquet("runs/nfcore-20260630T143512Z/datasets/chipseq1_shorttest.parquet"),
    # pd.read_parquet("runs/nfcore-20260630T152039Z/datasets/methlyseq1_shorttest.parquet"),
    # pd.read_parquet("runs/nfcore-20260630T152447Z/datasets/methylseq2_shorttest.parquet"),
    #pd.read_parquet("runs/nfcore-20260630T153034Z/datasets/sarek1_shorttest.parquet"),
    # pd.read_parquet("runs/nfcore-20260630T153034Z/datasets/sarek1_shorttest.parquet"),
    # pd.read_parquet("runs/nfcore-20260630T153801Z/datasets/sarek2_short_test.parquet"),
    #pd.read_parquet("data/siena12/stressng_siena12.parquet")
    
    #LOCAL Test profiles
    # pd.read_parquet("data/siena12/test/sarek_1.parquet"),
    # pd.read_parquet("data/siena12/test/rnaseq_siena12.parquet"),
    # pd.read_parquet("data/siena12/test/chip_seq_1.parquet")

]

data = [
    pd.read_parquet("data/siena12/full_test/ampliseq_1_0607.parquet"),
    # pd.read_parquet("data/siena12/full_test/ampliseq_2_0607.parquet"),
    #pd.read_parquet("data/siena12/full_test/ampliseq_3_0707.parquet")
    #pd.read_parquet("runs/stressng-custom-1782744477/datasets/process_interval_data.parquet")
    #pd.read_parquet("data/siena12/test/rnaseq_siena12.parquet"),
    #pd.read_parquet("runs/nfcore-20260701T114734Z/datasets/rnaseq_1_02027.parquet"),
    #pd.read_parquet("runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"),
    #pd.read_parquet("runs/nfcore-20260704T110043Z/datasets/chipseq_2_0607.parquet"),
    #pd.read_parquet("runs/nfcore-20260703T215123Z/datasets/ampliseq_1_0607.parquet"),
    #pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet"),
    #pd.read_parquet("runs/nfcore-20260706T112716Z/datasets/ampliseq_3_0707.parquet")
   # pd.read_parquet("data/siena12/full_test/ampliseq_triple_run.parquet")
]



data = pd.concat(data, ignore_index=True)

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
X_train, y_train, t_train, _ = preprocessor_train.preprocess_no_split()

#Params could be tuned as well
model = RandomForestRegressor(n_estimators=100,  n_jobs=-1, random_state=42)
#model = Ridge(alpha=1.0)
#model = Lasso(alpha=0.1)

#The constraints do not change the result, then why is the xgbrf slighty worse?
#Apparently these constraints dont even give us the necessary guarantees
# constraints = (1, 1, 0)
# build_model = XGBRFRegressor(
#     n_estimators=100,
#     monotonic_constraints=constraints,
#     random_state=42,
#     max_depth=0, # 0 means no limit in XGBoost (matches sklearn)
#     #tree_method='exact',
# )

#build_model = ExplainableBoostingRegressor( interactions=2, max_rounds=2000, n_jobs=-1, random_state=42)
#-------------------------------------------------------------------------------------------------------------


#These thresholds could be fine tuned
automatic_feature_selection = Pipeline(steps=[
    ('variance', VarianceThreshold(threshold=0.01)), #explain this

    ('decorrelate', CustomSpearmanFilter(threshold=0.80)),
 
    ('select_features', SelectFromModel(model, threshold='0.5*median'))
])

automatic_feature_selection.set_output(transform="pandas")
X_train = automatic_feature_selection.fit_transform(X_train, y_train)
good_features = X_train.columns.tolist()
print("Selected columns:")
print(good_features)

plot_dataset(t_train, y_train, "multi_training")


#test_data = pd.read_parquet("data/siena12/sarek_1.parquet")
#test_data  = pd.read_parquet("data/siena12/sarek_2_0207.parquet")
#test_data = pd.read_parquet("runs/nfcore-20260702T193504Z/datasets/sarek_2_0207.parquet")
#test_data = pd.read_parquet("runs/nfcore-20260704T110043Z/datasets/chipseq_2_0607.parquet")
#test_data = pd.read_parquet("runs/nfcore-20260702T072031Z/datasets/chip_seq_0207.parquet")
#test_data = pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet")
#test_data = pd.read_parquet("data/siena12/stressng_siena12.parquet")
#test_data = pd.read_parquet("data/siena12/test/sarek_2.parquet")
#test_data = pd.read_parquet("runs/stressng-custom-1782744477/datasets/process_interval_data.parquet")
test_data = pd.read_parquet("data/siena12/full_test/ampliseq_3_0707.parquet")

preprocessor_test = Preprocessor(test_data, good_features)
X_test, y_test, t_test , X_test_unaggregated = preprocessor_test.preprocess_no_split()

plot_dataset(t_test, y_test, "multi_testing")


#idle_power_isactually idle interval energy
builder = ModelBuilder(X_train, X_test, y_train, y_test, model, StandardScaler())
y_pred, learned_idle_power = builder.run_and_save_model(".")


plotter = Plotter(y_pred,y_test, t_test)#, window_start =50, window_end=200)
plotter.plot_and_save("", "multi__pred_lasso")





#check if we ann pass this differently
#attributor = ProcessAttributorSHAP( builder.X_test_scaled, builder.model, builder.scaler)
#attributor.attribute(X_test_unaggregated,good_features,t_test.values)


#Nur zum Test -> eigentlich Pauls Aufgabe
# attributor = ProcessAttributorLinear(  builder.model, builder.scaler)
# attributor.attribute(X_test_unaggregated,good_features,t_test.values)



