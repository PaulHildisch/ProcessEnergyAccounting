from model_builder import ModelBuilder
from preprocessing import Preprocessor
from plotting import Plotter
from plotting import plot_dataset
#from shapley import ProcessAttributor
from shapley_improved import ProcessAttributorSHAP
from shapley_improved import ProcessAttributorEBM
from universal_filtering import CustomSpearmanFilter
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


# from xgboost import XGBRFRegressor
from interpret.glassbox import ExplainableBoostingRegressor

# Dropped 1 timestamps.
#  Random Forest
#   R² Score:  0.9182
#   MAE:       4.04 Wh (2.12% of mean)
#['delta_cpu_ns', 'delta_io_bytes', 'delta_net_send_bytes', 'context_switches', 'syscall_count', 'syscall_class_network', 'syscall_class_memory', 'syscall_class_sched']
train_ampliseq = [
        pd.read_parquet("runs/nfcore-20260703T215123Z/datasets/ampliseq_1_0607.parquet"),
        pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet"),
        pd.read_parquet("runs/nfcore-20260708T125031Z/datasets/ampliseq_triple_run.parquet")

]

test_ampliseq = pd.read_parquet("runs/nfcore-20260706T112716Z/datasets/ampliseq_3_0707.parquet")



# Remove Outliers
# Dropped 0 timestamps.
# Random Forest
# R² Score:  0.9292
# MAE:       5.17 Wh (2.57% of mean)

#['delta_cpu_ns', 'delta_io_bytes', 'delta_net_send_bytes', 'context_switches', 'syscall_count', 'delta_rss_memory', 'delta_cpu_time_proc', 'syscall_class_file', 'syscall_class_network', 'syscall_class_memory', 'syscall_class_process', 'syscall_class_other']
# train_sarek = [
#     pd.read_parquet("runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"),
#     pd.read_parquet("runs/nfcore-20260702T193504Z/datasets/sarek_2_0207.parquet")

# ]

# test_sarek = pd.read_parquet("runs/nfcore-20260708T212252Z/datasets/sarek3_0907.parquet")

#---------------------------------

# Remove Outliers
# Dropped 0 timestamps.
#  Random Forest
#   R² Score:  0.0373
#   MAE:       16.27 Wh (8.53% of mean)
#['delta_cpu_ns', 'delta_io_bytes', 'delta_net_send_bytes', 'context_switches', 'syscall_count', 'delta_rss_memory', 'delta_cpu_time_proc', 'syscall_class_file', 'syscall_class_network', 'syscall_class_memory', 'syscall_class_process', 'syscall_class_other', 'syscall_class_sched']
#Predicting sarek as unseen type
# Remove Outliers
# Dropped 0 timestamps.
# Random Forest
# R² Score:  0.8262
# MAE:       8.16 Wh (4.08% of mean)
#['delta_cpu_ns', 'delta_io_bytes', 'delta_net_send_bytes', 'context_switches', 'syscall_count', 'delta_rss_memory', 'delta_cpu_time_proc', 'syscall_class_file', 'syscall_class_network', 'syscall_class_memory', 'syscall_class_process', 'syscall_class_other']

# train_mixed_unseen_type = [
#     pd.read_parquet("runs/nfcore-20260704T110043Z/datasets/chipseq_2_0607.parquet"),
#     pd.read_parquet("runs/nfcore-20260701T114734Z/datasets/rnaseq_1_02027.parquet"),
#     #pd.read_parquet("runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"),
#     pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet")

# ]
#test_mixed_unseen_type = pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet")
#test_mixed_unseen_type2 = pd.read_parquet("runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet")


#Maybe bad example -> should be split 80/20
# build mode that lets you test both?
#['delta_cpu_ns', 'delta_io_bytes', 'delta_net_send_bytes', 'context_switches', 'syscall_count', 'delta_rss_memory', 'syscall_class_process', 'syscall_class_signal']
# Remove Outliers
# Dropped 0 timestamps.
#  Random Forest
#   R² Score:  0.8606
#   MAE:       12.80 Wh (6.72% of mean)
# test_stressng = pd.read_parquet("stressng_test3_10_.parquet")
# train_stressng = pd.read_parquet("stressng_train0_3_.parquet")





# test_long_on_short_training = pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet")
# short_training = [
#     pd.read_parquet("runs/nfcore-20260630T142308Z/datasets/rnaseq1_shorttest.parquet"),
#     pd.read_parquet("runs/nfcore-20260630T143512Z/datasets/chipseq1_shorttest.parquet"),
#     pd.read_parquet("runs/nfcore-20260630T152039Z/datasets/methlyseq1_shorttest.parquet"),
#     # pd.read_parquet("runs/nfcore-20260630T152447Z/datasets/methylseq2_shorttest.parquet"),
#     pd.read_parquet("runs/nfcore-20260630T153034Z/datasets/sarek1_shorttest.parquet")
#     # pd.read_parquet("runs/nfcore-20260630T153034Z/datasets/sarek1_shorttest.parquet"),
#     # pd.read_parquet("runs/nfcore-20260630T153801Z/datasets/sarek2_short_test.parquet"),
#     #pd.read_parquet("data/siena12/stressng_siena12.parquet")]

# data = [
#     #pd.read_parquet("data/siena12/full_test/ampliseq_1_0607.parquet"),
#     # pd.read_parquet("data/siena12/full_test/ampliseq_2_0607.parquet"),
#     #pd.read_parquet("data/siena12/full_test/ampliseq_3_0707.parquet")
#     #pd.read_parquet("runs/stressng-custom-1782744477/datasets/process_interval_data.parquet")
#     #pd.read_parquet("data/siena12/test/rnaseq_siena12.parquet"),
#     #pd.read_parquet("runs/nfcore-20260701T114734Z/datasets/rnaseq_1_02027.parquet"),
#     #pd.read_parquet("runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"),
#     #pd.read_parquet("runs/nfcore-20260702T193504Z/datasets/sarek_2_0207.parquet"),
#     #pd.read_parquet("runs/nfcore-20260702T072031Z/datasets/chipseq1_0207.parquet"),
#     #pd.read_parquet("runs/nfcore-20260704T110043Z/datasets/chipseq_2_0607.parquet"),
#     #pd.read_parquet("runs/nfcore-20260704T110043Z/datasets/chipseq_2_0607.parquet"),
#     pd.read_parquet("runs/nfcore-20260703T215123Z/datasets/ampliseq_1_0607.parquet"),
#     pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet"),
#     #pd.read_parquet("runs/nfcore-20260706T112716Z/datasets/ampliseq_3_0707.parquet"),
#     pd.read_parquet("runs/nfcore-20260708T125031Z/datasets/ampliseq_triple_run.parquet")
# ]



training_data = pd.concat(train_ampliseq, ignore_index=True)
training_data = training_data
test_data = test_ampliseq
PNG_NAME = "TEsT"

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

class SafeEBMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, interactions=2, max_rounds=2000):
        self.interactions = interactions
        self.max_rounds = max_rounds
        self.model = None

    def fit(self, X, y):
        self.model = ExplainableBoostingRegressor(
            interactions=self.interactions,
            max_rounds=self.max_rounds,
            n_jobs=-1,
            random_state=42
        )
        
        self.model.fit(X, y)
        n_features = X.shape[1]
        all_importances = self.model.term_importances()
        
        # Now convert to numpy array and slice it for SelectFromModel
        self.feature_importances_ = np.array(all_importances)[:n_features]
        return self

    def predict(self, X):
        return self.model.predict(X)

#good_features =  ['delta_io_bytes', 'context_switches', 'delta_cpu_ns', 'delta_net_send_bytes', 'syscall_count']

preprocessor_train = Preprocessor(training_data, features)
X_train_FULL, y_train, t_train, _ = preprocessor_train.preprocess_no_split()

#Params could be tuned as well -> Optuna Tuner makes no real difference
#model = RandomForestRegressor(n_estimators=100,  n_jobs=-1, random_state=42)
#model = Ridge(alpha=1.0)
#model = Lasso(alpha=0.1)
model = SafeEBMWrapper()


#These thresholds could be fine tuned
#Don't forget scaling the linear stuff before using selec_features
automatic_feature_selection = Pipeline(steps=[
    ('variance', VarianceThreshold(threshold=0.01)), #explain this

    ('decorrelate', CustomSpearmanFilter(threshold=0.80)),
    ('scaler', StandardScaler()),
    ('select_features', SelectFromModel(model, threshold='0.5*median'))
])

automatic_feature_selection.set_output(transform="pandas")
automatic_feature_selection.fit_transform(X_train_FULL, y_train)
good_features = automatic_feature_selection.get_feature_names_out().tolist()
X_train = X_train_FULL[good_features]
print("Selected columns:")
print(good_features)

#plot_dataset(t_train, y_train, "multi_training")


preprocessor_test = Preprocessor(test_data, good_features)
X_test, y_test, t_test , X_test_unaggregated = preprocessor_test.preprocess_no_split()

#plot_dataset(t_test, y_test, "multi_testing")


#idle_power_isactually idle interval energy
builder = ModelBuilder(X_train, X_test, y_train, y_test, model, StandardScaler())
y_pred, learned_idle_power = builder.run_and_save_model(".", model_name="ridge_auto.joblib")


plotter = Plotter(y_pred,y_test, t_test)#, window_start =50, window_end=200)
plotter.plot_and_save("", PNG_NAME)





#check if we ann pass this differently
# attributor = ProcessAttributorSHAP( builder.X_test_scaled, builder.model, builder.scaler)
# attributor.attribute(X_test_unaggregated,good_features,t_test.values , "RF_SHAP")

attributor = ProcessAttributorEBM( builder.X_test_scaled, builder.model.model, builder.scaler)
attributor.attribute(X_test_unaggregated,good_features,t_test.values , "EBM")


#Nur zum Test -> eigentlich Pauls Aufgabe
# attributor = ProcessAttributorLinear(  builder.model, builder.scaler)
# attributor.attribute(X_test_unaggregated,good_features,t_test.values)



