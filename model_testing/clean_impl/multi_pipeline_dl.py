from model_builder import ModelBuilder
from model_builder_keras import KerasModelBuilder

from preprocessing import Preprocessor
from plotting_other import Plotter # Originally from plotting import Plotter
from plotting_other import plot_dataset
#from shapley import ProcessAttributor
from shapley_improved import ProcessAttributorSHAP
from shapley_improved_other import ProcessAttributorSHAPMLP

from universal_filtering import CustomSpearmanFilter
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import Ridge
#from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

from interpret.glassbox import ExplainableBoostingRegressor

# Basic Deep Learning with Sklearn MLP
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance

# Deep Learning with Keras Tensorflow
#import keras
from keras import layers, optimizers, callbacks, Sequential

# Dataset Selection

train_ampliseq = [
        pd.read_parquet("../../ProcessEnergyAccounting/runs/nfcore-20260703T215123Z/datasets/ampliseq_1_0607.parquet"),
        pd.read_parquet("../../ProcessEnergyAccounting/runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet"),
        pd.read_parquet("../../ProcessEnergyAccounting/runs/nfcore-20260708T125031Z/datasets/ampliseq_triple_run.parquet")

]

test_ampliseq = pd.read_parquet("../../ProcessEnergyAccounting/runs/nfcore-20260706T112716Z/datasets/ampliseq_3_0707.parquet")

#train_ampliseq = [
#        pd.read_parquet("runs/nfcore-20260703T215123Z/datasets/ampliseq_1_0607.parquet"),
#        pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet"),
#        pd.read_parquet("runs/nfcore-20260708T125031Z/datasets/ampliseq_triple_run.parquet")
#
#]

#test_ampliseq = pd.read_parquet("runs/nfcore-20260706T112716Z/datasets/ampliseq_3_0707.parquet")

#---------------------------------

# train_sarek = [
#     pd.read_parquet("runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"),
#     pd.read_parquet("runs/nfcore-20260702T193504Z/datasets/sarek_2_0207.parquet")

# ]

# test_sarek = pd.read_parquet("runs/nfcore-20260708T212252Z/datasets/sarek3_0907.parquet")

#---------------------------------

# train_mixed_unseen_type = [
#     pd.read_parquet("runs/nfcore-20260704T110043Z/datasets/chipseq_2_0607.parquet"),
#     pd.read_parquet("runs/nfcore-20260701T114734Z/datasets/rnaseq_1_02027.parquet"),
#     #pd.read_parquet("runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"),
#     pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet")

# ]
#test_mixed_unseen_type = pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet")
#test_mixed_unseen_type2 = pd.read_parquet("runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet")

#---------------------------------

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

# Load Data
training_data = pd.concat(train_ampliseq, ignore_index=True)
training_data = training_data
test_data = test_ampliseq
PNG_NAME = "mlp_pred_sarek"

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

# Train Set Preprocessing
preprocessor_train = Preprocessor(training_data, features)
X_train_FULL, y_train, t_train, _ = preprocessor_train.preprocess_no_split()

# MLP model structure.
# mlp_model = MLPRegressor(hidden_layer_sizes=(128,32,16),
#                    activation='relu',
#                    solver='adam',
#                    learning_rate_init=0.0001,
#                    max_iter=500,
#                    batch_size=64,
#                    early_stopping=True,
#                    random_state=42)

# Random Forest recommended for automatic feature selection pipeline.
# MLP can be also used for automatic feature selection pipeline.
# class SafeMLPWrapper(BaseEstimator, RegressorMixin):
#    def __init__(self,activation="relu", solver="adam",model=mlp_model):
#        self.activation = activation
#        self.solver = solver
#        self.model = None
#
#    def fit(self, X, y):
#        self.model = mlp_model
#        
#        self.model.fit(X, y)
#        n_features = X.shape[1]
#        all_importances = permutation_importance(self.model, X, y,
#                           n_repeats=30,
#                           random_state=0)
#        
#        # Now convert to numpy array and slice it for SelectFromModel
#        #print(np.array(all_importances))
#        self.feature_importances_ = np.array(all_importances.importances_mean)
#        return self
#
#    def predict(self, X):
#        return self.model.predict(X)

# 
# List of models for both feature selection and training.
# Params could be tuned as well -> Optuna Tuner makes no real difference

model = RandomForestRegressor(n_estimators=100,  n_jobs=-1, random_state=42)
#model = Ridge(alpha=1.0)
#model = Lasso(alpha=0.1)
#model = SafeMLPWrapper()

#These thresholds could be fine tuned.
#Don't forget scaling the linear data before selecting features.
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

# Test dataset preprocessing 
preprocessor_test = Preprocessor(test_data, good_features)
X_test, y_test, t_test , X_test_unaggregated = preprocessor_test.preprocess_no_split()

#plot_dataset(t_test, y_test, "multi_testing")

# For windowing fucntionality the size to raise over 1. 
# Windowing fucntionality is only intended for CNN and LSTM.
window_size = 20
num_features = len(good_features)

# Models only for training, not feature selection it could take too long.
# Convolutional Neural Network (1D)
cnn_model = Sequential([

    layers.Input(shape=(num_features, window_size)), # (num_features, sequence_length) #Only current value
    layers.Conv1D(32, kernel_size=num_features, padding='same', activation="relu"),
    layers.BatchNormalization(),

    layers.Conv1D(32, kernel_size=num_features, padding='same', activation="relu"),
    layers.BatchNormalization(),

    #layers.Conv1D(32, kernel_size=num_features, padding='same', activation="relu"),
    #layers.BatchNormalization(),
    
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
    
])

# Feed Forward Neural Network
#ffnn_model = Sequential([
#    layers.Input(shape=(num_features, window_size)), # (num_features, sequence_length) #Only current value
#    layers.Flatten(),

#    layers.Dense(64, activation='relu'),

#    layers.Dense(16, activation='relu'),
#    layers.Dense(1)
    
#])

# LSTM Model
#lstm_model = Sequential([
#    layers.Input(shape=(num_features, window_size)), # (num_features, sequence_length) #Only current value
#    layers.BatchNormalization(),
#    layers.LSTM(64, return_sequences=True),
#    layers.LSTM(64, return_sequences=True),
#    layers.Flatten(),

#    layers.Dense(1)
    
#])

#idle_power_is actually idle interval energy
# Replace the model with the chosen model.
# KerasModelBuilder has some extra functionality for Keras Deep Learning Framework.
builder= KerasModelBuilder(X_train, X_test, y_train, y_test, cnn_model, StandardScaler(), window_size=20, 
                                train_epochs=20)
# builder = ModelBuilder(X_train, X_test, y_train, y_test, model, StandardScaler())

y_pred, learned_idle_power = builder.run_and_save_model()
#y_pred, learned_idle_power = builder.run_and_save_model(".", model_name="mlp_model.joblib", save=True)

#plotter = Plotter(y_pred,y_test, t_test)#, window_start =50, window_end=200)
#plotter.plot_and_save("", PNG_NAME)

#plotter = Plotter(y_pred,y_test, t_test,"cnn_1d")#, window_start =50, window_end=200)
#plotter.plot_and_save("cnn_1d_")

#For windowing fucntionality 
plotter = Plotter(y_pred=y_pred,y_test=y_test[window_size - 1:], t_test= t_test[window_size - 1:],alg_name="lstm")
plotter.plot_and_save("cnn_1d_windowing_")

#check if we ann pass this differently
# attributor = ProcessAttributorSHAP( builder.X_test_scaled, builder.model, builder.scaler)
# attributor.attribute(X_test_unaggregated,good_features,t_test.values , "RF_SHAP")

# attributor = ProcessAttributorEBM( builder.X_test_scaled, builder.model.model, builder.scaler)
# attributor.attribute(X_test_unaggregated,good_features,t_test.values , "EBM")

#check if we ann pass this differently
#attributor = ProcessAttributorSHAPMLP( builder.X_test_scaled, builder.model, builder.scaler)
#attributor.attribute(X_test_unaggregated,good_features,t_test.values, "mlp_graphs_")

#Nur zum Test -> eigentlich Pauls Aufgabe
# attributor = ProcessAttributorLinear(  builder.model, builder.scaler)
# attributor.attribute(X_test_unaggregated,good_features,t_test.values)



