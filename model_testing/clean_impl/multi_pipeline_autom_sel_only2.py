
from time import perf_counter
import os
start_time = perf_counter()

from model_builder import ModelBuilder
from model_builder_keras import KerasModelBuilder

from preprocessing import Preprocessor
from plotting_other import Plotter
from plotting import plot_dataset
#from shapley import ProcessAttributor
from shapley_improved import ProcessAttributorSHAP
from shapley_improved_other import ProcessAttributorSHAPMLP

from universal_filtering import CustomSpearmanFilter
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

from interpret.glassbox import ExplainableBoostingRegressor

# Basic Deep Learning with Sklearn MLP
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance

# Deep Learning with Keras Tensorflow
#import keras
from keras import layers, optimizers, callbacks, Sequential,regularizers
from keras.wrappers import SKLearnRegressor

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load data

#train_sarek = [
#    pd.read_parquet("../../../../ProcessEnergyAccounting/runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"),
    #pd.read_parquet("../../../../ProcessEnergyAccounting/runs/nfcore-20260702T193504Z/datasets/sarek_2_0207.parquet")
#]
#test_sarek = pd.read_parquet("../../../../ProcessEnergyAccounting/runs/nfcore-20260708T212252Z/datasets/sarek3_0907.parquet")
DATA_PATH = "../../ProcessEnergyAccounting/runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"
data_sarek = pd.read_parquet(DATA_PATH)


number_slice = int(len(data_sarek)*0.8*0.1) #Smaller dataset
training_data = data_sarek[:number_slice]
test_data = data_sarek[number_slice: ]

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


preprocessor_train = Preprocessor(training_data, features)
X_train_FULL, y_train, t_train, _ = preprocessor_train.preprocess_no_split()

# For windowing fucntionality the size to raise over 1. 
# Windowing fucntionality is only intended for CNN and LSTM.
window_size = 20
num_features = len(X_train_FULL.columns)

# Models used
# Convolutional Neural Network (1D)
def dynamic_model(model_name, num_features, window_size):
    print(model_name)
    print(num_features)
    print(window_size)
    cnn_model = Sequential([

        layers.Input(shape=(num_features, window_size)), # (num_features, sequence_length) #Only current value
        layers.Conv1D(32, kernel_size=num_features, padding='same', activation="relu"),
        layers.BatchNormalization(),

        layers.Conv1D(32, kernel_size=num_features, padding='same', activation="relu"),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        #layers.Dense(32, activation='relu'),
        layers.Dense(1)
    
    ])
    ffnn_model = Sequential([
    layers.Input(shape=(num_features, 1)), # (num_features, sequence_length) #Only current value
    layers.Flatten(),

    layers.Dense(64, activation='relu'),

    layers.Dense(16, activation='relu'),
    layers.Dense(1)
    
])
    if model_name == "cnn":
        model = cnn_model
    elif model_name == "ffnn":
        model = ffnn_model
    else:
        ValueError("Model not implemented")
    return model

standard_callbacks = [
    callbacks.TerminateOnNaN(),
    callbacks.EarlyStopping(monitor='loss',patience=3),
    ]

class SafeKerasWrapper(RegressorMixin, BaseEstimator):
    def __init__(self,model = None, window_size = 1):
        self.model = model
        self.window_size = window_size

    def fit(self, X, y):
        
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.001, epsilon=1e-4), loss='mse', metrics=['mae'])
        training_start_time = perf_counter()
        X_special = X
        if self.window_size > 1:
            
            X = np.lib.stride_tricks.sliding_window_view(X, self.window_size, axis=0)
            y = y[self.window_size - 1:]
            print("train windowed:")
            print(X.shape)
            print(X.shape[0])
            print(y.shape)
            self.model.fit(X, y, epochs=10, batch_size=256, callbacks=standard_callbacks, verbose = 0) #validation_split=0.1) #callbacks = [self.callbacks])
            X_special = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
        else:
            self.model.fit(X, y, epochs=10, batch_size=256, callbacks=standard_callbacks, verbose = 0) #validation_split=0.1) #callbacks = [self.callbacks])

        training_end_time = perf_counter()
        print("permutation_importance:")
        print(X_special.shape)
        print(y.shape)

        all_importances = permutation_importance(model, X_special, y,
                           n_repeats=3,
                           scoring='neg_mean_squared_error',
                           random_state=42,
                           #max_samples=0.2, # It seems to not matter here.
                           n_jobs = -1)
        all_importances = np.array(all_importances.importances_mean)
        # Now convert to numpy array and slice it for SelectFromModel
        #print(np.array(all_importances))

        if window_size > 1:
            ws = self.window_size
            for i in range(len(all_importances)//ws):
                all_importances[i*ws:i*ws+ws] = np.tile(all_importances[i*ws:i*ws+ws].mean(),ws)
            print(all_importances.shape)
        self.feature_importances_ = all_importances
        #print(self.feature_importances_ )
        
        training_execution_time = training_end_time - training_start_time
        print(f"Training execution time: {training_execution_time:.2f} seconds")
        return self

    def predict(self, X):
        #print("predict:")
        #print(X.shape)
        if X.ndim == 2 and self.window_size > 1:
            # Dynamically calculate features per timestep if SelectFromModel pruned columns
            current_features = X.shape[1] // self.window_size  # e.g., 400 // 20 = 20
            X = X.reshape(X.shape[0], current_features, self.window_size)
        #if self.window_size > 1:
        #    X = np.lib.stride_tricks.sliding_window_view(X, self.window_size, axis=0)
        print(X.shape)
        #print(X.shape)
        return self.model.predict(X, verbose = 0)

# EBMWrapper
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

# MLP Wrapper
class SafeMLPWrapper(BaseEstimator, RegressorMixin):
    def __init__(self,activation="relu", solver="adam"):
        self.activation = activation
        self.solver = solver
        self.model = None

    def fit(self, X, y):
        self.model = MLPRegressor(hidden_layer_sizes=(128,32,16),
                    activation='relu',
                    solver='adam',
                    learning_rate_init=0.0001,
                    max_iter=500,
                    #alpha = 0.0000675,
                    batch_size=256,
                    early_stopping=True,    # Crucial for time-series stability
                    #validation_fraction=0.1,
                    random_state=42)
        training_start_time = perf_counter()
        self.model.fit(X, y)
        training_end_time = perf_counter()
        all_importances = permutation_importance(self, X, y,
                                   n_repeats=10,
                                   scoring='neg_mean_squared_error',
                                   random_state=42,
                                   n_jobs = -1
                                    )
        
        # Now convert to numpy array and slice it for SelectFromModel
        #print(np.array(all_importances))

        training_execution_time = training_end_time - training_start_time
        print(f"Training execution time: {training_execution_time:.2f} seconds")
        self.feature_importances_ = np.array(all_importances.importances_mean)
        return self

    def predict(self, X):
        return self.model.predict(X)




model = SafeKerasWrapper(dynamic_model("cnn",num_features,1),1) #only the current version
#model = SafeKerasWrapper(dynamic_model("ffnn",num_features,1),1)

#model = SafeMLPWrapper()
#model = RandomForestRegressor(n_estimators=100,  n_jobs=-1, random_state=42)
#model = SafeEBMWrapper()
afs_start_time = perf_counter()


# Automatic Feature Selection Pipeline
#These thresholds could be fine tuned
automatic_feature_selection = Pipeline(steps=[
    #('variance', VarianceThreshold(threshold=0.01)), #explain this

    #('decorrelate', CustomSpearmanFilter(threshold=0.80)),
    ('scaler', StandardScaler()),
    ('select_features', SelectFromModel(model, threshold='0.5*median'))#threshold='0.5*median'))
])

automatic_feature_selection.set_output(transform="pandas")
automatic_feature_selection.fit_transform(X_train_FULL, y_train)
good_features = automatic_feature_selection.get_feature_names_out().tolist()
X_train = X_train_FULL[good_features]
print("Selected columns:")
print(len(good_features))
print(good_features)


#plot_dataset(t_train, y_train, "multi_training")


afs_end_time = perf_counter()
end_time = perf_counter()
afs_execution_time = afs_end_time - afs_start_time
total_execution_time = end_time - start_time

print("Basic Time Calculation")
print(f" AFS execution time: {afs_execution_time:.2f} seconds")


print(f" Total execution time: {total_execution_time:.2f} seconds")