
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

# Dataset Selection

train_ampliseq = [
        pd.read_parquet("../../ProcessEnergyAccounting/runs/nfcore-20260703T215123Z/datasets/ampliseq_1_0607.parquet"),
        pd.read_parquet("../../ProcessEnergyAccounting/runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet"),
        pd.read_parquet("../../ProcessEnergyAccounting/runs/nfcore-20260708T125031Z/datasets/ampliseq_triple_run.parquet")

]

test_ampliseq = pd.read_parquet("../../ProcessEnergyAccounting/runs/nfcore-20260706T112716Z/datasets/ampliseq_3_0707.parquet")

#---------------------------------

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
#Be careful what you uncomment -> test data must not be in test data

# train_mixed_unseen_type = [
#     pd.read_parquet("runs/nfcore-20260704T110043Z/datasets/chipseq_2_0607.parquet"),
#     pd.read_parquet("runs/nfcore-20260701T114734Z/datasets/rnaseq_1_02027.parquet"),
#     #pd.read_parquet("runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"),
#     pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet")

# ]
#test_mixed_unseen_type = pd.read_parquet("runs/nfcore-20260704T093159Z/datasets/ampliseq_2_0607.parquet")
#test_mixed_unseen_type2 = pd.read_parquet("runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet")

#---------------------------------

#DATA_PATH = "../../ProcessEnergyAccounting/runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"
#data_sarek = pd.read_parquet(DATA_PATH)


#number_slice = int(len(data_sarek)*0.8*0.1) #Smaller dataset
#training_data = data_sarek[:number_slice]
#test_data = data_sarek[number_slice: ]

# Load Data
training_data = pd.concat(train_ampliseq, ignore_index=True)
training_data = training_data
test_data = test_ampliseq

#General sarek set
#good_features = ['delta_cpu_ns', 'delta_io_bytes', 'delta_net_send_bytes', 'context_switches', 'syscall_count', 'delta_rss_memory', 'delta_cpu_time_proc', 'syscall_class_file', 'syscall_class_network', 'syscall_class_memory', 'syscall_class_process', 'syscall_class_other']
#General unseen set, mixed unseen type2:
#good_features = ['delta_cpu_ns', 'delta_io_bytes', 'delta_net_send_bytes', 'context_switches', 'syscall_count', 'delta_rss_memory', 'delta_cpu_time_proc', 'syscall_class_file', 'syscall_class_network', 'syscall_class_memory', 'syscall_class_process', 'syscall_class_other']
#Generalized set
#good_features =  ['delta_io_bytes', 'context_switches', 'delta_cpu_ns', 'delta_net_send_bytes', 'syscall_count']

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
# Windowing fucntionality is only intended for CNN.
window_size = 1
num_features = len(X_train_FULL.columns)

# Models used
mlp_model = MLPRegressor(hidden_layer_sizes=(128,32,16),
                    activation='relu',
                    solver='adam',
                    learning_rate_init=0.0001,
                    max_iter=500,
                    #alpha = 0.0000675,
                    batch_size=64,
                    early_stopping=True,    # Crucial for time-series stability
                    #validation_fraction=0.1,
                    random_state=42)
# Convolutional Neural Network (1D)
def dynamic_model(model_name, num_features, window_size=1):
    cnn_model = Sequential([

        layers.Input(shape=(num_features, window_size)), # (num_features, sequence_length) #Only current value
        layers.Conv1D(32, kernel_size=num_features, padding='same', activation="relu"),
        layers.BatchNormalization(),

        layers.Conv1D(32, kernel_size=num_features, padding='same', activation="relu"),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    
    ])

    if model_name == "cnn":
        model = cnn_model
    else:
        ValueError("Model not implemented")
    return model

standard_callbacks = [
    callbacks.TerminateOnNaN(),
    callbacks.EarlyStopping(monitor='loss',patience=3),
    ]

class SafeKerasWrapper(RegressorMixin, BaseEstimator):
    def __init__(self,model = None):
        self.model = model

    def fit(self, X, y):
        
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.001, epsilon=1e-4), loss='mse', metrics=['mae'])
        training_fs_start_time = perf_counter()

        self.model.fit(X, y, epochs=20, batch_size=256, callbacks=standard_callbacks, verbose = 0) #validation_split=0.1) #callbacks = [self.callbacks])

        training_fs_end_time = perf_counter()
        print("permutation_importance:")
        print(X.shape)
        print(y.shape)

        all_importances = permutation_importance(model, X, y,
                           n_repeats=5,
                           scoring='neg_mean_squared_error',
                           random_state=42,
                           n_jobs = -1)
        all_importances = np.array(all_importances.importances_mean)
        # Now convert to numpy array and slice it for SelectFromModel
        #print(np.array(all_importances))

        self.feature_importances_ = all_importances
        #print(self.feature_importances_ )
        
        training_fs_time = training_fs_end_time - training_fs_start_time
        print(f"Training feature selection time: {training_fs_time:.2f} seconds")
        return self

    def predict(self, X):
        return self.model.predict(X, verbose = 0)


# MLP Wrapper
class SafeMLPWrapper(BaseEstimator, RegressorMixin):
    def __init__(self,activation="relu", solver="adam"):
        self.activation = activation
        self.solver = solver
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

    def fit(self, X, y):
        training_fs_start_time = perf_counter()
        self.model.fit(X, y)
        training_fs_end_time = perf_counter()
        all_importances = permutation_importance(self, X, y,
                                   n_repeats=10,
                                   scoring='neg_mean_squared_error',
                                   random_state=42,
                                   n_jobs = -1
                                    )
        
        # Now convert to numpy array and slice it for SelectFromModel
        #print(np.array(all_importances))

        training_fs_time = training_fs_end_time - training_fs_start_time
        print(f"Training feature selection time: {training_fs_time:.2f} seconds")
        self.feature_importances_ = np.array(all_importances.importances_mean)
        return self

    def predict(self, X):
        return self.model.predict(X)



#model = SafeKerasWrapper(dynamic_model("cnn",num_features,1)) #only the current version
model = SafeMLPWrapper()
#model = RandomForestRegressor(n_estimators=100,  n_jobs=-1, random_state=42)
afs_start_time = perf_counter()


# Automatic Feature Selection Pipeline
#These thresholds could be fine tuned
automatic_feature_selection = Pipeline(steps=[
    #For Keras, comment out VarianceThreshold and CustomSpearmanFilter.
    ('variance', VarianceThreshold(threshold=0.01)), #explain this
    ('decorrelate', CustomSpearmanFilter(threshold=0.80)),
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
num_features = len(good_features)

#plot_dataset(t_train, y_train, "multi_training")
afs_end_time = perf_counter()


# Test dataset preprocessing 
preprocessor_test = Preprocessor(test_data, good_features)
X_test, y_test, t_test , X_test_unaggregated = preprocessor_test.preprocess_no_split()

# Replace the model with the chosen model.
# KerasModelBuilder has some extra functionality for Keras Deep Learning Framework.
training_start_time = perf_counter()
#train_model = model
#train_model = dynamic_model("cnn",num_features,window_size)  
train_model = mlp_model
#builder= KerasModelBuilder(X_train, X_test, y_train, y_test, train_model, StandardScaler(), 
#           window_size=window_size, train_epochs=30)
builder = ModelBuilder(X_train, X_test, y_train, y_test, train_model, StandardScaler())

y_pred, learned_idle_power = builder.run_and_save_model()
training_end_time = perf_counter()

#plotter = Plotter(y_pred,y_test, t_test)#, window_start =50, window_end=200)
#plotter.plot_and_save("", PNG_NAME)

plotter = Plotter(y_pred,y_test, t_test,"cnn")#, window_start =50, window_end=200)
plotter.plot_and_save("cnn")

#For windowing functionality 
#plotter = Plotter(y_pred=y_pred,y_test=y_test[window_size - 1:], t_test= t_test[window_size - 1:],alg_name="lstm")
#plotter.plot_and_save("cnn_windowing_")

#check if we ann pass this differently
#attributor = ProcessAttributorSHAPMLP( builder.X_test_scaled, builder.model, builder.scaler)
#attributor.attribute(X_test_unaggregated,good_features,t_test.values, "mlp_graphs_")

#attributor = ProcessAttributorSHAP( builder.X_test_scaled, builder.model, builder.scaler)
#attributor.attribute(X_test_unaggregated,good_features,t_test.values , "RF_SHAP")

end_time = perf_counter()

print("Basic Time Calculation")

afs_execution_time = afs_end_time - afs_start_time
training_execution_time = training_end_time - training_start_time
total_execution_time = end_time - start_time
print(f"AFS execution time: {afs_execution_time:.2f} seconds")
print(f"Training execution time: {training_execution_time:.2f} seconds")

print(f"Total execution time: {total_execution_time:.2f} seconds")