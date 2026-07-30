#The code is terminating without error. Don't touch.
#(ProcessEnergyAccounting) @john:~/DistSys_Project/team-red/ProcessEnergyAccounting$ python3 model_testing/clean_impl/multi_pipeline_dl_cnn_laptop3.py
#Dropped 0 timestamps.
#Epoch 1/2
#4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 26ms/step - loss: 54997.9766 - mae: 231.8521
#Epoch 2/2
#4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step - loss: 53363.6992 - mae: 228.2755
#{'importances_mean': array([ 1.09688087e-02, -7.13462572e-05,  1.37027517e-03,  1.13649789e-02,
#        6.32914647e-03,  2.91069202e-03,  0.00000000e+00,  1.17114521e-02,
#        1.17854366e-02,  3.43022278e-03,  3.92351369e-03,  7.65193347e-03,
#       1.39375587e-02,  2.09434876e-03,  8.23626478e-03,  3.17250989e-03,
#        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]), 'importances_std': array([0.0003402 , 0.0001978 , 0.0002691 , 0.00059575, 0.00055485,
#       0.00024923, 0.        , 0.00039984, 0.00059612, 0.00033959,
#       0.0004622 , 0.00055705, 0.00104518, 0.00019642, 0.00041834,
#       0.00033023, 0.        , 0.        , 0.        , 0.        ]), 'importances': array([[ 1.11528385e-02,  1.08635749e-02,  1.11288033e-02,
#         1.13239041e-02,  1.04176806e-02,  1.09997300e-02,
#         1.10924491e-02,  1.04331400e-02,  1.07529009e-02,
#         1.15230653e-02],
#       [-2.78179761e-05, -8.62218424e-06,  1.46747746e-04,
#        -3.66556618e-04, -4.33108914e-04,  6.70725187e-05,
#        -1.40239297e-04, -1.92344411e-04,  1.81963935e-04,
#         5.94426286e-05],
#       [ 8.80352844e-04,  1.48067558e-03,  1.37561002e-03,
#        1.76594339e-03,  1.44439277e-03,  1.70424817e-03,
#         1.00600709e-03,  1.17234692e-03,  1.53062778e-03,
#         1.34254714e-03],
#      [ 1.10322965e-02,  1.21161132e-02,  1.13329693e-02,
#         1.09860155e-02,  1.17766414e-02,  1.06117158e-02,
#         1.15310947e-02,  1.06087818e-02,  1.25221622e-02,
#         1.11319985e-02],
#       [ 6.30530150e-03,  6.22431379e-03,  6.03991403e-03,
#         6.80822198e-03,  5.06843319e-03,  6.94523667e-03,
#         5.86912286e-03,  6.96193224e-03,  6.74489590e-03,
#         6.32409250e-03],
#       [ 2.49519083e-03,  2.82908166e-03,  3.19659627e-03,
#         3.12958451e-03,  2.49302730e-03,  2.97021567e-03,
#         3.15212307e-03,  2.77332901e-03,  2.91829038e-03,
#         3.14948145e-03],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00],
#       [ 1.20692957e-02,  1.13268318e-02,  1.17181906e-02,
#         1.18044580e-02,  1.09987542e-02,  1.20481772e-02,
#         1.17450613e-02,  1.15016336e-02,  1.14349449e-02,
#         1.24671737e-02],
#       [ 1.27823185e-02,  1.18566606e-02,  1.11742660e-02,
#         1.15175683e-02,  1.10962913e-02,  1.19152186e-02,
#         1.13986406e-02,  1.14860210e-02,  1.29541534e-02,
#         1.16732280e-02],
#       [ 3.68695052e-03,  3.56018567e-03,  3.66323337e-03,
#         3.57236957e-03,  3.19774052e-03,  3.68054619e-03,
#         3.77478272e-03,  2.68964104e-03,  3.50147440e-03,
#         2.97530383e-03],
#       [ 4.40125341e-03,  4.35830936e-03,  2.98977542e-03,
#         4.18019582e-03,  4.62358571e-03,  3.74189526e-03,
#         3.70101251e-03,  3.72330432e-03,  3.99632081e-03,
#         3.51948425e-03],
#       [ 6.76353799e-03,  7.53991500e-03,  7.20343770e-03,
#         7.35923436e-03,  7.68931731e-03,  8.19651033e-03,
#         7.18468212e-03,  7.76671256e-03,  8.82136617e-03,
#         7.99462120e-03],
#       [ 1.38060489e-02,  1.50780389e-02,  1.29014147e-02,
#         1.27407826e-02,  1.41789502e-02,  1.39315115e-02,
#         1.42899102e-02,  1.28646093e-02,  1.62594488e-02,
#         1.33248720e-02],
#       [ 2.15228980e-03,  2.00533468e-03,  2.37290927e-03,
#         2.21581012e-03,  1.67770970e-03,  2.18087858e-03,
#         1.85768560e-03,  2.24726886e-03,  2.21849680e-03,
#         2.01510419e-03],
#       [ 8.12814133e-03,  8.98144290e-03,  7.43219539e-03,
#         8.29270580e-03,  8.76978925e-03,  7.97984430e-03,
#         8.02226184e-03,  8.34123336e-03,  7.95796318e-03,
#         8.45707046e-03],
#       [ 3.51108335e-03,  3.01559964e-03,  3.35708791e-03,
#         2.88278000e-03,  3.16504736e-03,  3.62249584e-03,
#         3.64483275e-03,  2.62224070e-03,  3.01336177e-03,
#         2.89056953e-03],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00]])}
#Selected columns:
#['delta_cpu_ns', 'context_switches', 'syscall_count', 'delta_cpu_time_proc', 'syscall_class_file', 'syscall_class_network', 'syscall_class_memory', 'syscall_class_process', 'syscall_class_other', 'syscall_class_signal']

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

# Basic Deep Learning with Sklearn MLP
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance

# Deep Learning with Keras Tensorflow
#import keras
from keras import layers, optimizers, callbacks, Sequential
from keras.wrappers import SKLearnRegressor

#train_sarek = [
#    pd.read_parquet("../../../../ProcessEnergyAccounting/runs/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"),
    #pd.read_parquet("../../../../ProcessEnergyAccounting/runs/nfcore-20260702T193504Z/datasets/sarek_2_0207.parquet")
#]
#test_sarek = pd.read_parquet("../../../../ProcessEnergyAccounting/runs/nfcore-20260708T212252Z/datasets/sarek3_0907.parquet")
DATA_PATH = "workflows/siena12/nfcore-20260701T215234Z/datasets/sarek_1_0207.parquet"
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

# Models used
# Convolutional Neural Network (1D)
num_features = len(X_train_FULL.columns)

def dynamic_model(num_features, model_name):
    cnn_model = Sequential([

        layers.Input(shape=(num_features, 1)), # (num_features, sequence_length) #Only current value
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
    def __init__(self,model= None):
        self.model = model

    def fit(self, X, y):
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.001, epsilon=1e-4), loss='mse', metrics=['mae'])
        self.model.fit(X, y, epochs=2, batch_size=256, callbacks=standard_callbacks) #validation_split=0.1) #callbacks = [self.callbacks])

        all_importances = permutation_importance(self, X, y,
                           n_repeats=10,
                           #scoring='neg_mean_absolute_error',
                           scoring='r2',
                           random_state=0)
        
        # Now convert to numpy array and slice it for SelectFromModel
        print(np.array(all_importances))
        self.feature_importances_ = np.array(all_importances.importances_mean)
        #print(self.feature_importances_ )
        return self

    def predict(self, X):
        return self.model.predict(X, verbose = 0)

model = SafeKerasWrapper(dynamic_model(num_features,"cnn"))

#These thresholds could be fine tuned
automatic_feature_selection = Pipeline(steps=[
    #('variance', VarianceThreshold(threshold=0.01)), #explain this

    #('decorrelate', CustomSpearmanFilter(threshold=0.80)),
    ('scaler', StandardScaler()),
    ('select_features', SelectFromModel(model, threshold='median'))#threshold='0.5*median'))
])

automatic_feature_selection.set_output(transform="pandas")
automatic_feature_selection.fit_transform(X_train_FULL, y_train)
good_features = automatic_feature_selection.get_feature_names_out().tolist()
X_train = X_train_FULL[good_features]
print("Selected columns:")
print(good_features)


#plot_dataset(t_train, y_train, "multi_training")