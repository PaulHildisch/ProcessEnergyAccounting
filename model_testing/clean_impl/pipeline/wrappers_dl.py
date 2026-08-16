import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
#from time import perf_counter

# Basic Deep Learning
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance

from keras import layers, optimizers, callbacks, Sequential,regularizers
from keras.wrappers import SKLearnRegressor

# MLP Wrapper
class SafeMLPWrapper(BaseEstimator, RegressorMixin):
    def __init__(self,activation="relu", solver="adam", batch_size=256, learning_rate_init=0.0001,
                                max_iter=500,n_repeats=10):
        self.activation = activation
        self.solver = solver
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.n_repeats = n_repeats
        self.model = MLPRegressor(hidden_layer_sizes=(128,32,16),
                            activation='relu',
                            solver='adam',
                            learning_rate_init=self.learning_rate_init,
                            max_iter=self.max_iter,
                            batch_size=batch_size,
                            early_stopping=True,    # Crucial for time-series stability
                            random_state=42)

    def fit(self, X, y):
        #training_fs_start_time = perf_counter()
        self.model.fit(X, y)
        #training_fs_end_time = perf_counter()
        all_importances = permutation_importance(self, X, y,
                                   n_repeats=self.n_repeats,
                                   scoring='neg_mean_squared_error',
                                   random_state=42,
                                   n_jobs = -1
                                    )
        
        # Now convert to numpy array and slice it for SelectFromModel
        #print(np.array(all_importances))

        #training_fs_time = training_fs_end_time - training_fs_start_time
        #print(f"Training feature selection time: {training_fs_time:.2f} seconds")
        self.feature_importances_ = np.array(all_importances.importances_mean)
        return self

    def predict(self, X):
        return self.model.predict(X)

standard_callbacks = [
    callbacks.TerminateOnNaN(),
    callbacks.EarlyStopping(monitor='loss',patience=3),
    ]

class SafeKerasWrapper(RegressorMixin, BaseEstimator):
    def __init__(self,model = None):
        self.model = model

    def fit(self, X, y):
        
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.001, epsilon=1e-4), loss='mse', metrics=['mae'])
        #training_fs_start_time = perf_counter()

        self.model.fit(X, y, epochs=20, batch_size=256, callbacks=standard_callbacks, verbose = 0)
        #training_fs_end_time = perf_counter()
        #print("permutation_importance:")
        #print(X.shape)
        #print(y.shape)

        all_importances = permutation_importance(self.model, X, y,
                           n_repeats=3,
                           scoring='neg_mean_squared_error',
                           random_state=42,
                           n_jobs = -1)
        # Now convert to numpy array and slice it for SelectFromModel
        #print(np.array(all_importances))

        #training_fs_time = training_fs_end_time - training_fs_start_time
        #print(f"Training feature selection time: {training_fs_time:.2f} seconds")
        self.feature_importances_ = np.array(all_importances.importances_mean)
        return self

    def predict(self, X):
        return self.model.predict(X, verbose = 0)

