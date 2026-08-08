import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
#from time import perf_counter

# Basic Deep Learning with Sklearn MLP
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance

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
        #training_fs_start_time = perf_counter()
        self.model.fit(X, y)
        #training_fs_end_time = perf_counter()
        all_importances = permutation_importance(self, X, y,
                                   n_repeats=10,
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