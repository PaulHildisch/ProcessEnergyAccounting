import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.linear_model import Lasso


#Non scikit learn models need a wrapper to implement the scikit learn interface to be compatible with the feature selection pipeline
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
        
        #convert to numpy array and slice it for SelectFromModel
        self.feature_importances_ = np.array(all_importances)[:n_features]
        return self

    def predict(self, X):
        return self.model.predict(X)


class CvxpyMimicLasso(BaseEstimator, RegressorMixin):
    def __init__(self, l1_penalty=0.1):
        self.l1_penalty = l1_penalty
        self.model = None

    def fit(self, X, y):

        N = X.shape[0]
        sklearn_alpha = self.l1_penalty / (2 * N)
    
        self.model = Lasso(
            alpha=sklearn_alpha, 
            positive=True, 
            fit_intercept=True, 
            max_iter=10000
        )
        self.model.fit(X, y)
        
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        return self

    def predict(self, X):
        return self.model.predict(X)
