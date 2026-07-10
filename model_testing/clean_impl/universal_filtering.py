import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np



class CustomSpearmanFilter(BaseEstimator, TransformerMixin):
    
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.to_drop_cols_ = [] 
    
    def fit(self, X, y=None):
        #important for the optuna optimiuzation
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        

        corr_matrix = X.corr(method='spearman').abs()
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_triangle_matrix = corr_matrix.where(mask)
        self.to_drop_cols_ = upper_triangle_matrix.columns[(upper_triangle_matrix > self.threshold).any()].tolist()
        return self
    
    def transform(self, X):

        df = pd.DataFrame(X)
        return df.drop(columns=self.to_drop_cols_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided by the pipeline.")
        
        return np.array([feat for feat in input_features if feat not in self.to_drop_cols_])


