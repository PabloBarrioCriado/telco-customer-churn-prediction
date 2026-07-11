import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ServiceAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, drop_originals=True):
        self.drop_originals = drop_originals
        self.service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                             'TechSupport', 'StreamingTV', 'StreamingMovies']
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        # How many "Yes" are there in the extra services
        X_copy['TotalAdditionalServices'] = X_copy[self.service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
        
        # Drop or keep the original service columns based on the strategy
        if self.drop_originals:
            X_copy = X_copy.drop(columns=self.service_cols)
            
        return X_copy