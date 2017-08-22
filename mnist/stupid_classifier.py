from sklearn.base import BaseEstimator;
import numpy as np

class StupidClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
