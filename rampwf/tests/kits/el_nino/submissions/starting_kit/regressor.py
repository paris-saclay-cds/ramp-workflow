from sklearn.base import BaseEstimator
from sklearn import linear_model


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = linear_model.BayesianRidge()

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
