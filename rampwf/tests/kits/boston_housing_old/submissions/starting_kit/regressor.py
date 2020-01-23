from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor


class Regressor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.reg = RandomForestRegressor(
            n_estimators=2, max_leaf_nodes=2, random_state=61)
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
