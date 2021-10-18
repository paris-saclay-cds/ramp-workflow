from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = RandomForestRegressor(
            n_estimators=10, max_depth=10, max_features=10)

    def fit(self, X, y, prev_regressor=None):
        if prev_regressor is not None:
            self.reg = prev_regressor.reg
            self.reg.set_params(
                n_estimators=2 * self.reg.n_estimators, warm_start=True)
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
