from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.linear_model import LinearRegression


class GenerativeRegressor(BaseEstimator):
    def __init__(self, max_dists, current_fold):
        self.reg = LinearRegression()
        self.max_gauss = max_dists
        self.sigma = None
        self.a = None
        self.b = None

    def fit(self, X, y):
        self.reg.fit(X, y)
        yGuess = self.reg.predict(X)
        yGuess = np.array([yGuess]).reshape(-1, 1)
        error = y - yGuess
        self.sigma = np.sqrt((1 / X.shape[0]) * np.sum(error ** 2))
        self.a = np.min(y)
        self.b = np.max(y)

    def predict(self, X):
        # The first generative regressor is gaussian, the second is uniform
        types = np.array([[0, 1], ] * len(X))

        # Normal
        preds = self.reg.predict(X)
        sigmas = np.array([self.sigma] * len(X))
        sigmas = sigmas[:, np.newaxis]
        params_normal = np.concatenate((preds, sigmas), axis=1)

        # Uniform
        a_array = np.array([self.a] * len(X))
        b_array = np.array([self.b] * len(X))
        params_uniform = np.stack((a_array, b_array), axis=1)

        # We give more weight to the gaussian one
        weights = np.array([[0.99, 0.01], ] * len(X))

        # We concatenate the params
        params = np.concatenate((params_normal, params_uniform), axis=1)
        return weights, types, params
