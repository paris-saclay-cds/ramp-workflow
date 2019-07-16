from sklearn.base import BaseEstimator
from scipy.optimize import minimize
import numpy as np
from sklearn.linear_model import LinearRegression

from scipy.integrate import quad
from scipy.stats import norm

# Math from https://www.statlect.com/fundamentals-of-statistics/linear-regression-maximum-likelihood


BINS_RANGE = 4  # we have 4 sigmas on the left, 4 on the right


class GenerativeRegressor(BaseEstimator):
    def __init__(self, nb_bins):
        self.reg = LinearRegression()
        self.nb_bins = nb_bins
        self.sigma = np.nan

    def fit(self, X, y):
        self.reg.fit(X, y)
        yGuess = self.reg.predict(X)
        yGuess = np.array([yGuess]).reshape(-1, 1)
        error = y - yGuess
        self.sigma = np.sqrt((1 / X.shape[0]) * np.sum(error ** 2))
        return (self.reg, self.sigma)

    def predict(self, X):

        preds = self.reg.predict(X)

        step = BINS_RANGE * 2 / self.nb_bins

        # We build bins from -3 sigma to 3 sigma
        bins = [i * step - BINS_RANGE for i in range(self.nb_bins + 1)]
        bins = np.array(bins) * self.sigma

        prob = []
        for i in range(len(bins) - 1):
            prob.append(quad(norm.pdf, bins[i], bins[i + 1], args=(0, self.sigma,))[0])

        bins_matrix = []
        for pred in preds:
            bins_matrix.append(bins + pred)
        bins_matrix = np.array(bins_matrix)

        preds_proba = np.expand_dims(np.array([prob] * len(X)), axis=1)
        bins_matrix = np.expand_dims(bins_matrix, axis=1)

        return preds_proba , bins_matrix
