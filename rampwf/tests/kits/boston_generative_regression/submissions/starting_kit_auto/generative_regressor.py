import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression


class GenerativeRegressor(BaseEstimator):
    def __init__(self, max_dists, target_dim):
        """
        Parameters
        ----------
        max_dists : int
            The maximum number of distributions (kernels) in the mixture.
        target_dim : int
            The index of the target column to be predicted.
        """
        self.decomposition = 'autoregressive'

    def fit(self, X_array, y_array):
        """Linear regression + residual sigma.

        For an autoregressive submission, as the target joint distribution is
        decomposed using the chain rule, fit is called once for each target
        dimension.

        Parameters
        ----------
        X_array : pandas.DataFrame
            The input array. The features extracted by the feature extractor.
            When fitting on the target dimension j, the values of the previous
            target dimensions (<j) are used as additional inputs in X_array.

        y_array : numpy array, shape (n_samples, 1)
            The ground truth array (system observables of time step t+1 to be
            predicted from observables before time step t).
            As fit is called once for each target dimension, y_array contains
            only one target dimension and must be of shape (n_samples, 1).
        """
        self.reg = LinearRegression()
        self.reg.fit(X_array, y_array)
        y_pred = self.reg.predict(X_array)
        residuals = y_array - y_pred
        # Estimate a single sigma from residual variance
        if residuals.all() == 0:
            print('WARNING: all residuals are 0 in linear regressor.')
        self.sigma = np.sqrt(
            (1 / (X_array.shape[0] - 1)) * np.sum(residuals ** 2))

    def predict(self, X_array):
        """Construct a one dimensional conditional mixture distribution.

        Be careful not to use any information from the future
        (X_array[t + 1:]) when constructing the output.

        Parameters
        ----------
        X_array : pandas.DataFrame
            The input array. When estimating the distribution of the target
            dimension j, the values of the previous target dimensions (<j) are
            used as additional inputs in X_array.

        Return
        ------
        XXX FIXME with mixture object
        weights : np.array of float
            discrete probabilities of each component of the mixture
        types : np.array of int
            integer codes referring to component types
            see rampwf.utils.distributions_dict
        params : np.array of float tuples
            parameters for each component in the mixture
        """
        types = ["norm"]
        y_pred = self.reg.predict(X_array)  # means
        sigmas = np.array([self.sigma] * len(X_array))  # constant sigma
        sigmas = sigmas[:, np.newaxis]
        params = np.concatenate((y_pred, sigmas), axis=1)
        weights = np.array([[1.0], ] * len(X_array))
        return weights, types, params
