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
        self.decomposition = None

    def fit(self, X_array, y_array):
        """Linear regression + residual sigma.

        Parameters
        ----------
        X_array : pandas.DataFrame
            The input array. The features extracted by the feature extractor,
            plus `target_dim` system observables from time step t+1.

        y_array : numpy array, shape (n_samples, n_targets)
            The ground truth array (system observables at time step t+1).
        """
        self.reg = LinearRegression()
        self.reg.fit(X_array, y_array)
        y_pred = self.reg.predict(X_array)
        residuals = y_array - y_pred
        # Estimate a single sigma from residual variance
        if residuals.all() == 0:
            print('WARNING: all residuals are 0 in linear regressor.')
        self.sigmas = np.sqrt(
            (1 / (X_array.shape[0] - 1)) * np.sum(residuals ** 2, axis=0))

    def predict(self, X_array):
        """Construct a multivariate conditional Gaussian mixture distribution.

        Be careful not to use any information from the future
        (X_array[t + 1:]) when constructing the output.

        Parameters
        ----------
        X_array : pandas.DataFrame
            The input array. The features extracted by the feature extractor,
            plus `target_dim` system observables from time step t+1.

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

        types = ['norm', 'norm']
        y_pred = self.reg.predict(X_array)  # means
        sigmas = np.array([self.sigmas] * len(X_array))  # constant sigma
        params = np.empty((len(y_pred), y_pred.shape[1] * 2))
        params[:, 0::2] = y_pred
        params[:, 1::2] = sigmas
        weights = np.array([[1., 1.], ] * len(X_array))
        return weights, types, params
