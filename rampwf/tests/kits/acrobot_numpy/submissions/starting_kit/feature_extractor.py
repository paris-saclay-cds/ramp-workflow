import numpy as np


class FeatureExtractor:
    def __init__(self):
        """
        Parameters
        ----------
        restart_name : str
            The name of the 0/1 column indicating restarts in the time series.
        """

    def fit(self, X, y):
        pass

    def transform(self, X):
        """Transform time series into list of states.
        We use the observables at time t as the state

        Be careful not to use any information from the future (X_ds[t + 1:])
        when constructing X_df[t].
        Parameters
        ----------
        X_df : pandas DataFrame
            The raw time series.
        Return
        ------
        X_df : pandas Dataframe

        """
        cos_sin = []
        for i in range(2):
            theta_i = X[:, i]
            cos_sin.append(np.cos(theta_i).reshape(-1, 1))
            cos_sin.append(np.sin(theta_i).reshape(-1, 1))

        cos_sin_array = np.concatenate(cos_sin, axis=1)
        X = np.concatenate([X[:, :-1], cos_sin_array], axis=1)
        X[:, [0, 3]] = X[:, [3, 0]]  # to match acrobot_ts_generative_regression kit

        return X
