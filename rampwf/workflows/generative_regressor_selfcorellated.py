from rampwf.utils.importing import import_file
import numpy as np
import pandas as pd


class GenerativeRegressorSelf(object):
    def __init__(self, target_column_name, nb_bins, workflow_element_names=['gen_regressor'], ):
        self.element_names = workflow_element_names
        self.target_column_name = target_column_name
        self.nb_bins = nb_bins

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        gen_regressor = import_file(module_path, self.element_names[0])

        if type(X_array).__module__ != np.__name__:
            X_array = X_array.values

        X_array = X_array[train_is,]

        regressors = []
        for i in range(len(self.target_column_name)):
            reg = gen_regressor.GenerativeRegressor(self.nb_bins)

            if i == 0 and y_array.shape[1] == 1:
                y = y_array[train_is]
            else:
                y = y_array[train_is, i]

            shape = y.shape
            if len(shape) == 1:
                y = y.reshape(-1, 1)
            elif len(shape) == 2:
                pass
            else:
                raise ValueError("More than two dims for y not supported")

            reg.fit(X_array, y)
            X_array = np.hstack([X_array, y])
            regressors.append(reg)
        return regressors

    def test_submission(self, trained_model, X_array):
        """Test submission, here we assume that the last i columns of
        X_array corespond to the ground truth, labeled with the target
        name self.target_column_name  and a y before (to avoid duplicate names
        in RL setup, where the target is a shifted column form the observation)
        in the same order, is a numpy object is provided. """
        regressors = trained_model
        dims = []
        for i, reg in enumerate(regressors):
            if type(X_array).__module__ == np.__name__:
                X = X_array[:i]
            else:
                to_drop = ["y_" + y for y in self.target_column_name[i:]]
                X = X_array.drop(columns=to_drop)
                X = X.values
            y_pred, bin_edges = reg.predict(X)
            dims.append(np.concatenate((bin_edges, y_pred), axis=2))

        return np.concatenate(dims, axis=1)

    def step(self, trained_model, X_array):
        """Careful, for now, for every x in the time dimension, we will sample
        a y. To sample only one y, provide only one X.
        If X is not a panda array, the assumed order is the same as
        given in training"""
        regressors = trained_model
        y_sampled = []

        # #   If we get a numpy array instead of a xarray
        #     X_array = pd.DataFrame([X_array, ])
        #     extra_actions = [act + "_extra" for act in action_names]
        #     X_array.columns = self.target_column_name + extra_actions + \
        #                       action_names

        for i, reg in enumerate(regressors):
            X = X_array.copy()
            if type(X_array).__module__ != np.__name__:
                for j, predicted_dim in enumerate(np.array(y_sampled)):
                    X["y_" + self.target_column_name[j]] = predicted_dim
                X = X.values
            if X.ndim == 1:
                X = [X, ]

            if type(X_array).__module__ == np.__name__:
                sampled_array = np.array(y_sampled).T
                if sampled_array.ndim == 1:
                    sampled_array = [sampled_array, ]
                if i>0:
                    X = np.concatenate((X, sampled_array), axis =1)

            y_pred, bin_edges = reg.predict(X)
            y_dim = []
            for prob, bins in zip(y_pred, bin_edges):
                prob = prob.ravel()
                bins = bins.ravel()
                selected = np.random.choice(list(range(len(prob))), p=prob)
                y_dim.append(np.random.uniform(low=bins[selected], high=bins[selected + 1]))
            y_sampled.append(y_dim)

        return np.array(y_sampled).swapaxes(0, 1)
