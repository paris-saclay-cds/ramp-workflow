from rampwf.utils.importing import import_file
import numpy as np
import pandas as pd


class GenerativeRegressorSelfGaussian(object):
    def __init__(self, target_column_name, max_gauss,
                 workflow_element_names=['gen_regressor_gauss'], restart_name=None,
                 **kwargs):
        self.element_names = workflow_element_names
        self.target_column_name = target_column_name
        self.max_gauss = max_gauss
        self.restart_name = restart_name
        self.kwargs = kwargs

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        gen_regressor = import_file(module_path, self.element_names[0])

        truths = ["y_" + t for t in self.target_column_name]
        X_array = X_array.copy()
        restart = None

        if self.restart_name is not None:
            try:
                restart = X_array[self.restart_name].values
                X_array = X_array.drop(columns=self.restart_name)
                restart = restart[train_is,]
            except KeyError:
                print("The generative regressor was not given information about"
                      " restarts, make sur this is intended.")
                restart = None

        if type(X_array).__module__ != np.__name__:
            try:
                X_array.drop(columns=truths, inplace=True)
            except KeyError:
                # We remove the truth from X, if present
                pass
            X_array = X_array.values
        X_array = X_array[train_is,]

        regressors = []
        for i in range(len(self.target_column_name)):
            reg = gen_regressor.GenerativeRegressorGaussian(self.max_gauss, **self.kwargs)

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

            if restart is not None:
                reg.fit(X_array, y, restart)
            else:
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
        n_columns = X_array.shape[1]
        X_array = X_array.copy()
        n_regressors = len(regressors)

        restart = None

        if self.restart_name is not None:
            try:
                restart = X_array[self.restart_name].values
                X_array = X_array.drop(columns=self.restart_name)
                n_columns -= len(self.restart_name)
            except KeyError:
                print("The generative regressor was not given information about"
                      " restarts, make sur this is intended.")
                restart = None

        truths = ["y_" + t for t in self.target_column_name]

        if type(X_array).__module__ != np.__name__:
            y = X_array[truths]
            X_array.drop(columns=truths, inplace=True)
            X_array = np.hstack([X_array.values, y.values])

        for i, reg in enumerate(regressors):
            X = X_array[:, :n_columns - n_regressors + i]
            if restart is not None:
                gaussians = reg.predict(X, restart)
            else:
                gaussians = reg.predict(X)
            dims.append(gaussians)

        return np.concatenate(dims, axis=1)

    def step(self, trained_model, X_array, seed=None):
        """Careful, for now, for every x in the time dimension, we will sample
        a y. To sample only one y, provide only one X.
        If X is not a panda array, the assumed order is the same as
        given in training"""
        regressors = trained_model
        y_sampled = []
        np.random.seed(seed)

        for i, reg in enumerate(regressors):
            X = X_array
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
                if i > 0:
                    X = np.concatenate((X, sampled_array), axis=1)

            gaussians = reg.predict(X)
            gaussians = gaussians.swapaxes(0, 1)

            for gaussian in gaussians:
                for nb_gaussians, item in enumerate(gaussian[0]):
                    # All the predictions must have the same number
                    # of gaussians for a given dimension
                    if np.isnan(item):
                        nb_gaussians -= 1
                        break
                nb_gaussians = int((nb_gaussians + 1) / 3)
                mus = gaussian[:, 0:nb_gaussians]
                sigmas = gaussian[:, nb_gaussians:nb_gaussians * 2]
                weights = gaussian[:, nb_gaussians * 2:nb_gaussians * 3]

                y_dim = []
                for i in range(len(mus)):
                    w = weights[i].ravel()
                    w = w / sum(w)
                    selected = np.random.choice(list(range(nb_gaussians)), p=w)
                    y_dim.append(np.random.normal(loc=mus[i, selected],
                                                  scale=sigmas[i, selected]))

                y_sampled.append(y_dim)

        return np.array(y_sampled).swapaxes(0, 1)
