import numpy as np

from rampwf.utils.importing import import_file
from ..utils import distributions_dispatcher


class GenerativeRegressor(object):
    def __init__(self, target_column_name, max_dists, check_sizes, check_indexs,
                 workflow_element_names=['generative_regressor'],
                 restart_name=None,
                 **kwargs):
        """
        The regressors are expected to return :
            weights: The importance of each of the distributions.
                    They should sum up to one a each timestep
            types: The type of distributions, ordered in the same fashion
                    as params.
                        0 is Gaussian
                        1 is Uniform
             params: The parameters that describe the distributions.
                    If gaussian, the order expected is mu, sigma
                    If uniform, it is a and b

        max_dists: the maximum number of distributions a generative
                    regressor can output
        """
        self.check_indexs = check_indexs
        self.check_sizes = check_sizes
        self.element_names = workflow_element_names
        self.target_column_name = target_column_name
        self.max_dists = max_dists
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
            reg = gen_regressor.GenerativeRegressor(
                self.max_dists, i, **self.kwargs)

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
        original_predict = self.predict_submission(trained_model, X_array)

        self.check_cheat(trained_model, X_array)
        return original_predict

    def predict_submission(self, trained_model, X_array):
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
                restart = None

        truths = ["y_" + t for t in self.target_column_name]

        if type(X_array).__module__ != np.__name__:
            y = X_array[truths]
            X_array.drop(columns=truths, inplace=True)
            X_array = np.hstack([X_array.values, y.values])

        for i, reg in enumerate(regressors):
            X = X_array[:, :n_columns - n_regressors + i]
            if restart is not None:
                dists = reg.predict(X, restart)
            else:
                dists = reg.predict(X)

            weights, types, params = dists

            nb_dists_curr = types.shape[1]
            assert nb_dists_curr <= self.max_dists

            sizes = np.full((len(types), 1), nb_dists_curr)
            result = np.concatenate((sizes, weights, types, params), axis=1)

            dims.append(result)

        preds_concat = np.concatenate(dims, axis=1)

        return preds_concat

    def check_cheat(self, trained_model, X_array):
        for check_size, check_index in zip(
                self.check_sizes, self.check_indexs):
            X_check = X_array.iloc[:check_size].copy()
            # Adding random noise to future.
            original_predict = self.predict_submission(trained_model, X_check)
            X_check.iloc[check_index] += np.random.normal()
            # Calling predict on changed future.
            X_check_array = self.predict_submission(trained_model, X_check)
            X_neq = np.not_equal(
                original_predict[:check_size], X_check_array[:check_size])
            x_neq = np.any(X_neq, axis=1)
            x_neq_nonzero = x_neq.nonzero()
            if len(x_neq_nonzero[0]) == 0:  # no change anywhere
                first_modified_index = check_index
            else:
                first_modified_index = np.min(x_neq_nonzero)
            # Normally, the features should not have changed before check_index
            if first_modified_index < check_index:
                message = 'The generative_regressor looks into the future by' +\
                    ' at least {} time steps'.format(
                        check_index - first_modified_index)
                raise AssertionError(message)
        pass

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
                if len(y_sampled)==1:
                    X["y_" + self.target_column_name[0]] = y_sampled[0][0]
                else:
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

            dists = reg.predict(X)

            # TODO: rewrite sampling
            weights, types, params = dists
            nb_dists = types.shape[1]
            y_dim = []
            for i in range(len(types)): # Number of timesteps
                w = weights[i].ravel()
                w = w / sum(w)
                empty_dist = distributions_dispatcher()
                selected_type = empty_dist
                while selected_type == empty_dist:
                    selected = np.random.choice(list(range(nb_dists)), p=w)
                    dist = distributions_dispatcher(int(types[i, selected]))
                    selected_type = int(types[i, selected])
                sel_id = 0
                for k in range(selected):
                    firs_valid = np.where(
                                ~np.array(types[:, k] == empty_dist)
                                )[0][0]
                    sel_id += distributions_dispatcher(firs_valid).nb_params
                y_dim.append(
                    dist.sample(params[i, sel_id:sel_id+dist.nb_params])
                )
            y_sampled.append(y_dim)

        return np.array(y_sampled).swapaxes(0, 1)