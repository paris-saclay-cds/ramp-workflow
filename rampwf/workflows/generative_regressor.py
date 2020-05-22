import numpy as np
import json
import collections
import os
from sklearn.utils.validation import check_random_state

from ..utils.importing import import_module_from_source
from ..utils import distributions_dispatcher


class GenerativeRegressor(object):
    """Build one generative regressor per target dimension.

    By default, this is done in an autoregressive way in which case the target
    dimension j also uses the value of the target dimension j-1 as inputs.
    This autoregressive decomposition is based on the chain rule:
    p(y_1, y_2, ...) = p(y_1) * p(y_2 | y_1) * ...

    The regressors are parametrized as mixture distributions and are expected
    to return :
        weights: The importance of each of the distributions.
            They should sum up to one at each timestep.
        types: The type of distributions, ordered in the same fashion
            as params.
                0 is Gaussian
                1 is Uniform
        params: The parameters that describe the distributions.
            For Gaussians, the order expected is the mean mu and standard
            deviation sigma.
            For uniform, the support bounds a and b.

    Parameters
    ----------
    autoregressive : bool
        Whether to build the regressors using an autoregressive scheme. If
        true, the order to use for the autoregressive decomposition can be
        specified in an order.json file located in the submission folder. If
        there is no such file, the default order is used.
        If autoregressive is set to False, the generative regressor of each
        target is built without the knowledge of the other target values.

    max_dists: the maximum number of distributions a generative
        regressor can output for its returned mixture.
    """
    def __init__(self, target_column_name, max_dists,
                 check_sizes=None, check_indexs=None,
                 workflow_element_names=['generative_regressor'],
                 restart_name=None, autoregressive=True,
                 **kwargs):
        self.check_indexs = check_indexs
        self.check_sizes = check_sizes
        self.element_names = workflow_element_names
        self.target_column_name = target_column_name
        self.max_dists = max_dists
        self.restart_name = restart_name
        self.kwargs = kwargs
        self.autoregressive = autoregressive

    def _check_restart(self, X_array, train_is=slice(None, None, None)):
        restart = None
        if self.restart_name is not None:
            try:
                restart = X_array[self.restart_name].values
                X_array = X_array.drop(columns=self.restart_name)
                restart = restart[train_is,]
            except KeyError:
                restart = None

        return X_array, restart

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        gen_regressor = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=False
        )

        order_path = os.path.join(module_path, 'order.json')

        try:
            with open(order_path, "r") as json_file:
                order = json.load(json_file)
                # Check if the names in the order and observables are all here
                if set(order.keys()) == set(self.target_column_name):
                    # We sort the variable names by user-defined order
                    order = [k for k,_ in sorted(
                        order.items(), key=lambda item: item[1])]
                    # Map it to original order
                    order = [self.target_column_name.index(i) for i in order]
                    print(order)
                    y_array = y_array[:, order]
                else:
                    raise RuntimeError("Order variables are not correct")
        except FileNotFoundError as e:
            print("Using default order")
            order = range(len(self.target_column_name))

        truths = ["y_" + t for t in self.target_column_name]
        X_array = X_array.copy()

        X_array, restart = self._check_restart(X_array, train_is)

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

            if self.autoregressive:
                X_array = np.hstack([X_array, y])
            regressors.append(reg)
        return regressors, order

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
        regressors, order = trained_model
        dims = []
        n_columns = X_array.shape[1]
        X_array = X_array.copy()
        n_regressors = len(regressors)

        X_array, restart = self._check_restart(X_array)

        if self.restart_name is not None:
                n_columns -= len(self.restart_name)

        truths = ["y_" + t for t in self.target_column_name]

        if type(X_array).__module__ != np.__name__:
            y = X_array[truths]
            X_array.drop(columns=truths, inplace=True)
            if self.autoregressive:
                X_array = np.hstack([X_array.values, y.values[:, order]])
            else:
                X = X_array.values

        for i, reg in enumerate(regressors):
            if self.autoregressive:
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

        dims_original_order = np.array(dims)[np.argsort(order)]
        preds_concat = np.concatenate(dims_original_order, axis=1)

        return preds_concat

    def check_cheat(self, trained_model, X_array):
        if not self.check_sizes is None and not self.check_indexs is None:
            for check_size, check_index in zip(
                    self.check_sizes, self.check_indexs):
                X_check = X_array.iloc[:check_size].copy()
                # Adding random noise to future.
                original_predict = self.predict_submission(
                    trained_model, X_check)
                X_check.iloc[check_index] += np.random.normal()
                # Calling predict on changed future.
                X_check_array = self.predict_submission(
                    trained_model, X_check)
                X_neq = np.not_equal(
                    original_predict[:check_size],
                    X_check_array[:check_size])
                x_neq = np.any(X_neq, axis=1)
                x_neq_nonzero = x_neq.nonzero()
                if len(x_neq_nonzero[0]) == 0:  # no change anywhere
                    first_modified_index = check_index
                else:
                    first_modified_index = np.min(x_neq_nonzero)
                # Normally, the features should not have changed before
                # check_index
                if first_modified_index < check_index:
                    message = 'The generative_regressor looks into the' +\
                        ' future by at least {} time steps'.format(
                            check_index - first_modified_index)
                    raise AssertionError(message)

    def step(self, trained_model, X_array, random_state=None):
        """Careful, for now, for every x in the time dimension, we will
        sample a y. To sample only one y, provide only one X.
        If X is not a panda array, the assumed order is the same as
        given in training

        X_array must contain only one timestep.
        """
        rng = check_random_state(random_state)
        regressors, order = trained_model

        X_array, restart = self._check_restart(X_array)
        n_features_init = X_array.shape[1]

        if self.autoregressive:
            # preallocate array by concatenating with unknown predicted array
            predicted_array = np.full((1, len(regressors)), fill_value=np.nan)
            X = np.concatenate([X_array.to_numpy(), predicted_array], axis=1)
            X_used = X[:, :n_features_init]
        else:
            X_used = X_array.to_numpy()

        y_sampled = np.zeros(len(regressors))
        for i, reg in enumerate(regressors):
            if i >= 1 and self.autoregressive:
                X[:, n_features_init + (i-1)] = y_sampled[i-1]
                X_used = X[:, :n_features_init + i]

            if restart is not None:
                dists = reg.predict(X_used, restart)

            else:
                dists = reg.predict(X_used)

            weights, types, params = dists
            n_dists = types.shape[1]
            w = weights[0].ravel()
            w = w / sum(w)
            selected = rng.choice(n_dists, p=w)
            dist = distributions_dispatcher(int(types[0, selected]))
            selected_type = int(types[0, selected])

            sel_id = (distributions_dispatcher(selected_type).nb_params *
                      selected)
            y_sampled[i] = dist.sample(params[0, sel_id:sel_id+dist.nb_params])

        y_sampled = np.array(y_sampled)[np.argsort(order)]
        return y_sampled[np.newaxis, :]
