"""A time series feature extractor.

Train and test a time series feature extractor.

The input object is an `xarray` `Dataset`, containing possibly several
`DataArrays` corresponding to the input sequence. It contains a special burn
in period in the beginning (carried by X_ds.n_burn_in) for which we do not
give ground truth and we do not require the user to provide predictions.
The ground truth sequence `y_array` in train and the output of the user
submission `ts_fe.transform` are thus `n_burn_in` shorter than the input
sequence `X_ds`, making the training and testing slightly complicated.

The other particularity of this workflow is that the input `X_ds` that the
*test* receives may contain information about the (future) labels, so it is
technically possible to cheat. We developed a randomized technique to
safeguard against this. The workflow has two init parameters: `check_sizes`
and `check_indexs`. Both are lists of indices. The idea is that we first
run `transform` on the original `X_ds`, obtaining the feature matrix
`X`. Then we randomly change elements of `X_ds` after
`n_burn_in + check_index`, and then check if the features in the new
`X_check_array` change *before* `n_burn_in + check_index` wrt `X`.
If they do, the submission is illegal. If they don't, it is possible that
the user carefully avoided looking ahead at this prticular index, so we may
test at another index, to be added to the list `check_indexs`. The other
list `check_sizes` makes it possible to make a shorter copy of the full
sequence in this check, to save time. Obviously each `check_size` should be
bigger than the corresponding `check_index`.

"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import os

import numpy as np
import pandas as pd
import xarray as xr

from ..utils.importing import import_module_from_source


class TimeSeriesFeatureExtractor(object):
    """
    restart_name should be None, or a one item list containing 1 on the timestep
    where time continuity is broken (e.g. a system restart)
    """
    def __init__(self, check_sizes=None, check_indexs=None,
                 restart_name=None, n_burn_in=0, n_lookahead=1,
                 workflow_element_names=['ts_feature_extractor']):
        self.element_names = workflow_element_names
        self.check_sizes = check_sizes
        self.check_indexs = check_indexs
        self.restart_name = restart_name
        self.n_burn_in = n_burn_in
        self.n_lookahead = n_lookahead

    def train_submission(self, module_path, X, y_array, train_is=None):
        """
        Train a time series feature extractor.

        `X` is `n_burn_in` longer than `y_array` since `y_array` contains
        targets without the initial burn in period. `train_is` are wrt
        `y_array`, so `X` has to be _extended_ by `n_burn_in` when sent
        to `ts_fe.fit`.

        """
        if (not isinstance(X, pd.DataFrame) and
                not isinstance(X, xr.Dataset)):
            raise ValueError('X must either be an xarray Dataset or a '
                             'pandas DataFrame')

        if train_is is None:
            # slice doesn't work here because of the way `extended_train_is`
            # is computed below
            train_is = np.arange(len(y_array))
        ts_feature_extractor = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=True
        )

        ts_fe = ts_feature_extractor.FeatureExtractor(
            self.restart_name, self.n_burn_in, self.n_lookahead)
        # Fit is not required in the submissions but we add it here in case
        # of, e.g., a recurrent neural net which is impossible to train once
        # the features are digested into a classical tabular format (one row
        # per time step).
        try:
            burn_in_range = np.arange(
                train_is[-1] + 1, train_is[-1] + 1 + self.n_burn_in)
            extended_train_is = np.concatenate((train_is, burn_in_range))
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[extended_train_is]
            elif isinstance(X, xr.Dataset):
                X_train = X.isel(time=extended_train_is)
            y_train = y_array[train_is]
            ts_fe.fit(X_train, y_train)
        except (AttributeError, IndexError, ValueError) as FitError:
            pass
        return ts_fe

    def test_submission(self, trained_model, X):
        """
        Test a time series feature extractor.

        `X` is `n_burn_in` longer than `X_test_array` below since
        `X_test_array` contains feautures only beyond the initial burn in
        period.

        We check if the `transform` of the submission looks ahead into the
        future.
        """
        if (not isinstance(X, pd.DataFrame) and
                not isinstance(X, xr.Dataset)):
            raise ValueError('X must either be an xarray Dataset or a '
                             'pandas DataFrame')
        ts_fe = trained_model
        X_test_tf = ts_fe.transform(X)

        # Checking if feature extractor looks ahead: we change the input
        # array after index n_burn_in + check_index, and check if the first
        # check_size features have changed
        n_burn_in = self.n_burn_in
        if self.check_sizes is not None and self.check_indexs is not None:
            for check_size, check_index in zip(
                    self.check_sizes, self.check_indexs):
                # We use a short prefix to save time
                # and add random noise to future.
                if isinstance(X, pd.DataFrame):
                    X_check_array = (X.iloc[:n_burn_in + check_size]
                                     .copy(deep=True))
                    column_names = list(X.columns)
                    if self.restart_name is not None:
                        column_names.remove(self.restart_name)
                    # get indices to avoid SettingWithCopyWarning
                    ind_col = [X_check_array.columns.get_loc(col)
                               for col in column_names]
                    start_row = n_burn_in + check_index
                    X_check_array.iloc[
                        start_row:, ind_col] += np.random.normal()

                elif isinstance(X, xr.Dataset):
                    X_check_array = X.isel(
                        time=slice(0, n_burn_in + check_size)).copy(deep=True)
                    # Assigning Dataset slices is not yet supported so we need
                    # to iterate over the arrays. To generalize we should maybe
                    # check the types.
                    column_names = list(X.data_vars.keys())
                    if self.restart_name is not None:
                        column_names.remove(self.restart_name)
                    for col_name in column_names:
                        time_slice = slice(n_burn_in + check_index, None)
                        X_check_array[col_name][
                            dict(time=time_slice)] += np.random.normal()

                # Calling transform on changed future.
                X_check_array_tf = ts_fe.transform(X_check_array)
                X_neq = np.not_equal(
                    X_test_tf[:check_size], X_check_array_tf[:check_size])
                x_neq = np.any(X_neq, axis=1)
                x_neq_nonzero = x_neq.nonzero()
                if len(x_neq_nonzero[0]) == 0:  # no change anywhere
                    first_modified_index = check_index
                else:
                    first_modified_index = np.min(x_neq_nonzero)
                # Normally, the features should not have changed before
                # check_index
                if first_modified_index < check_index:
                    message = 'The feature extractor looks into the' +\
                        ' future by at least {} time steps'.format(
                            check_index - first_modified_index)
                    raise AssertionError(message)
        return X_test_tf
