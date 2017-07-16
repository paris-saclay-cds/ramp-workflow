"""A time series feature extractor followed by a regressor.

Train and test a time series feature extractor followed by a regressor.

The input object is an `xarray` `Dataset`, containing possibly several
`DataArrays` corresponding to the input sequence. It contains a special burn
in period in the beginning (carried by X_ds.n_burn_in) for which we do not
give ground truth and we do not require the user to provide predictions.
The ground truth sequence `y_array` in train and the output of the user
submission `ts_fe.transform` are thus `n_burn_in` shorter than the input
sequence `X_ds`, making the training and testing slightly complicated.
"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause
import numpy as np
from .ts_feature_extractor import TimeSeriesFeatureExtractor
from .regressor import Regressor


class ElNino(object):
    def __init__(self, check_sizes, check_indexs, workflow_element_names=[
            'ts_feature_extractor', 'regressor']):
        self.element_names = workflow_element_names
        self.ts_feature_extractor_workflow = TimeSeriesFeatureExtractor(
            check_sizes, check_indexs, [self.element_names[0]])
        self.regressor_workflow = Regressor([self.element_names[1]])

    def train_submission(self, module_path, X_ds, y_array, train_is=None):
        """
        Train a time series feature extractor + regressor workflow.

        `X_ds` is `n_burn_in` longer than `y_array` since `y_array` contains
        targets without the initial burn in period. `train_is` are wrt
        `y_array`, so `X_ds` has to be _extended_ by `n_burn_in` when sent to
        the time series feature extractor.

        """
        if train_is is None:
            # slice doesn't work here because of the way `extended_train_is`
            # is computed below
            train_is = np.arange(len(y_array))
        ts_fe = self.ts_feature_extractor_workflow.train_submission(
            module_path, X_ds, y_array)

        n_burn_in = X_ds.n_burn_in
        # X_ds contains burn-in so it needs to be extended by n_burn_in
        # timesteps. This assumes that train_is is a block of consecutive
        # time points.
        burn_in_range = np.arange(train_is[-1], train_is[-1] + n_burn_in)
        extended_train_is = np.concatenate((train_is, burn_in_range))
        X_train_ds = X_ds.isel(time=extended_train_is)
        # At this point X_train_ds is n_burn_in longer than y_array[train_is]

        # ts_fe.transform should return an array corresponding to time points
        # without burn in, so X_train_array and y_array[train_is] should now
        # have the same length.
        X_train_array = self.ts_feature_extractor_workflow.test_submission(
            ts_fe, X_train_ds)

        reg = self.regressor_workflow.train_submission(
            module_path, X_train_array, y_array[train_is])
        return ts_fe, reg

    def test_submission(self, trained_model, X_ds):
        ts_fe, reg = trained_model
        X_test_array = self.ts_feature_extractor_workflow.test_submission(
            ts_fe, X_ds)
        y_pred = self.regressor_workflow.test_submission(reg, X_test_array)
        return y_pred
