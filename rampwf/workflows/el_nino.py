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
        if train_is is None:
            train_is = slice(None, None, None)
        ts_fe = self.ts_feature_extractor_workflow.train_submission(
            module_path, X_ds, y_array, train_is)
        n_burn_in = X_ds.attrs['n_burn_in']
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

    def test_submission(self, trained_model, X_df):
        ts_fe, reg = trained_model
        X_test_array = self.ts_feature_extractor_workflow.test_submission(
            ts_fe, X_df)
        y_pred = self.regressor_workflow.test_submission(reg, X_test_array)
        return y_pred
