import imp
import numpy as np


class TimeSeriesFeatureExtractor(object):
    def __init__(self, check_sizes, check_indexs, workflow_element_names=[
            'ts_feature_extractor']):
        self.element_names = workflow_element_names
        self.check_sizes = check_sizes
        self.check_indexs = check_indexs

    def train_submission(self, module_path, X_ds, y_array, train_is=None):
        """Since there is no `fit`, this just returns the new ts_fe object."""
        submitted_ts_feature_extractor_file = '{}/{}.py'.format(
            module_path, self.element_names[0])
        ts_feature_extractor = imp.load_source(
            '', submitted_ts_feature_extractor_file)
        ts_fe = ts_feature_extractor.FeatureExtractor()
        return ts_fe

    def test_submission(self, trained_model, X_ds):
        # Don't forget that the length of the feature vector X_test_array
        # is shorter than X_ds by n_burn_in
        ts_fe = trained_model
        X_test_array = ts_fe.transform(X_ds)

        # Checking if feature extractor looks ahead: we change the input
        # array after index n_burn_in + check_index, and check if the first
        # check_size features have changed
        n_burn_in = X_ds.attrs['n_burn_in']
        for check_size, check_index in zip(
                self.check_sizes, self.check_indexs):
            # We use a short prefix to save time
            X_check_ds = X_ds.isel(
                time=slice(0, n_burn_in + check_size)).copy(deep=True)
            # Adding random noise to future.
            # Assigning Dataset slices is not yet supported so we need to
            # iterate over the arrays. To generalize we should maybe check
            # the types.
            data_var_names = X_check_ds.data_vars.keys()
            for data_var_name in data_var_names:
                X_check_ds[data_var_name][dict(time=slice(
                    n_burn_in + check_index, None))] += np.random.normal()
            # Calling transform on changed future.
            X_check_array = ts_fe.transform(X_check_ds)
            X_neq = np.not_equal(
                X_test_array[:check_size], X_check_array[:check_size])
            x_neq = np.any(X_neq, axis=1)
            x_neq_nonzero = x_neq.nonzero()
            if len(x_neq_nonzero[0]) == 0:  # no change anywhere
                first_modified_index = check_index
            else:
                first_modified_index = np.min(x_neq_nonzero)
            # Normally, the features should not have changed before check_index
            if first_modified_index < check_index:
                message = 'The feature extractor looks into the future by' +\
                    ' at least {} time steps'.format(
                        check_index - first_modified_index)
                raise AssertionError(message)

        return X_test_array
