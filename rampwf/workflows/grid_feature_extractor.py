"""
Extract features from a set of spatially gridded data.

This feature extractor can be used for any problem in which one has a series of
spatial grids arranged in time, y, x dimensions. The xarray library is used
to read the data into a Dataset, which is represented by X_ds.
y_array can be a set of binary labels or a regressor value. The values in
y_array are assumed to be paired exactly with the values in the first
dimension of X_ds.
"""

import os

import pandas as pd

from ..utils.importing import import_module_from_source


class GridFeatureExtractor(object):
    def __init__(self, workflow_element_names=['feature_extractor']):
        self.element_names = workflow_element_names

    def train_submission(self, module_path, X_ds, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        feature_extractor = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=True
        )
        fe = feature_extractor.FeatureExtractor()
        dim_set = pd.Series(list(X_ds.dims.keys()))
        time_dim = dim_set[dim_set.str.contains("time")][0]
        fe.fit(X_ds.isel(**{time_dim: train_is}), y_array[train_is])
        return fe

    def test_submission(self, trained_model, X_df):
        fe = trained_model
        X_test_array = fe.transform(X_df)
        return X_test_array
