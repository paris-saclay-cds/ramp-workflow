import os

import pandas as pd
from sklearn.base import is_classifier

from ..utils.importing import import_module_from_source


class SKLearnPipeline(object):
    def __init__(self, fname):
        self.fname = fname

    def train_submission(self, module_path, X_df, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        # import files
        pipeline = import_module_from_source(
            os.path.join(module_path, self.fname + '.py'),
            self.fname
        )
        pipeline = pipeline.get_estimator()
        if isinstance(X_df, pd.DataFrame):
            X_train = X_df.iloc[train_is]
        else:
            X_train = X_df[train_is]
        return pipeline.fit(X_train, y_array[train_is])

    def test_submission(self, trained_model, X_df):
        if is_classifier(trained_model):
            return trained_model.predict_proba(X_df)
        return trained_model.predict(X_df)
