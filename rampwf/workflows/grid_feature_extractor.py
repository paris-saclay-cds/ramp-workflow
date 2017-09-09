import imp
import numpy as np

class GridFeatureExtractor(object):
    def __init__(self, workflow_element_names=['feature_extractor']):
        self.element_names = workflow_element_names

    def train_submission(self, module_path, X_ds, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        submitted_feature_extractor_file = '{}/{}.py'.format(
            module_path, self.element_names[0])
        feature_extractor = imp.load_source(
            '', submitted_feature_extractor_file)
        fe = feature_extractor.FeatureExtractor()
        fe.fit(X_ds.isel(enstime=np.where(train_is)[0]), y_array[train_is])
        return fe

    def test_submission(self, trained_model, X_df):
        fe = trained_model
        X_test_array = fe.transform(X_df)
        return X_test_array
