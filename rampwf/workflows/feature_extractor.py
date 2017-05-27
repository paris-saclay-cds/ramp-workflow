from importlib import import_module


class FeatureExtractor(object):
    def __init__(self, workflow_element_names=['feature_extractor']):
        self.workflow_element_names = workflow_element_names

    def train_submission(self, module_path, X_df, y_array, train_is=None):
        if train_is is None:
            train_is = range(len(y_array))
        submitted_feature_extractor_module = '.{}'.format(
            self.workflow_element_names[0])
        feature_extractor = import_module(
            submitted_feature_extractor_module, module_path)
        fe = feature_extractor.FeatureExtractor()
        fe.fit(X_df.iloc[train_is], y_array[train_is])
        return fe

    def test_submission(self, trained_model, X_df):
        fe = trained_model
        X_test_array = fe.transform(X_df)
        return X_test_array

workflow = FeatureExtractor()
