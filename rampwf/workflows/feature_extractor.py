import imp


class FeatureExtractor(object):
    def __init__(self, workflow_element_names=['feature_extractor']):
        self.element_names = workflow_element_names

    def train_submission(self, module_path, X_df, y_array, train_idxs=None):
        if train_idxs is None:
            train_idxs = slice(None, None, None)
        submitted_feature_extractor_file = '{}/{}.py'.format(
            module_path, self.element_names[0])
        feature_extractor = imp.load_source(
            '', submitted_feature_extractor_file)
        fe = feature_extractor.FeatureExtractor()
        fe.fit(X_df.iloc[train_idxs], y_array[train_idxs])
        return fe

    def test_submission(self, trained_model, X_df):
        fe = trained_model
        X_test_array = fe.transform(X_df)
        return X_test_array
