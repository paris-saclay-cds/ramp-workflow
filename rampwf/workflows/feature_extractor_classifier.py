from .feature_extractor import FeatureExtractor
from .classifier import Classifier


class FeatureExtractorClassifier(object):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'classifier']):
        self.element_names = workflow_element_names
        self.feature_extractor_workflow = FeatureExtractor(
            [self.element_names[0]])
        self.classifier_workflow = Classifier([self.element_names[1]])

    def train_submission(self, module_path, X_df, y_array, train_idxs=None):
        if train_idxs is None:
            train_idxs = slice(None, None, None)
        fe = self.feature_extractor_workflow.train_submission(
            module_path, X_df, y_array, train_idxs)
        X_train_array = self.feature_extractor_workflow.test_submission(
            fe, X_df.iloc[train_idxs])
        clf = self.classifier_workflow.train_submission(
            module_path, X_train_array, y_array[train_idxs])
        return fe, clf

    def test_submission(self, trained_model, X_df):
        fe, clf = trained_model
        X_test_array = self.feature_extractor_workflow.test_submission(
            fe, X_df)
        y_proba = self.classifier_workflow.test_submission(clf, X_test_array)
        return y_proba
