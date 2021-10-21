from .feature_extractor import FeatureExtractor
from .classifier import Classifier


class FeatureExtractorClassifier(object):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'classifier']):
        self.element_names = workflow_element_names
        self.feature_extractor_workflow = FeatureExtractor(
            [self.element_names[0]])
        self.classifier_workflow = Classifier([self.element_names[1]])

    def train_submission(self, module_path, X_df, y_array, train_is=None,
                         prev_trained_model=None):
        if train_is is None:
            train_is = slice(None, None, None)
        fe = self.feature_extractor_workflow.train_submission(
            module_path, X_df, y_array, train_is)
        X_train_array = self.feature_extractor_workflow.test_submission(
            fe, X_df.iloc[train_is])
        prev_classifier = None if prev_trained_model is None\
            else prev_trained_model[1]
        clf = self.classifier_workflow.train_submission(
            module_path, X_train_array, y_array[train_is],
            prev_trained_model=prev_classifier)
        return fe, clf

    def test_submission(self, trained_model, X_df):
        fe, clf = trained_model
        X_test_array = self.feature_extractor_workflow.test_submission(
            fe, X_df)
        y_proba = self.classifier_workflow.test_submission(clf, X_test_array)
        return y_proba
