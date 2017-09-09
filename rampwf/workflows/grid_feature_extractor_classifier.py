from .grid_feature_extractor import GridFeatureExtractor
from .classifier import Classifier
import numpy as np

class GridFeatureExtractorClassifier(object):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'classifier']):
        self.element_names = workflow_element_names
        self.feature_extractor_workflow = GridFeatureExtractor(
            [self.element_names[0]])
        self.classifier_workflow = Classifier([self.element_names[1]])

    def train_submission(self, module_path, X_ds, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        fe = self.feature_extractor_workflow.train_submission(
            module_path, X_ds, y_array, train_is)
        X_train_array = self.feature_extractor_workflow.test_submission(
            fe, X_ds.isel(enstime=np.where(train_is)[0]))
        clf = self.classifier_workflow.train_submission(
            module_path, X_train_array, y_array[train_is])
        return fe, clf

    def test_submission(self, trained_model, X_ds):
        fe, clf = trained_model
        X_test_array = self.feature_extractor_workflow.test_submission(
            fe, X_ds)
        y_proba = self.classifier_workflow.test_submission(clf, X_test_array)
        return y_proba
