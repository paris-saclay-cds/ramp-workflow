from .feature_extractor import FeatureExtractor
from .regressor import Regressor


class FeatureExtractorRegressor(object):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'regressor']):
        self.element_names = workflow_element_names
        self.feature_extractor_workflow = FeatureExtractor(
            [self.element_names[0]])
        self.regressor_workflow = Regressor([self.element_names[1]])

    def train_submission(self, module_path, X_df, y_array, train_is=None,
                         prev_trained_model=None):
        if train_is is None:
            train_is = slice(None, None, None)
        fe = self.feature_extractor_workflow.train_submission(
            module_path, X_df, y_array, train_is)
        X_train_array = self.feature_extractor_workflow.test_submission(
            fe, X_df.iloc[train_is])
        prev_regressor = None if prev_trained_model is None\
            else prev_trained_model[1]
        reg = self.regressor_workflow.train_submission(
            module_path, X_train_array, y_array[train_is],
            prev_trained_model=prev_regressor)
        return fe, reg

    def test_submission(self, trained_model, X_df):
        fe, reg = trained_model
        X_test_array = self.feature_extractor_workflow.test_submission(
            fe, X_df)
        y_pred = self.regressor_workflow.test_submission(reg, X_test_array)
        return y_pred
