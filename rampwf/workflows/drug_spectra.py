import numpy as np
from .feature_extractor_classifier import FeatureExtractorClassifier
from .feature_extractor_regressor import FeatureExtractorRegressor

# I had to rename the classes in feature_extractor_clf.py and
# feature_extractor_reg.py in to FeatureExtractor() from
# FeatureExtractorClf and FeatureExtractorReg. This makes
# the semantics clearer, but use test in the notebook will only
# work for executing from the command line (otherwsie there is a class
# name clash).


class DrugSpectra(object):
    def __init__(self, workflow_element_names=[
            'feature_extractor_clf', 'classifier',
            'feature_extractor_reg', 'regressor']):
        self.workflow_element_names = workflow_element_names
        self.feature_extractor_classifier_workflow =\
            FeatureExtractorClassifier(self.workflow_element_names[:2])
        self.feature_extractor_regressor_workflow =\
            FeatureExtractorRegressor(self.workflow_element_names[2:])

    def train_submission(self, module_path, X_df, y_array, train_idxs=None):
        if train_idxs is None:
            train_idxs = slice(None, None, None)
        X_train_df = X_df.iloc[train_idxs]
        y_train_array = y_array[train_idxs]
        y_train_clf_array = y_train_array[:, 0]
        y_train_reg_array = y_train_array[:, 1].astype(float)
        fe_clf, clf = self.feature_extractor_classifier_workflow.\
            train_submission(module_path, X_train_df, y_train_clf_array)

        # Concatenating ground truth y_proba (on-hot, derived from labels)
        # to X_train_df.
        # This makes it vulnerable to training sets that don't contain
        # all the classes. So better to use it with stratified CV.
        labels = np.unique(y_array[:, 0])
        for i, label in enumerate(labels):
            X_train_df.loc[:, label] = (y_train_clf_array == label)
        fe_reg, reg = self.feature_extractor_regressor_workflow.\
            train_submission(module_path, X_train_df, y_train_reg_array)

        # It's a bit ugly that we return the labels here, but I don't see
        # a better solution
        return labels, fe_clf, clf, fe_reg, reg

    def test_submission(self, trained_model, X_df):
        labels, fe_clf, clf, fe_reg, reg = trained_model
        y_proba_clf = self.feature_extractor_classifier_workflow.\
            test_submission((fe_clf, clf), X_df)
        # Concatenating ground y_proba to X_train_df.
        for i, label in enumerate(labels):
            X_df.loc[:, label] = y_proba_clf[:, i]
        y_pred_reg = self.feature_extractor_regressor_workflow.\
            test_submission((fe_reg, reg), X_df)
        return np.concatenate([y_proba_clf, y_pred_reg.reshape(-1, 1)], axis=1)

workflow = DrugSpectra()
