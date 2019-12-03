from ..utils.importing import import_file


class workflow(object):
    def __init__(self, workflow_element_names):
        self.element_names = workflow_element_names

    def train_submission(self, module_path, X_df, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        # import files
        solution_files = {filename: import_file(module_path, filename) \
                          for filename in self.element_names}

        if 'feature_extractor' in solution_files:
            fe = solution_files['feature_extractor'].FeatureExtractor()
            fe = fe.fit(X_df.iloc[train_is], y_array[train_is])
            X_train_array = fe.transform(X_df.iloc[train_is])
        else:
            fe = None
            X_train_array = X_df.iloc[train_is]

        if 'classifier' in solution_files:
            clf = solution_files['classifier'].Classifier()
            clf.fit(X_train_array, y_array[train_is])
        else:
            clf = None

        if 'regressor' in solution_files:
            reg = solution_files['regressor'].Regressor()
            reg.fit(X_train_array, y_array[train_is])
        else:
            reg = None

        return fe, clf, reg

    def test_submission(self, trained_model, X_df):
        fe, clf, reg = trained_model

        if fe is not None:
            X_test_array = fe.transform(X_df)
        else X_test_array = X_df

        if clf is not None:
            y_pred = clf.predict_proba(X_test_array)

        if reg is not None:
            y_pred = reg.predict(X_test_array)

        return y_pred