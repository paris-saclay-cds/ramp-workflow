from ..utils.importing import import_file
from sklearn.pipeline import Pipeline


class Workflow(object):
    def __init__(self, workflow_element_names):
        self.element_names = workflow_element_names

    def train_submission(self, module_path, X_df, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        # import files
        solution_files = {filename: import_file(module_path, filename)
                            for filename in self.element_names}

        if 'feature_extractor' in solution_files:
            fe = solution_files['feature_extractor'].get_feature_extractor()
        else:
            fe = None

        if 'classifier' in solution_files:
            clf = solution_files['classifier'].get_classifier()
        else:
            clf = None

        if 'regressor' in solution_files:
            reg = solution_files['regressor'].get_regressor()
        else:
            reg = None

        steps = []
        for wkfl in (fe, clf, reg):
            if wkfl is not None:
                steps.append(wkfl)

        self.pipeline = Pipeline(steps=list(zip(self.element_names, steps)))
        return self.pipeline.fit(X_df.iloc[train_is], y_array[train_is])

    def test_submission(self, trained_model, X_df):
        if 'regressor' in self.element_names:
            return trained_model.predict(X_df)
        elif 'classifier' in self.element_names:
            return trained_model.predict_proba(X_df)
