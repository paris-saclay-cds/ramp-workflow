from ..utils.importing import import_file
from sklearn.pipeline import Pipeline


class sklearn_pipeline(object):
    def __init__(self, file_name, is_proba):
        self.file_name = file_name
        self.is_proba = is_proba

    def train_submission(self, module_path, X_df, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        # import files
        pipeline = import_file(module_path, self.file_name)
        pipeline = pipeline.get_pipeline()
        self.pipeline = pipeline
        return pipeline.fit(X_df[train_is], y_array[train_is])

    def test_submission(self, trained_model, X_df):
        if self.is_proba:
            y = trained_model.predict_proba(X_df)
        else:
            y = trained_model.predict(X_df)
        return y
