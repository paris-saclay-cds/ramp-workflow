from pathlib import Path
from ..utils.importing import import_module_from_source

class FeatureExtractorClassifierWithEDA(object):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'classifier']):
        self.element_names = workflow_element_names

    def train_submission(self, module_path, X_and_eda, y, train_is=None,
                         prev_trained_model=None):
        if train_is is None:
            train_is = slice(None, None, None)

        feature_extractor = import_module_from_source(
            Path(module_path) / f'{self.element_names[0]}.py',
            self.element_names[0])
        X = X_and_eda[0]
        eda = X_and_eda[1]
        X = X.copy()
        fe = feature_extractor.FeatureExtractor(eda)
        fe.fit(X.iloc[train_is], y[train_is].ravel())
        X_tr = fe.transform(X.iloc[train_is])

        classifier = import_module_from_source(
            Path(module_path) / f'{self.element_names[1]}.py',
            self.element_names[1],
        )
        clf = classifier.Classifier(eda)
        if prev_trained_model is None:
            clf.fit(X_tr, y[train_is].ravel())
        else:
            clf.fit(X_tr, y[train_is].ravel(), prev_trained_model[1])

        return fe, clf

    def test_submission(self, trained_model, X_and_eda):
        fe, clf = trained_model
        X = X_and_eda[0]
        X_tr = fe.transform(X)
        y_proba = clf.predict_proba(X_tr)
        return y_proba
