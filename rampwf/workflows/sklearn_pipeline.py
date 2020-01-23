import os

from sklearn.base import is_classifier
from sklearn.utils import _safe_indexing

from ..utils.importing import import_module_from_source


class SKLearnPipeline(object):
    def __init__(self, fname='estimator'):
        self.fname = fname

    def train_submission(self, module_path, X, y, train_idx=None):
        train_idx = slice(None, None, None) if train_idx is None else train_idx
        pipeline = import_module_from_source(
            os.path.join(module_path, self.fname + '.py'),
            self.fname
        )
        pipeline = pipeline.get_estimator()
        X_train = _safe_indexing(X, train_idx)
        y_train = _safe_indexing(y, train_idx)
        return pipeline.fit(X_train, y_train)

    def test_submission(self, estimator_fitted, X):
        if is_classifier(estimator_fitted):
            return estimator_fitted.predict_proba(X)
        return estimator_fitted.predict(X)
