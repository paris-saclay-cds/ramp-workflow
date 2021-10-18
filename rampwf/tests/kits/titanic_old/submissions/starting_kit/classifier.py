# This file is generated from the notebook, you need to edit it there
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', RandomForestClassifier(
                n_estimators=10, max_leaf_nodes=10, random_state=61))
        ])

    def fit(self, X, y, prev_classifier=None):
        if prev_classifier is not None:
            self.clf = prev_classifier.clf
            rf = self.clf.steps[1][1]
            rf.set_params(n_estimators=2 * rf.n_estimators, warm_start=True)
        self.clf.fit(X, y)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
