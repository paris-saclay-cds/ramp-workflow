from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from rampwf.hyperopt import Hyperparameter

# test with only one hyperparameter
# RAMP START HYPERPARAMETERS
max_depth = Hyperparameter(
    dtype="int",
    default=5,
    values=[1, 2, 3, 4, 5, 6, 7, 8, 20, 30, 50, 70, 100, 500, 1000],
)
# RAMP END HYPERPARAMETERS


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(max_depth=int(max_depth))),
            ]
        )

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
