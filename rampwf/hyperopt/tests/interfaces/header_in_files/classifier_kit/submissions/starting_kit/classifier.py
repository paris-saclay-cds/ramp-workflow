from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from rampwf.hyperopt import Hyperparameter

# RAMP START HYPERPARAMETERS
vals = [0.01, 0.1, 0.9, 1.0]
logreg_C = Hyperparameter(dtype="float", default=1.0, values=vals)
imputer_strategy = Hyperparameter(
    dtype="object", default="median", values=["mean", "median"]
)
# RAMP END HYPERPARAMETERS


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline(
            [
                ("imputer", SimpleImputer(strategy=str(imputer_strategy))),
                ("classifier", LogisticRegression(C=float(logreg_C))),
            ]
        )

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
