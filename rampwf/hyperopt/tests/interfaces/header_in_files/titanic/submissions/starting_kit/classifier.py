from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

# RAMP START HYPERPARAMETERS
impute_strategy = 'mean'  # opt: ['median', 'mean']
logistic_C = 0.01  # opt: [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]
# RAMP END HYPERPARAMETERS


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline([
            ('imp', Imputer(
                strategy=impute_strategy, missing_values=-1)),
            ('clf', LogisticRegression(C=logistic_C))
        ])

    def fit(self, X, y):
            self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
