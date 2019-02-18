from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

# RAMP START HYPERPARAMETERS
hyper_parameters = {'logreg_C': 1,
                    'imputer_strategy': 'median'}
# RAMP END HYPERPARAMETERS


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline([
            ('imputer',
             Imputer(strategy=hyper_parameters['imputer_strategy'])),
            ('classifier', LogisticRegression(C=hyper_parameters['logreg_C']))
        ])

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
