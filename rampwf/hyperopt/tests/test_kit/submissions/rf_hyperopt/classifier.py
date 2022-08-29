from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from rampwf.hyperopt import Hyperparameter

# RAMP START HYPERPARAMETERS
max_features = Hyperparameter(
    dtype='object', default='auto', values=['auto', 'sqrt', 'log2'])
# RAMP END HYPERPARAMETERS


class Classifier(BaseEstimator):
    def __init__(self, dtypes, data_dict, cols):
        pass


    def fit(self, X, y):
        self.clf = RandomForestClassifier(
            max_features=str(max_features), random_state=61)
        self.clf.fit(X, y)

    def predict(self, X, y):
        return self.clf.predict(X)

    def predict_proba(self, X, y):
        return self.clf.predict_proba(X)
