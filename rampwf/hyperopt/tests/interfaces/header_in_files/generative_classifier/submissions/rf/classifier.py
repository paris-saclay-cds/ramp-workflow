from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from rampwf.hyperopt import Hyperparameter

# RAMP START HYPERPARAMETERS
max_leaf_nodes = Hyperparameter(
    dtype='int', default=8, values=[2, 4, 8, 16])
# RAMP END HYPERPARAMETERS


class Classifier(BaseEstimator):
    def __init__(self, dtypes, data_dict, cols):
        pass


    def fit(self, X, y):
        self.clf = RandomForestClassifier(
            max_leaf_nodes=int(max_leaf_nodes),
            random_state=61)
        self.clf.fit(X, y)

    def predict(self, X, y):
        return self.clf.predict(X)

    def predict_proba(self, X, y):
        return self.clf.predict_proba(X)
