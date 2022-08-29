from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from rampwf.hyperopt import Hyperparameter

# RAMP START HYPERPARAMETERS
max_leaf_nodes = Hyperparameter(
    dtype='int', default=8, values=[2, 4, 8, 16])
N_estimators = Hyperparameter(
    dtype='int', default=100, values=[50, 100,1000, 5000])
max_depth = Hyperparameter(
    dtype='int', default=None, values=[10, 20, 30, 40, 100])
min_samples_split = Hyperparameter(
    dtype='int', default=10, values=[2, 5, 10, 15, 20])
min_samples_leaf = Hyperparameter(
    dtype='int', default=10, values=[2, 5, 10, 15])
# RAMP END HYPERPARAMETERS


class Classifier(BaseEstimator):
    def __init__(self, dtypes, data_dict, cols):
        pass


    def fit(self, X, y):
        self.clf = RandomForestClassifier(
            n_estimators=int(N_estimators), max_leaf_nodes=int(max_leaf_nodes),
            max_depth=int(max_depth), min_samples_split = int(min_samples_split),
            min_samples_leaf= int(min_samples_leaf), random_state=61)
        self.clf.fit(X, y)

    def predict(self, X, y):
        return self.clf.predict(X)

    def predict_proba(self, X, y):
        return self.clf.predict_proba(X)
