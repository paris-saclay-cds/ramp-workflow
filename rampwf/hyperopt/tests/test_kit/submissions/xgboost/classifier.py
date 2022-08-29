from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
from rampwf.hyperopt import Hyperparameter

# RAMP START HYPERPARAMETERS
max_features = Hyperparameter(
    dtype='float', default = 0.5, values=[0.2, 0.3, 0.5, 0.8, 1])
n_estimators = Hyperparameter(
    dtype='int', default=100, values=[2,5,10, 20, 30, 50, 100,1000])
max_depth = Hyperparameter(
    dtype='int', default=10, values=[2,5,10, 20, 30, 50,100,100])
min_samples_split = Hyperparameter(
    dtype='int', default=10, values=[5, 10, 15, 20])
# RAMP END HYPERPARAMETERS

class Classifier(BaseEstimator):
    def __init__(self, dtypes, data_dict, cols):
        pass


    def fit(self, X, y):
        self.clf = XGBClassifier()
        self.clf.fit(X, y)
        print(self.clf.get_params())

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X, y):
        return self.clf.predict_proba(X)

