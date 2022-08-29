from sklearn.base import BaseEstimator
from sklearn.svm import SVC


class Classifier(BaseEstimator):
    def __init__(self, dtypes, data_dict, cols):

        self.dtypes = dtypes


    def fit(self, X, y):
        self.clf = SVC(random_state=61, probability=True)
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X, y):
        return self.clf.predict_proba(X)
