from sklearn.base import BaseEstimator
from gen_models.models import ARRF_SIGMA
from gen_models.metrics import get_likelihoods
import numpy as np


class Classifier(BaseEstimator):
    def __init__(self, dtypes, data_dict, cols):
        self.is_autoregressive = True
        self.dtypes = dtypes

    def fit(self, X, y):

        unique, counts = np.unique(y, return_counts=True)
        self.counts = counts/(sum(counts))
        print("XXX", self.counts)

        self.n_classes = len(unique)
        self.flag = [False] * len(unique)
        self.clf = [None] * len(unique)

        for i in range(2):

            locs = np.where(y==i)
            self.clf[i] = ARRF_SIGMA(dtypes=self.dtypes).fit(None, X[locs]), self.counts[i]



    def predict(self, X, y, i):

        return self.clf[i][0].predict(None, X), self.clf[i][1]

    def predict_proba(self, X, y):
        self.proba = np.zeros((len(X),self.n_classes))
        for i in range(2):
            preds, probas = self.predict(X, y, i)
            a, _ = get_likelihoods(X, preds)

            self.proba[:,i] = np.exp(-np.sum(a,axis=0)) * probas

        return self.proba

