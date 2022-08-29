from sklearn.base import BaseEstimator

from sklearn.model_selection import cross_val_score

# from pytorch_tabular import TabularModel
# from pytorch_tabular.models import CategoryEmbeddingModelConfig, NodeConfig, TabNetModelConfig
# from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

import numpy as np
import pandas as pd

# # RAMP START HYPERPARAMETERS
# max_features = Hyperparameter(
#     dtype='float', default = 0.5, values=[0.2, 0.3, 0.5, 0.8, 1])
# n_estimators = Hyperparameter(
#     dtype='int', default=100, values=[2,5,10, 20, 30, 50, 100,1000])
# max_depth = Hyperparameter(
#     dtype='int', default=10, values=[2,5,10, 20, 30, 50,100,100])
# min_samples_split = Hyperparameter(
#     dtype='int', default=10, values=[5, 10, 15, 20])
# # RAMP END HYPERPARAMETERS


class Classifier(BaseEstimator):
    def __init__(self, types, data_dict, cols):
        self.clf = TabNetClassifier()  # TabNetRegressor()

        # preds = clf.predict(X_test)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        # preds = clf.predict(X_test)
        # return self.tabular_model.predict(X)
        pass

    def predict_proba(self, X, y):

        return self.clf.predict_proba(X)

