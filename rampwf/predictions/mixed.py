"""Mixed classification/regression prediction predictions.

The first shape[1] - 1 columns are classification posteriors, the
last column is a regression.

Alternatively, we would need to define workflows with multiple outputs,
and let scores handle them separately.

"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import numpy as np
from .base import BasePrediction
from . import multiclass
from . import regression


class Predictions(BasePrediction):
    def __init__(self, labels=None, y_pred=None, y_true=None, shape=None):
        self.labels = labels
        # multiclass.labels = labels
        if y_pred is not None:
            self.multiclass = multiclass.Predictions(
                labels=self.labels, y_pred=y_pred[:, :-1])
            self.regression = regression.Predictions(
                labels=self.labels, y_pred=y_pred[:, -1])
        elif y_true is not None:
            self.multiclass = multiclass.Predictions(
                labels=self.labels, y_true=y_true[:, 0])
            self.regression = regression.Predictions(
                labels=self.labels, y_true=y_true[:, 1])
        elif shape is not None:
            # last col is reg, first shape[1] - 1 cols are clf
            self.multiclass = multiclass.Predictions(
                labels=self.labels, shape=(shape[0], shape[1] - 1))
            self.regression = regression.Predictions(
                labels=self.labels, shape=shape[0])

    def set_valid_in_train(self, predictions, test_is):
        self.multiclass.set_valid_in_train(predictions.multiclass, test_is)
        self.regression.set_valid_in_train(predictions.regression, test_is)

    @property
    def valid_indexes(self):
        return self.multiclass.valid_indexes

    @property
    def y_pred(self):
        return np.concatenate(
            [self.multiclass.y_pred, self.regression.y_pred.reshape(-1, 1)],
            axis=1)
