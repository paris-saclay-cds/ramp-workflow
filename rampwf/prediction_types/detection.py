"""Multiclass predictions.

``y_pred`` should be two dimensional (n_samples x n_classes).

"""

import numpy as np
from .base import BasePrediction


def _detection_init(self, y_pred=None, y_true=None, n_samples=None):
    if y_pred is not None:
        self.y_pred = y_pred
    elif y_true is not None:
        self.y_pred = y_true
    elif n_samples is not None:
        if self.n_columns == 0:
            shape = (n_samples)
        else:
            shape = (n_samples, self.n_columns)
        self.y_pred = np.empty(shape, dtype=float)
        self.y_pred.fill(np.nan)
    else:
        raise ValueError(
            'Missing init argument: y_pred, y_true, or n_samples')
    self.check_y_pred_dimensions()


def _check_y_pred_dimensions(self):
    pass


@classmethod
def _combine(cls, predictions_list, index_list=None):
    if len(predictions_list) == 0:
        return []
    else:
        return predictions_list[0]


def make_detection(label_names=[]):
    Predictions = type(
        'Predictions',
        (BasePrediction,),
        {'__init__': _detection_init,
         'check_y_pred_dimensions': _check_y_pred_dimensions,
         'combine': _combine,
         })
    return Predictions
