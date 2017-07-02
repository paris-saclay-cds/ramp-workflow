"""Regression predictions.

y_pred can be one or two-dimensional (for multi-target regression)
"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import numpy as np
from .base import BasePrediction


def _regression_init(self, y_pred=None, y_true=None, n_samples=None):
    if y_pred is not None:
        self.y_pred = np.array(y_pred)
    elif y_true is not None:
        self.y_pred = np.array(y_true)
    elif n_samples is not None:
        n_columns = len(self.label_names)
        if n_columns == 0:
            shape = (n_samples)
        else:
            shape = (n_samples, n_columns)
        self.y_pred = np.empty(shape, dtype=float)
        self.y_pred.fill(np.nan)
    else:
        raise ValueError(
            'Missing init argument: y_pred, y_true, or n_samples')


def make_predictions_type(label_names=[]):
    Predictions = type(
        'Predictions',
        (BasePrediction,),
        {'label_names': label_names,
         '__init__': _regression_init, })
    return Predictions
