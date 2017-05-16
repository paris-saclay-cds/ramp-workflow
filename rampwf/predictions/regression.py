"""Regression predictions.

y_pred can be one or two-dimensional (for multi-target regression)
"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import numpy as np
from .base import BasePrediction


class Predictions(BasePrediction):

    def __init__(self, labels=None, y_pred=None, y_true=None, shape=None):
        self.labels = labels
        if y_pred is not None:
            self.y_pred = np.array(y_pred)
        elif y_true is not None:
            self.y_pred = np.array(y_true)
        elif shape is not None:
            self.y_pred = np.empty(shape, dtype=float)
            self.y_pred.fill(np.nan)
        else:
            raise ValueError("Missing init argument: y_pred, y_true, f_name "
                             "or n_samples")
