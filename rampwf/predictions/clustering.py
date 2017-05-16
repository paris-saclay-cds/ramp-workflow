"""Clustering predictions.

Predictions have two columns: event_id and cluster_id. event_id
comes from the first column of X when test is called. See
``rampwf.workflows.clustering``.

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
            self.y_pred = np.empty((shape[0], 2))
            self.y_pred.fill(np.nan)
        else:
            raise ValueError("Missing init argument: y_pred, y_true, f_name "
                             "or n_samples")

    @property
    def valid_indexes(self):
        return ~np.isnan(self.y_pred[:, 1])
