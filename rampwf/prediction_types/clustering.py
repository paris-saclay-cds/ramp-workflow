"""Clustering predictions.

Predictions have two columns: event_id and cluster_id. event_id
comes from the first column of X when test is called. See
``rampwf.workflows.clustering``.

The predictions are used in "supervised" clustering where the data
consists of a set of events ad each event has an unknown number of
clusters. At training time we know the classes (clusters) of each event.
At test time we only know which instances belong to the same event. Events
share statistical properties which can be exploited.

From another point of view, the setup is transfer learning: each event is
a new task, and classed in the task share statistical properties.
"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import numpy as np
from .base import BasePrediction


def _clustering_init(self, y_pred=None, y_true=None, n_samples=None,
                     fold_is=None):
    if y_pred is not None:
        if fold_is is not None:
            y_pred = y_pred[fold_is]
        self.y_pred = np.array(y_pred)
    elif y_true is not None:
        if fold_is is not None:
            y_true = y_true[fold_is]
        self.y_pred = np.array(y_true)
    elif n_samples is not None:
        self.y_pred = np.empty((n_samples, 2), dtype=float)
        self.y_pred.fill(np.nan)
    else:
        raise ValueError(
            'Missing init argument: y_pred, y_true, or n_samples')
    self.check_y_pred_dimensions()


@property
def _valid_indexes(self):
    """Return valid indices (e.g., a cross-validation slice)."""
    return ~np.isnan(self.y_pred[:, 1])


def make_clustering():
    Predictions = type(
        'Clustering',
        (BasePrediction,),
        {'n_columns': 2,
         'n_columns_true': 2,
         '__init__': _clustering_init,
         'valid_indexes': _valid_indexes,
         })
    return Predictions
