"""Regression predictions.

y_pred can be one or two-dimensional (for multi-target regression)
"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import numpy as np
from .base import BasePrediction


def _regression_init(self, y_pred=None, y_true=None, n_samples=None,
                     fold_is=None):
    """Initialize a regression prediction type.

    The input is either y_pred, or y_true, or n_samples.

    Parameters
    ----------
    y_pred : a numpy array
        representing the predictions returned by
        problem.workflow.test_submission; 1D (single target regression)
        or 2D (multi-target regression)
    y_true : a numpy array
        representing the ground truth returned by problem.get_train_data
        and problem.get_test_data; 1D (single target regression)
        or 2D (multi-target regression)
    n_samples : int
        to initialize an empty container, for the combined predictions
    fold_is : a list of integers
        either the training indices, validation indices, or None when we
        use the (full) test data.
    """
    if y_pred is not None:
        if fold_is is not None:
            y_pred = y_pred[fold_is]
        self.y_pred = np.array(y_pred)
    elif y_true is not None:
        if fold_is is not None:
            y_true = y_true[fold_is]
        self.y_pred = np.array(y_true)
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


def make_regression(label_names=[]):
    Predictions = type(
        'Regression',
        (BasePrediction,),
        {'label_names': label_names,
         'n_columns': len(label_names),
         'n_columns_true': len(label_names),
         '__init__': _regression_init,
         })
    return Predictions
