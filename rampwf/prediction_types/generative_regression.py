import numpy as np
from .base import BasePrediction


def _regression_init(self, y_pred=None, y_true=None, n_samples=None):
    if y_pred is not None:
        self.y_pred = y_pred
    elif y_true is not None:
        self.y_pred = np.array(y_true)
    elif n_samples is not None:
        if self.n_columns == 0:
            shape = (n_samples, 1, 2 * self.n_bins + 1)
        else:
            shape = (n_samples, self.n_columns, 2 * self.n_bins + 1)
        self.y_pred = np.empty(shape, dtype=float)
        self.y_pred.fill(np.nan)
    else:
        raise ValueError(
            'Missing init argument: y_pred, y_true, or n_samples')


def make_generative_regression(n_bins, label_names=[]):
    Predictions = type(
        'GenerativeRegression',
        (BasePrediction,),
        {'label_names'   : label_names,
         'n_bins'        : n_bins,
         'n_columns'     : len(label_names),
         'n_columns_true': len(label_names),
         '__init__'      : _regression_init,
         })
    return Predictions
