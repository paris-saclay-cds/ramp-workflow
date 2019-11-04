import numpy as np
from .base import BasePrediction
import itertools


def _valid_indexes(self):
    """Return valid indices (e.g., a cross-validation slice)."""
    if len(self.y_pred.shape) == 1:
        return ~np.isnan(self.y_pred)
    elif len(self.y_pred.shape) == 2:
        return ~np.isnan(self.y_pred[:, 0])
    elif len(self.y_pred.shape) == 3:
        return ~np.isnan(self.y_pred[:, 0, 0])
    else:
        raise ValueError('y_pred.shape > 3 is not implemented')


def _regression_init(self, y_pred=None, y_true=None, n_samples=None):
    if y_pred is not None:
        self.y_pred = y_pred
    elif y_true is not None:
        self.y_pred = np.array(y_true)
    elif n_samples is not None:
        # for each dim, 1 for the nb of dists (which is max self.max_dists),
        # then nb_dists for weights, nb of dists for types
        # and lastly nb of dists*2 for dist parameters
        shape = (n_samples, self.n_columns * (1 + 4 * self.max_dists))
        self.y_pred = np.empty(shape, dtype=float)
        self.y_pred.fill(np.nan)
    else:
        raise ValueError(
            'Missing init argument: y_pred, y_true, or n_samples')


# TODO rewrite the combine
@classmethod
def _combine(cls, predictions_list, index_list=None):
    raise NotImplementedError("combine not implemented yet")

def make_generative_regression(max_dists, label_names=[]):
    Predictions = type(
        'GenerativeRegressionGaussian',
        (BasePrediction,),
        {'label_names'   : label_names,
         'max_dists'     : max_dists,
         'n_columns'     : len(label_names),
         'n_columns_true': len(label_names),
         '__init__'      : _regression_init,
         'combine'       : _combine,
         'valid_indexes' : _valid_indexes,
         'set_valid_in_train' : _combine
         })
    return Predictions
