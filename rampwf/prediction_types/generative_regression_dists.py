import numpy as np
from .base import BasePrediction
from ..utils import MAX_PARAMS
import warnings
import itertools

@property
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
        shape = (n_samples, self.n_columns * (1 + (2+MAX_PARAMS) * self.max_dists))
        self.y_pred = np.empty(shape, dtype=float)
        self.y_pred.fill(np.nan)
    else:
        raise ValueError(
            'Missing init argument: y_pred, y_true, or n_samples')


# TODO rewrite the combine
@classmethod
def _combine(cls, predictions_list, index_list=None):
    if index_list is None:
        index_list = range(len(predictions_list))
    curr_indicies = np.zeros(len(predictions_list)).astype(int)
    dims = []

    for idx_curr_dim in range(cls.n_columns):
        combined_size=0
        curr_weights = []
        curr_types = []
        curr_params = []
        for i in index_list:
            curr_pred= predictions_list[i].y_pred
            dim_sizes = curr_pred[:,curr_indicies[i]]
            selected = np.isfinite(dim_sizes)
            curr_size = dim_sizes[selected]
            if curr_size.size != 0:
                curr_size= int(curr_size[0])
            else:
                raise ValueError("One or more dimensions missing")

            combined_size += curr_size

            temp_weights = curr_pred[:,curr_indicies[i]+1:
                            curr_indicies[i]+1+curr_size]

            temp_weights[~selected] =0

            curr_weights.append(
                temp_weights)

            temp_types = curr_pred[:, curr_indicies[i]+1+curr_size:
                             curr_indicies[i]+1+2*curr_size]

            temp_types[~selected] = -1

            curr_types.append(
                temp_types)

            curr_params.append(
                curr_pred[:, curr_indicies[i]+1+2*curr_size:
                             curr_indicies[i]+1+(2+MAX_PARAMS)*curr_size])


            curr_indicies[i] += 1 + curr_size * (2 + MAX_PARAMS)
        weights = np.concatenate(curr_weights, axis = 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            weights /= weights.sum(axis=1)[:,None]
        types = np.concatenate(curr_types, axis = 1)
        params = np.concatenate(curr_params, axis = 1)
        sizes = np.full((params.shape[0], 1), combined_size).astype(float)
        sizes[np.isnan(params[:,0])] = np.nan

        curr_dim = np.concatenate((sizes, weights, types, params), axis=1)
        dims.append(curr_dim)

    combined_predictions = cls(y_pred= np.concatenate(dims, axis=1))
    return combined_predictions


def set_valid_in_train(self, predictions, test_is):
    """Set a cross-validation slice."""
    self.y_pred[test_is,:predictions.y_pred.shape[1]] = predictions.y_pred

def make_generative_regression_dists(max_dists, label_names=[]):
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
         'set_valid_in_train' : set_valid_in_train
         })
    return Predictions
