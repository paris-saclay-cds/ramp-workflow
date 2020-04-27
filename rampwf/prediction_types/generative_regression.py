import numpy as np
from .base import BasePrediction
from ..utils import MAX_MDN_PARAMS, distributions_dispatcher
import warnings
import itertools


@property
def _valid_indexes(self):
    """Return valid indices (e.g., a cross-validation slice)."""
    return self.__valid_indexes


def _regression_init(self, y_pred=None, y_true=None, n_samples=None):
    self.__valid_indexes = None
    if y_pred is not None:
        self.y_pred = y_pred
    elif y_true is not None:
        self.y_pred = np.array(y_true)
    elif n_samples is not None:
        # for each dim, 1 for the nb of dists (which is max self.max_dists),
        # then nb_dists for weights, nb of dists for types
        # and lastly nb of dists*2 for dist parameters
        shape = (
            n_samples,
            self.n_columns * (1 + (2 + MAX_MDN_PARAMS) * self.max_dists))
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
    curr_indicies = np.zeros(len(index_list)).astype(int)
    dims = []
    idx_curr_dim = 0
    while idx_curr_dim < cls.n_columns:
        combined_size = 0
        curr_weights = []
        curr_types = []
        curr_params = []
        for i in range(len(index_list)):
            curr_pred = predictions_list[index_list[i]].y_pred
            dim_sizes = curr_pred[:, curr_indicies[i]]
            selected = np.isfinite(dim_sizes)
            curr_sizes = dim_sizes[selected]
            if curr_sizes.size != 0:
                curr_size = int(curr_sizes[0])
            else:
                raise ValueError("One or more dimensions missing")

            combined_size += curr_size

            temp_weights = curr_pred[:, curr_indicies[i] + 1:
                                     curr_indicies[i] + 1 + curr_size]

            temp_weights[~selected] = 0

            curr_weights.append(
                temp_weights)

            temp_types = curr_pred[:, curr_indicies[i] + 1 + curr_size:
                                      curr_indicies[i] + 1 + 2 * curr_size]

            temp_types[~selected] = -1

            active_types = temp_types[selected][0]

            curr_types.append(
                temp_types)

            curr_indicies[i] += 1 + curr_size * 2

            end_single_genreg = curr_indicies[i]
            # Recover the right number of params for each distribution
            for k in range(curr_size):
                active_type = int(active_types[k])
                dist = distributions_dispatcher(active_type)
                end_single_genreg += dist.n_params

            curr_params.append(
                curr_pred[:, curr_indicies[i]:end_single_genreg])

            curr_indicies[i] = end_single_genreg

        weights = np.concatenate(curr_weights, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            weights /= weights.sum(axis=1)[:, None]
        types = np.concatenate(curr_types, axis=1)
        params = np.concatenate(curr_params, axis=1)
        sizes = np.full((params.shape[0], 1), combined_size).astype(float)

        curr_dim = np.concatenate((sizes, weights, types, params), axis=1)
        dims.append(curr_dim)
        idx_curr_dim += 1

    combined_predictions = cls(y_pred=np.concatenate(dims, axis=1))

    set_valid_indicies = []
    for i in range(len(predictions_list)):
        set_valid_indicies.append(~np.isnan(predictions_list[i].y_pred[:, 0]))

    combined_predictions.__valid_indexes = \
        np.logical_or.reduce(set_valid_indicies)

    return combined_predictions


def set_valid_in_train(self, predictions, test_is):
    """Set a cross-validation slice."""

    # Blending can create arbitrary-length mixtures. Sometimes
    # we need to extend the prediction array to accomodate this.
    if predictions.y_pred.shape[1] > self.y_pred.shape[1]:
        shape = (self.y_pred.shape[0], predictions.y_pred.shape[1])
        y_pred = np.empty(shape, dtype=float)
        y_pred.fill(np.nan)
        y_pred[:, :self.y_pred.shape[1]] = self.y_pred
        self.y_pred = y_pred
    self.y_pred[test_is, :predictions.y_pred.shape[1]] = predictions.y_pred


def make_generative_regression(max_dists, label_names=[]):
    Predictions = type(
        'GenerativeRegressionGaussian',
        (BasePrediction,),
        {'label_names'       : label_names,
         'max_dists'         : max_dists,
         'n_columns'         : len(label_names),
         'n_columns_true'    : len(label_names),
         '__init__'          : _regression_init,
         'combine'           : _combine,
         'valid_indexes'     : _valid_indexes,
         'set_valid_in_train': set_valid_in_train
         })
    return Predictions
