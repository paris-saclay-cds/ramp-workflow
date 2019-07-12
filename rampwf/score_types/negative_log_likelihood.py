import numpy as np
from sklearn.metrics import log_loss

from .base import BaseScoreType


class NegativeLogLikelihood(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='negative log likelihood', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_proba, y_proba):
        score = log_loss(y_true_proba, y_proba)
        return score


class logLKGenerative(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, nb_bins, name='logLK', precision=2):
        self.name = name
        self.precision = precision
        self.nb_bins = nb_bins

    def __call__(self, y_true, y_pred):

        if len(y_true.shape) == 1:
            y_true = np.array([y_true])
        else:
            y_true = y_true.swapaxes(0, 1)

        bins = y_pred[:, :, :self.nb_bins + 1].swapaxes(1, 0)
        prob = y_pred[:, :, self.nb_bins + 1:].swapaxes(1, 0)
        classes_matrix = np.full(y_true.shape, np.nan)
        for i, dim_bin in enumerate(bins):
            for j, dim in enumerate(dim_bin):
                truth = y_true[i, j]
                for k in range(len(dim)):
                    if dim[k] > truth:
                        classes_matrix[i, j] = int(k - 1)
                        break
                    elif k == len(dim) - 1:
                        classes_matrix[i, j] = int(k - 1)
        classes_matrix = classes_matrix.astype('int')
        prob += 10e-6  # to avoid instability with log

        bins_sliding = np.full(prob.shape, np.nan)
        for i, dim_n in enumerate(bins):
            for j, dim_t in enumerate(dim_n):
                for k in range(len(dim_t) - 1):
                    bins_sliding[i, j, k] = dim_t[k + 1] - dim_t[k]

        # Multi target regression
        preds_matrix = []
        selected_bins = []
        for classes, prob_dim, bin_dim in zip(classes_matrix, prob, bins_sliding):
            preds = prob_dim[range(len(classes)), classes]
            bins = bin_dim[range(len(classes)), classes]
            preds_matrix.append(preds)
            selected_bins.append(bins)

        preds_matrix = np.array(preds_matrix)
        selected_bins = np.array(selected_bins)

        lk = np.log(preds_matrix / selected_bins)
        return np.sum(-lk) / lk.size
