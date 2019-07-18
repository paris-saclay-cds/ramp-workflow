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


EPSILON = 10e-6


class NegativeLogLikelihoodReg(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, n_bins, name='logLK', precision=2):
        self.name = name
        self.precision = precision
        self.n_bins = n_bins

    def __call__(self, y_true, y_pred):

        if len(y_true.shape) == 1:
            y_true = np.array([y_true])
        else:
            y_true = y_true.swapaxes(0, 1)

        bins = y_pred[:, :, :self.n_bins + 1].swapaxes(1, 0)
        prob = y_pred[:, :, self.n_bins + 1:].swapaxes(1, 0)

        summed_prob = np.sum(prob, axis=2, keepdims=True)
        if not np.all(summed_prob == 1):
            prob = (prob / summed_prob)

        # If one of the bins is not ordered
        if any([time_step[i]>=time_step[i+1] for dim_bin in bins for time_step in dim_bin for i in range(len(time_step)-1)]):
            raise ValueError("Bins must be ordered and non empty")


        classes_matrix = np.full(y_true.shape, np.nan)
        for i, dim_bin in enumerate(bins):
            for j, dim in enumerate(dim_bin):
                truth = y_true[i, j]
                if dim[0] > truth:
                    classes_matrix[i, j] = -1
                else:
                    for k in range(1, len(dim)):
                        if dim[k] > truth:
                            classes_matrix[i, j] = int(k - 1)
                            break
                        elif k == len(dim) - 1:
                            classes_matrix[i, j] = -1
        classes_matrix = classes_matrix.astype('int')

        prob = np.add(prob, EPSILON, casting="unsafe")

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

            # If the value is outside of the probability distribution we discretized, it's probability is epsilon small,
            # and the scaling uses the size of the largest bin
            preds[classes == -1] = EPSILON
            bins[classes == -1] = np.max(bin_dim)

            preds_matrix.append(preds)
            selected_bins.append(bins)

        preds_matrix = np.array(preds_matrix)
        selected_bins = np.array(selected_bins)

        lk = np.log(preds_matrix / selected_bins)
        return np.sum(-lk) / lk.size
