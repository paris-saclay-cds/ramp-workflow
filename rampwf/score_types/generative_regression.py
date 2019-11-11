import numpy as np
from scipy.stats import norm
from .base import BaseScoreType
from ..utils import distributions_dispatcher


class NegativeLogLikelihoodReg(BaseScoreType):
    is_lower_the_better = True
    minimum = -float('inf')  # This is due to the fact that bins are possibly infinitesimally small
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
        prob = y_pred[:, :, self.n_bins + 1: 2 * self.n_bins + 1].swapaxes(1, 0)

        if np.isnan(bins).any() or np.isnan(prob).any():
            raise ValueError(
                """The output of the regressor contains nans, or isof the wrong shape. 
                It should be (time_step, dim_step, bins+probas)""")

        summed_prob = np.sum(prob, axis=2, keepdims=True)
        if not np.all(summed_prob == 1):
            prob = (prob / summed_prob)

        # If one of the bins is not ordered
        if any([time_step[i] >= time_step[i + 1] for dim_bin in bins for time_step in dim_bin for i in
                range(len(time_step) - 1)]):
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

            # If the value is outside of the probability distribution we discretized, it is 0,
            # and the scaling uses the size of the largest bin
            preds[classes == -1] = 0
            bins[classes == -1] = -1

            preds_matrix.append(preds)
            selected_bins.append(bins)

        preds_matrix = np.array(preds_matrix)
        selected_bins = np.array(selected_bins)

        lk = np.log(preds_matrix / selected_bins)
        lk = np.clip(lk, WORST_LK, None, out=lk)
        # To avoid infinitely bad loss for a single y outside of the bins,
        # we bound the likelihood by example, to be no smaller than WORST_LK
        return np.sum(-lk) / lk.size


class LikelihoodRatio(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, n_bins, name='ll_ratio', precision=2):
        self.name = name
        self.precision = precision
        self.n_bins = n_bins

    def __call__(self, y_true, y_pred):
        nll_reg_score = NegativeLogLikelihoodReg(self.n_bins)
        nll_reg = nll_reg_score(y_true, y_pred)

        if len(y_true.shape) == 1:
            y_true = np.array([y_true])
        else:
            y_true = y_true.swapaxes(0, 1)

        means = np.mean(y_true, axis=1)
        stds = np.std(y_true, axis=1)
        baseline_lls = np.array([
            norm.logpdf(y, loc=mean, scale=std)
            for y, mean, std in zip(y_true, means, stds)])

        return np.exp(-nll_reg - np.sum(
            baseline_lls) / baseline_lls.size)


class NegativeLogLikelihoodRegDists(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='logLKGauss', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):

        if len(y_true.shape) == 1:
            y_true = np.array([y_true])
        else:
            y_true = y_true.swapaxes(0, 1)

        logLK = 0

        curr_idx = 0

        for y_true_dim in y_true:

            nb_dists = int(y_pred[0, curr_idx])
            curr_idx += 1
            id_params_start = curr_idx + nb_dists * 2
            weights = y_pred[:, curr_idx:curr_idx + nb_dists]
            types = y_pred[:, curr_idx + nb_dists:id_params_start]

            assert np.allclose(weights.sum(axis=1), 1.0), \
                "The weight should sum up to 1"

            curr_idx = id_params_start
            weighted_probs = np.zeros(len(y_true_dim))
            for i in range(nb_dists):
                empy_dist_id = distributions_dispatcher().id
                mask = ~np.array(types[:, i] == empy_dist_id)
                currtype = int(types[:, i][mask][0])
                dist = distributions_dispatcher(currtype)
                end_params = curr_idx + dist.nb_params
                probs = dist.pdf(y_true_dim[mask],
                                 y_pred[:, curr_idx:end_params][mask])
                curr_idx = end_params
                weighted_probs[mask] += weights[:, i][mask] * probs
            partial_lk = np.log(weighted_probs)
            logLK += np.sum(-partial_lk)

        return logLK / y_true.size


class LikelihoodRatioDists(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='ll_ratio', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        nll_reg_score = NegativeLogLikelihoodRegDists()
        nll_reg = nll_reg_score(y_true, y_pred)

        if len(y_true.shape) == 1:
            y_true = np.array([y_true])
        else:
            y_true = y_true.swapaxes(0, 1)

        means = np.mean(y_true, axis=1)
        stds = np.std(y_true, axis=1)
        baseline_lls = np.array([
            norm.logpdf(y, loc=mean, scale=std)
            for y, mean, std in zip(y_true, means, stds)])

        return np.exp(-nll_reg - np.sum(
            baseline_lls) / baseline_lls.size)
