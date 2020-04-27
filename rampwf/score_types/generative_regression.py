"""Score types (metrics) for generative regression.

`y_true` is either n x d dimensional where n is the number of data points and 
d is the number of (output) dimensions, or n dimensional if d=1. `y_pred` is
nxD dimensional where
D = sum_j(1 + 2 * n_dists_j + sum_ell(n_params_{j, ell})). The
following scheme was designed to be able to store the full y_pred as a
numerical numpy array.
 - y_pred[i] contains d variable-size blocks, one for each output dimension
   j = 1, ..., d.
 - The first element of the block is n_dists_j, the number of mixture
   components in dimension j, decided by the submitters. Its maximum is
   determined by the organizer in problem.py.
 - This followed by n_dists_j integers representing mixture component types
   (see utils.generative_regression). A submitter can mix different types of
   components but the type sequence must be the same for all instances (to
   make y_pred[i] having the same length for all i). The only exception is
   that we can have instances with EmptyDist (id=-1) replacing any type.
 - This followed by n_dists_j floats, the component weights in dimension j.
   They all must be nonnegative and they have to add up to one.
 - This followed by a variable length block of n_dists_j mixture component
   parameters. The number of component parameters depend on the component
   types (see utils.generative_regression).
"""

# Authors: Gabriel Hurtado <>,
#          Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from .base import BaseScoreType
from ..utils import distributions_dispatcher


def convert_y_true(y_true):
    """Convert y_pred into output dimensions first."""
    if len(y_true.shape) == 1:
        return np.array([y_true])
    else:
        return y_true.swapaxes(0, 1)


class NegativeLogLikelihoodReg(BaseScoreType):
    is_lower_the_better = True
    # This is due to the fact that bins are possibly infinitesimally small
    minimum = -float('inf')
    maximum = float('inf')

    def __init__(self, n_bins, name='logLK', precision=2):
        self.name = name
        self.precision = precision
        self.n_bins = n_bins

    def __call__(self, y_true, y_pred):
        y_true = convert_y_true(y_true)
        bins = y_pred[:, :, :self.n_bins + 1].swapaxes(1, 0)
        prob = y_pred[:, :, self.n_bins + 1: 2 * self.n_bins + 1].swapaxes(
            1, 0)

        if np.isnan(bins).any() or np.isnan(prob).any():
            raise ValueError(
                """The output of the regressor contains nans, or isof the
                wrong shape. It should be (time_step, dim_step, bins+probas)
                """)

        summed_prob = np.sum(prob, axis=2, keepdims=True)
        if not np.all(summed_prob == 1):
            prob = (prob / summed_prob)

        # If one of the bins is not ordered
        if any([time_step[i] >= time_step[i + 1] for dim_bin in bins\
                for time_step in dim_bin for i in range(len(time_step) - 1)]):
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
        for classes, prob_dim, bin_dim in\
                zip(classes_matrix, prob, bins_sliding):
            preds = prob_dim[range(len(classes)), classes]
            bins = bin_dim[range(len(classes)), classes]

            # If the value is outside of the probability distribution we
            # discretized, it is 0,
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
        y_true = convert_y_true(y_true)
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

    def __init__(self, name='logLKGauss', precision=2, verbose=False):
        self.name = name
        self.precision = precision
        self.verbose = verbose
        
    def __call__(self, y_true, y_pred):
        y_true = convert_y_true(y_true)  # output dimension first
        log_lk = 0  # negative log likelihood to be returned  
        # negative log likelihoods of each output dimension and instance  
        log_lks = np.zeros(y_true.shape)
        # pointer within the vector representation of mixtures y_pred[i]
        curr_idx = 0
        for j_dim, y_true_dim in enumerate(y_true):
            # number of actual distributions
            n_dists = int(y_pred[0, curr_idx])
            curr_idx += 1
            id_params_start = curr_idx + n_dists * 2
            weights = y_pred[:, curr_idx:curr_idx + n_dists]
            types = y_pred[:, curr_idx + n_dists:id_params_start]
            sum_weights = weights.sum(axis=1)
            assert np.allclose(sum_weights, 1.0), \
                "The weights should sum up to 1, not {}.".format(sum_weights)

            curr_idx = id_params_start
            weighted_probs = np.zeros(len(y_true_dim))
            for i in range(n_dists):
                empty_dist_id = distributions_dispatcher().id
                non_empty_mask = ~np.array(types[:, i] == empty_dist_id)
                currtype = int(types[:, i][non_empty_mask][0])
                # TODO: raise exception if type is not consistent
                dist = distributions_dispatcher(currtype)
                end_params = curr_idx + dist.n_params
                probs = dist.pdf(
                    y_true_dim[non_empty_mask],
                    y_pred[:, curr_idx:end_params][non_empty_mask])
                curr_idx = end_params
                weighted_probs[non_empty_mask] +=\
                    weights[:, i][non_empty_mask] * probs
            log_lks[j_dim, :] = -np.log(weighted_probs)
            log_lk += np.sum(log_lks[j_dim, :])

        if self.verbose:  
            return log_lk / y_true.size, log_lks
        else:
            return log_lk / y_true.size


class LikelihoodRatioDists(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='ll_ratio', precision=2, verbose=False,
                 plot=False):
        self.name = name
        self.precision = precision
        self.verbose = verbose
        self.plot = plot

    def __call__(self, y_true, y_pred):

        nll_reg_score = NegativeLogLikelihoodRegDists(verbose=self.verbose)
        if self.verbose:
            nll_reg, lk_by_i = nll_reg_score(y_true, y_pred)
        else:
            nll_reg = nll_reg_score(y_true, y_pred)
        y_true = convert_y_true(y_true)

        means = np.mean(y_true, axis=1)
        stds = np.std(y_true, axis=1)
        baseline_lls = np.array([
            norm.logpdf(y, loc=mean, scale=std)
            for y, mean, std in zip(y_true, means, stds)])
        
        if self.verbose:
            print(
                np.exp(np.sum(-lk_by_i, axis=1) / baseline_lls.size -
                np.sum(baseline_lls, axis = 1) / baseline_lls.size))
            if self.plot:
                ratio_by_point = np.exp(-lk_by_i - baseline_lls)
                for i, ratio in enumerate(ratio_by_point):
                    plt.plot(ratio)
                    plt.title(i)
                    plt.show()

        return np.exp(-nll_reg - np.sum(baseline_lls) / baseline_lls.size)
