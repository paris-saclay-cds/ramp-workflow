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


def get_components(curr_idx, y_pred):
    n_dists = int(y_pred[0, curr_idx])
    curr_idx += 1
    id_params_start = curr_idx + n_dists * 2
    weights = y_pred[:, curr_idx:curr_idx + n_dists]
    assert (weights >= 0).all(), "Weights should all be positive."
    weights /= weights.sum(axis=1)[:, np.newaxis]
    types = y_pred[:, curr_idx + n_dists:id_params_start]
    curr_idx = id_params_start
    dists = []
    paramss = []
    for i in range(n_dists):
        empty_dist_id = distributions_dispatcher().id
        non_empty_mask = ~np.array(types[:, i] == empty_dist_id)
        currtype = int(types[:, i][non_empty_mask][0])
        # TODO: raise exception if type is not consistent
        dists.append(distributions_dispatcher(currtype))
        end_params = curr_idx + dists[i].n_params
        paramss.append(y_pred[:, curr_idx:end_params])
        curr_idx = end_params

    return curr_idx, n_dists, weights, types, dists, paramss


class NegativeLogLikelihoodRegDists(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='logLKGauss', precision=2, output_dim=None,
                 verbose=False):
        self.name = name
        self.precision = precision
        self.output_dim = output_dim
        self.verbose = verbose
        
    def __call__(self, y_true, y_pred):
        n_instances = len(y_true)
        y_true = convert_y_true(y_true)  # output dimension first
        n_dims = len(y_true)
        log_lk = 0  # negative log likelihood to be returned  
        # negative log likelihoods of each output dimension and instance  
        log_lks = np.zeros(y_true.shape)
        # pointer within the vector representation of mixtures y_pred[i]
        curr_idx = 0
        for j_dim, y_true_dim in enumerate(y_true):
            curr_idx, n_dists, weights, types, dists, paramss =\
                get_components(curr_idx, y_pred)
            weighted_probs = np.zeros(len(y_true_dim))
            for i in range(n_dists):
                empty_dist_id = distributions_dispatcher().id
                non_empty_mask = ~np.array(types[:, i] == empty_dist_id)
                probs = dists[i].pdf(
                    y_true_dim[non_empty_mask],
                    paramss[i][non_empty_mask])
                weighted_probs[non_empty_mask] +=\
                    weights[:, i][non_empty_mask] * probs
            log_lks[j_dim, :] = -np.log(weighted_probs)
            log_lk += np.sum(log_lks[j_dim, :])

        if self.output_dim is None:
            if self.verbose:  
                return log_lk / n_instances / n_dims, log_lks
            else:
                return log_lk / n_instances / n_dims
        else:
            return np.sum(log_lks[self.output_dim, :]) / n_instances


class LikelihoodRatioDists(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='ll_ratio', precision=2, output_dim=None,
                 verbose=False, plot=False):
        self.name = name
        self.precision = precision
        self.output_dim = output_dim
        self.verbose = verbose
        self.plot = plot

    def __call__(self, y_true, y_pred):
        n_instances = len(y_true)
        nll_reg_score = NegativeLogLikelihoodRegDists(
            output_dim=self.output_dim, verbose=self.verbose or self.plot)
        if self.verbose or self.plot:
            nll_reg, log_lks = nll_reg_score(y_true, y_pred)
        else:
            nll_reg = nll_reg_score(y_true, y_pred)
        y_true = convert_y_true(y_true)
        n_dims = len(y_true)
        
        means = np.mean(y_true, axis=1)
        stds = np.std(y_true, axis=1)
        if self.output_dim is None:
            baseline_lls = np.array([
                norm.logpdf(y, loc=mean, scale=std)
                for y, mean, std in zip(y_true, means, stds)])
            n_dims = len(y_true)
        else:
            baseline_lls = norm.logpdf(
                y_true[self.output_dim], loc=means[self.output_dim],
                scale=stds[self.output_dim])
            n_dims = 1

        if self.verbose:
            print(
                np.exp(np.sum(-log_lks, axis=1) / n_instances -
                np.sum(baseline_lls, axis = 1) / n_instances))
        if self.plot:
            ratio_by_point = np.exp(-log_lks - baseline_lls)
            for i, ratio in enumerate(ratio_by_point):
                plt.plot(ratio)
                plt.title(i)
                plt.show()
        return np.exp(-nll_reg - np.sum(baseline_lls) / n_instances / n_dims)


class RMSERegDists(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='RMSE', precision=2, output_dim=None):
        self.name = name
        self.precision = precision
        self.output_dim = output_dim
        
    def __call__(self, y_true, y_pred):
        n_instances = len(y_true)
        y_true = convert_y_true(y_true)  # output dimension first
        n_dims = len(y_true)
        mean_preds = np.zeros(y_true.shape)
        rmse = 0  # rmse to be returned  
        # pointer within the vector representation of mixtures y_pred[i]
        curr_idx = 0
        for j_dim, y_true_dim in enumerate(y_true):
            curr_idx, n_dists, weights, types, dists, paramss =\
                get_components(curr_idx, y_pred)
            for i in range(n_dists):
                empty_dist_id = distributions_dispatcher().id
                non_empty_mask = ~np.array(types[:, i] == empty_dist_id)
                means = dists[i].mean(paramss[i][non_empty_mask])
                mean_preds[j_dim, non_empty_mask] +=\
                    weights[:, i][non_empty_mask] * means

        if self.output_dim is None:
            return np.sqrt(
                ((y_true - mean_preds) ** 2).sum() / n_instances / n_dims)
        else:
            return np.sqrt((
                (y_true[self.output_dim] - mean_preds[self.output_dim])
                ** 2).sum() / n_instances)


class R2RegDists(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='R2', precision=2, output_dim=None):
        self.name = name
        self.precision = precision
        self.output_dim = output_dim
        
    def __call__(self, y_true, y_pred):
        n_instances = len(y_true)
        y_true = convert_y_true(y_true)  # output dimension first
        n_dims = len(y_true)
        stds = np.std(y_true, axis=1)
        mean_preds = np.zeros(y_true.shape)
        rmse = 0  # rmse to be returned  
        # pointer within the vector representation of mixtures y_pred[i]
        curr_idx = 0
        for j_dim, y_true_dim in enumerate(y_true):
            curr_idx, n_dists, weights, types, dists, paramss =\
                get_components(curr_idx, y_pred)
            for i in range(n_dists):
                empty_dist_id = distributions_dispatcher().id
                non_empty_mask = ~np.array(types[:, i] == empty_dist_id)
                means = dists[i].mean(paramss[i][non_empty_mask])
                mean_preds[j_dim, non_empty_mask] +=\
                    weights[:, i][non_empty_mask] * means
        r2s = ((y_true - mean_preds) ** 2).mean(axis=1)
        r2s /= stds ** 2
        r2s = 1 - r2s

        if self.output_dim is None:
            return r2s.mean()
        else:
            return r2s[self.output_dim]


class KSCalibrationRegDists(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='KS', precision=2, output_dim=None, plot=False):
        self.name = name
        self.precision = precision
        self.output_dim = output_dim
        self.plot = plot
        
    def __call__(self, y_true, y_pred):
        n_instances = len(y_true)
        y_true = convert_y_true(y_true)  # output dimension first
        n_dims = len(y_true)
        cdfs = np.zeros(y_true.shape)
        # pointer within the vector representation of mixtures y_pred[i]
        curr_idx = 0
        for j_dim, y_true_dim in enumerate(y_true):
            curr_idx, n_dists, weights, types, dists, paramss =\
                get_components(curr_idx, y_pred)
            for i in range(n_dists):
                empty_dist_id = distributions_dispatcher().id
                non_empty_mask = ~np.array(types[:, i] == empty_dist_id)
                cdfss = dists[i].cdf(
                    y_true_dim[non_empty_mask],
                    paramss[i][non_empty_mask])
                cdfs[j_dim, non_empty_mask] +=\
                    weights[:, i][non_empty_mask] * cdfss
        ks_stats = np.zeros(n_dims)
        for j_dim in range(n_dims):
            ks_stats[j_dim] = np.max(np.abs(
                np.sort(cdfs[j_dim]) - np.arange(n_instances) / n_instances))
 
        if self.plot:
            if self.output_dim is None:
                for j_dim in range(n_dims):
                    plt.hist(cdfs[j_dim])
                    plt.title(j_dim)
                    plt.show()
                    plt.plot(
                        np.sort(cdfs[j_dim]),
                        np.arange(n_instances) / n_instances)
                    plt.plot([0, 1], [0, 1])
                    plt.title(j_dim)
                    plt.show()
            else:
                plt.hist(cdfs[self.output_dim])
                plt.title(self.output_dim)
                plt.show()

        if self.output_dim is None:
            return ks_stats.mean()
        else:
            return ks_stats[self.output_dim]
