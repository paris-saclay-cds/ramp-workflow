"""Score types (metrics) for generative regression.

`y_true` is either n x d dimensional where n is the number of data points and
d is the number of (output) dimensions, or n dimensional if d=1. `y_pred` is
nxD dimensional where
D = sum_j(1 + 2 * n_components_j + sum_ell(n_params_{j, ell})). The
following scheme was designed to be able to store the full y_pred as a
numerical numpy array.
 - y_pred[i] contains d variable-size blocks, one for each output dimension
   j = 1, ..., d.
 - The first element of the block is n_components_j, the number of mixture
   components in dimension j, decided by the submitters. Its maximum is
   determined by the organizer in problem.py.
 - This is followed by n_components_j integers representing mixture component
   types (see utils.generative_regression). A submitter can mix different types
   of components but the type sequence must be the same for all instances (to
   make y_pred[i] having the same length for all i). The only exception is
   that we can have instances with EmptyDist (id=-1) replacing any type.
 - This is followed by n_components_j floats, the component weights in
   dimension j. They all must be nonnegative and they have to add up to one.
 - This is followed by a variable length block of n_components_j mixture
   component parameters. The number of component parameters depend on the
   component types (see utils.generative_regression).
"""

# Authors: Gabriel Hurtado <gabriel.j.hurtado@gmail.com>,
#          Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import numpy as np
from scipy.stats import norm

from .base import BaseScoreType
from ..utils import get_components, EMPTY_DIST


def convert_y_true(y_true):
    """Convert y_pred into output dimensions first."""
    if len(y_true.shape) == 1:
        return np.array([y_true])
    else:
        return y_true.swapaxes(0, 1)


def get_likelihoods(y_true, y_pred, min_likelihood, multivar=False):
    curr_idx = 0

    if multivar:
        _, n_components, weights, types, dists, paramss = \
            get_components(curr_idx, y_pred)

        n_instances = np.zeros(n_components)
        # negative log likelihoods of each output dimension and instance

        log_lks = np.zeros((n_components, y_true.shape[1]))
        # pointer within the vector representation of mixtures y_pred[i]

        for i in range(n_components):
            curr_idx = 0
            weighted_probs = np.zeros(y_true.shape[1])
            for j_dim, y_true_dim in enumerate(y_true):
                curr_idx, n_components, weights, types, dists, paramss = \
                    get_components(curr_idx, y_pred)
                non_empty_mask = ~np.array(types[:, i] == EMPTY_DIST)
                probs = dists[i].pdf(
                    y_true_dim[non_empty_mask],
                    *paramss[i][non_empty_mask].swapaxes(0, 1))
                # Some continuous scipy distributions (e.g. Gamma) return
                # infinity at certain singular points (like 0). From our
                # point of view the likelihood should be 0 at those point
                probs[np.isinf(probs)] = 0
                weighted_probs[non_empty_mask] += \
                    weights[:, i][non_empty_mask] * probs
            valid_mask = np.array(weighted_probs > min_likelihood)
            n_instances[i] = valid_mask.sum()
            log_lks[i, valid_mask] = -np.log(weighted_probs[valid_mask])
    else:
        n_instances = np.zeros(y_true.shape[0])
        log_lks = np.zeros(y_true.shape)
        for j_dim, y_true_dim in enumerate(y_true):
            curr_idx, n_components, weights, types, dists, paramss = \
                get_components(curr_idx, y_pred)
            weighted_probs = np.zeros(len(y_true_dim))
            for i in range(n_components):
                non_empty_mask = ~np.array(types[:, i] == EMPTY_DIST)
                probs = dists[i].pdf(
                    y_true_dim[non_empty_mask],
                    *paramss[i][non_empty_mask].swapaxes(0, 1))
                # Some continuous scipy distributions (e.g. Gamma) return
                # infinity at certain singular points (like 0). From our
                # point of view the likelihood should be 0 at those point
                probs[np.isinf(probs)] = 0
                weighted_probs[non_empty_mask] += \
                    weights[:, i][non_empty_mask] * probs
            valid_mask = np.array(weighted_probs > min_likelihood)
            n_instances[j_dim] = valid_mask.sum()
            log_lks[j_dim, valid_mask] = -np.log(weighted_probs[valid_mask])
    return log_lks, n_instances


class MDNegativeLogLikelihood(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='nll', precision=2,
                 min_likelihood=1.4867195147342979e-06,  # 5 sigma
                 output_dim=None, verbose=False, multivar=False):
        self.name = name
        self.precision = precision
        self.output_dim = output_dim
        self.min_likelihood = min_likelihood
        self.verbose = verbose
        self.multivar = multivar

    def __call__(self, y_true, y_pred):
        y_true = convert_y_true(y_true)  # output dimension first

        if self.multivar:
            log_lks, n_instances = get_likelihoods(
                y_true, y_pred, self.min_likelihood, self.multivar)
        else:
            log_lks, n_instances = get_likelihoods(
                y_true, y_pred, self.min_likelihood)
        if self.output_dim is None:
            if self.verbose:
                return log_lks.sum() / n_instances.sum(), log_lks
            else:
                return log_lks.sum() / n_instances.sum()
        elif not self.multivar:
            return (np.sum(log_lks[self.output_dim]) /
                    n_instances[self.output_dim])
        else:
            print("Non supported when doin multivariate")


class MDOutlierRate(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='or', precision=2,
                 min_likelihood=1.4867195147342979e-06,  # 5 sigma
                 output_dim=None, verbose=False):
        self.name = name
        self.precision = precision
        self.output_dim = output_dim
        self.min_likelihood = min_likelihood
        self.verbose = verbose

    def __call__(self, y_true, y_pred):
        y_true = convert_y_true(y_true)  # output dimension first
        log_lks, n_instances = get_likelihoods(
            y_true, y_pred, self.min_likelihood)
        if self.output_dim is None:
            return 1 - n_instances.sum() / (y_true.shape[0] * y_true.shape[1])
        else:
            return 1 - n_instances[self.output_dim] / y_true.shape[1]


class MDLikelihoodRatio(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='lr', precision=2,
                 min_likelihood=1.4867195147342979e-06,  # 5 sigma
                 output_dim=None, verbose=False, multivar=False):
        self.name = name
        self.precision = precision
        self.output_dim = output_dim
        self.min_likelihood = min_likelihood
        self.verbose = verbose
        self.multivar = multivar

    def __call__(self, y_true, y_pred):
        n_instances = len(y_true)
        nll_reg_score = MDNegativeLogLikelihood(
            multivar=self.multivar,
            min_likelihood=self.min_likelihood,
            output_dim=self.output_dim,
            verbose=self.verbose)
        if self.verbose:
            nll_reg, log_lks = nll_reg_score(y_true, y_pred)
        else:
            nll_reg = nll_reg_score(y_true, y_pred)
        y_true = convert_y_true(y_true)

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
                       np.sum(baseline_lls, axis=1) / n_instances))
        return np.exp(-nll_reg - np.sum(baseline_lls) / n_instances / n_dims)


class MDRMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='rmse', precision=2, output_dim=None,
                 verbose=False):
        self.name = name
        self.precision = precision
        self.output_dim = output_dim
        self.verbose = verbose

    def __call__(self, y_true, y_pred):
        n_instances = len(y_true)
        y_true = convert_y_true(y_true)  # output dimension first
        n_dims = len(y_true)
        mean_preds = np.zeros(y_true.shape)
        # pointer within the vector representation of mixtures y_pred[i]
        curr_idx = 0
        for j_dim, y_true_dim in enumerate(y_true):
            curr_idx, n_components, weights, types, dists, paramss = \
                get_components(curr_idx, y_pred)
            for i in range(n_components):
                non_empty_mask = ~np.array(types[:, i] == EMPTY_DIST)
                means = dists[i].mean(
                    *paramss[i][non_empty_mask].swapaxes(0, 1))
                mean_preds[j_dim, non_empty_mask] += \
                    weights[:, i][non_empty_mask] * means

        if self.output_dim is None:
            if self.verbose:
                errors = y_true - mean_preds
                return np.sqrt(
                    (errors ** 2).sum() / n_instances / n_dims), errors
            else:
                return np.sqrt(
                    ((y_true - mean_preds) ** 2).sum() / n_instances / n_dims)
        else:
            return np.sqrt(
                ((y_true[self.output_dim] - mean_preds[self.output_dim])
                 ** 2).sum() / n_instances)


class MDR2(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='r2', precision=2, output_dim=None):
        self.name = name
        self.precision = precision
        self.output_dim = output_dim

    def __call__(self, y_true, y_pred):
        y_true = convert_y_true(y_true)  # output dimension first
        stds = np.std(y_true, axis=1)
        mean_preds = np.zeros(y_true.shape)
        # pointer within the vector representation of mixtures y_pred[i]
        curr_idx = 0
        for j_dim, y_true_dim in enumerate(y_true):
            curr_idx, n_components, weights, types, dists, paramss = \
                get_components(curr_idx, y_pred)
            for i in range(n_components):
                non_empty_mask = ~np.array(types[:, i] == EMPTY_DIST)
                means = dists[i].mean(
                    *paramss[i][non_empty_mask].swapaxes(0, 1))
                mean_preds[j_dim, non_empty_mask] += \
                    weights[:, i][non_empty_mask] * means
        r2s = ((y_true - mean_preds) ** 2).mean(axis=1)
        r2s /= stds ** 2
        r2s = 1 - r2s

        if self.output_dim is None:
            return r2s.mean()
        else:
            return r2s[self.output_dim]


class MDKSCalibration(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='ks', precision=2, output_dim=None, verbose=False):
        self.name = name
        self.precision = precision
        self.output_dim = output_dim
        self.verbose = verbose

    def __call__(self, y_true, y_pred):
        n_instances = len(y_true)
        y_true = convert_y_true(y_true)  # output dimension first
        n_dims = len(y_true)
        cdfs = np.zeros(y_true.shape)
        # pointer within the vector representation of mixtures y_pred[i]
        curr_idx = 0
        for j_dim, y_true_dim in enumerate(y_true):
            curr_idx, n_components, weights, types, dists, paramss = \
                get_components(curr_idx, y_pred)
            for i in range(n_components):
                non_empty_mask = ~np.array(types[:, i] == EMPTY_DIST)
                cdfss = dists[i].cdf(
                    y_true_dim[non_empty_mask],
                    *paramss[i][non_empty_mask].swapaxes(0, 1))
                cdfs[j_dim, non_empty_mask] += \
                    weights[:, i][non_empty_mask] * cdfss
        ks_stats = np.zeros(n_dims)
        for j_dim in range(n_dims):
            ks_stats[j_dim] = np.max(np.abs(
                np.sort(cdfs[j_dim]) - np.arange(n_instances) / n_instances))

        if self.output_dim is None:
            if self.verbose:
                return ks_stats.mean(), cdfs
            else:
                return ks_stats.mean()
        else:
            return ks_stats[self.output_dim]
