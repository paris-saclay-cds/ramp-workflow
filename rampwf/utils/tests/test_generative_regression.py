import numpy as np
from numpy.testing import assert_array_almost_equal

from rampwf.utils.generative_regression import BaseGenerativeRegressor


def test_samples_to_distribution():
    # gross check of returned sigma values

    # create samples array of shape 1 timestep * 2 targets * 10 samples
    rng = np.random.RandomState(0)
    n_samples = 100
    X1 = np.sqrt(2) * rng.randn(1, n_samples)  # variance of 2
    X2 = rng.randn(1, n_samples)  # variance of 1
    samples = np.concatenate([X1, X2], axis=0)
    samples = samples[np.newaxis, :]  # one timestep
    n_targets = samples.shape[1]

    base_gen_reg = BaseGenerativeRegressor()
    _, _, params = base_gen_reg.samples_to_distributions(samples)

    sigmas = params[:, 1::2].ravel()
    silverman_factor = (
        n_samples * (n_targets + 2) / 4.)**(-1. / (n_targets + 4))
    data_variances = sigmas ** 2 / silverman_factor ** 2
    true_variances = np.repeat([2, 1], n_samples)
    assert_array_almost_equal(data_variances, true_variances, decimal=1)
