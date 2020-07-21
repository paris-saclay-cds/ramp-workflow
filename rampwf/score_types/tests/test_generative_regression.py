from __future__ import division

import numpy as np

from rampwf.score_types.generative_regression import (
    MDNegativeLogLikelihood, MDLikelihoodRatio, MDRMSE,
    MDR2, MDKSCalibration, MDOutlierRate)

import pytest

y_result_1 = np.array([[1, 1, 0, 0.5, 0.5]*2])
y_result_2 = np.array([[1, 1, 0, 0.1, 0.5]*2])
y_truth_1 = np.array([[0.5], [0.51]])

y_result_3 = np.array([[[0, 1.1, 2, 0.9, 0.1]]])
y_result_4 = np.array([[[0, 1.9, 2, 0.1, 0.9]]])
y_truth_2 = np.array([1.6])

y_result_5 = np.array([[[0, 1, 2, 0.1, 0.9],
                        [0, 1, 2, 0.9, 0.1]]])

y_result_6 = np.array([[[0, 1, 2, 0.9, 0.1],
                        [0, 1, 2, 0.1, 0.9]]])
y_truth_2 = np.array([[1.5, 1.5]])

y_result_out = np.array([[[0, 1, 2, 0.1, 0.9]]])
y_truth_out_1 = np.array([2.5])
y_truth_out_2 = np.array([-2.5])
y_truth_out_3 = np.array([-0.5])

y_truth_fake = np.array([0.5])

y_result_wrong_bins = np.array([[[0, -1, 2, 0.1, 0.9]]])
y_result_wrong_bins_2 = np.array([[[0, 0, 2, 0.1, 0.9]]])
y_result_wrong_proba = np.array([[[0, 1, 2, 0.2, 0.2]]])
y_result_corrected_proba = np.array([[[0, 1, 2, 0.5, 0.5]]])
y_truth_wrong = np.array([0.5])


def test_likelihood():
    score = MDLikelihoodRatio()
    assert score(y_truth_1, y_result_1) > score(y_truth_1, y_result_2)

    val = np.exp(-1) - 10e-6
    y_custom = np.array([[[0, 1, 2, 1 - val, val]]])
    assert score(y_truth_1, y_custom) == pytest.approx(1.0, 10e-5)

    val = np.exp(-0.1) - 10e-6
    y_custom = np.array([[[0, 1, 2, 1 - val, val]]])
    assert score(y_truth_1, y_custom) == pytest.approx(0.1, 10e-5)

    # We have a confidence of .1 on the truth interval in both cases, but is it smaller in the first example
    assert score(y_truth_2, y_result_3) < score(y_truth_2, y_result_4)

    assert score(y_truth_2, y_result_5) == score(y_truth_2, y_result_6)


def test_binned_likelihood_outside():
    score = NegativeLogLikelihoodReg(n_bins=NBINS)
    # outside
    assert score(y_truth_out_1, y_result_out) == score(y_truth_out_2, y_result_out)

    # Completely wrong
    y_custom = np.array([[[0, 1, 2, 0, 1]]])

    assert score(y_truth_fake, y_custom) == score(y_truth_out_3, y_result_out)


def test_wrong_proba_wrong_bins():
    score = NegativeLogLikelihoodReg(n_bins=NBINS)
    with pytest.raises(Exception):
        assert score(y_truth_wrong, y_result_wrong_bins)
    with pytest.raises(Exception):
        assert score(y_truth_wrong, y_result_wrong_bins_2)
    assert score(y_truth_wrong, y_result_wrong_proba) == score(y_truth_wrong, y_result_corrected_proba)
