from __future__ import division

import numpy as np

from rampwf.score_types.negative_log_likelihood import logLKGenerative
import pytest

NBINS = 2

y_result_1 = np.array([[[0, 1, 2, 0.5, 0.5]]])
y_result_2 = np.array([[[0, 1, 2, 0.1, 0.9]]])
y_truth_1 = np.array([1.5])

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


def test_binned_likelihood():
    score_1 = logLKGenerative(nb_bins=NBINS)
    assert score_1(y_truth_1, y_result_1) > score_1(y_truth_1, y_result_2)

    val = np.exp(-1) - 10e-6
    y_custom = np.array([[[0, 1, 2, 1 - val, val]]])
    assert score_1(y_truth_1, y_custom) == pytest.approx(1.0, 10e-5)

    val = np.exp(-0.1) - 10e-6
    y_custom = np.array([[[0, 1, 2, 1 - val, val]]])
    assert score_1(y_truth_1, y_custom) == pytest.approx(0.1, 10e-5)

    # We have a confidence of .1 on the truth inteval in both cases, but is it smaller in the first example
    assert score_1(y_truth_2, y_result_3) < score_1(y_truth_2, y_result_4)

    assert score_1(y_truth_2, y_result_5) == score_1(y_truth_2, y_result_6)


def test_binned_likelihood_outside():
    score_2 = logLKGenerative(nb_bins=NBINS)
    # outside
    assert score_2(y_truth_out_1, y_result_out) == score_2(y_truth_out_2, y_result_out)

    # Completely wrong
    y_custom = np.array([[[0, 1, 2, 0, 1]]])

    assert score_2(y_truth_fake, y_custom) == score_2(y_truth_out_3, y_result_out)
