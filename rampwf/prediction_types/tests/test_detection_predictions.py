"""Testing for detection predictions (rampwf.prediction.detection)."""

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from rampwf.prediction_types.detection import Predictions


def test_combine():

    pred1 = Predictions(
        y_pred=[[(1, 1, 1, 1), (0.65, 3, 3, 1)]])
    pred2 = Predictions(
        y_pred=[[(0.9, 1.1, 1.1, 1.1), (0.5, 3, 3, 1), (0.8, 5, 5, 1)]])
    pred3 = Predictions(
        y_pred=[[(0.7, 3, 3, 1), (0.6, 5.1, 5.1, 1), (0.3, 10, 10, 1)]])

    y_pred_combined = Predictions.combine([pred1, pred2, pred3]).y_pred

    assert y_pred_combined.shape[0] == 1
    assert len(y_pred_combined[0]) == 3


def test_combine_no_match():

    pred1 = Predictions(
        y_pred=[[], [(1, 1, 1, 1)]])
    pred2 = Predictions(
        y_pred=[[], [(1, 3, 3, 1)]])

    y_pred_combined = Predictions.combine([pred1, pred2]).y_pred

    expected = np.empty(2, dtype=object)
    expected[:] = [[], []]

    assert_array_equal(expected, y_pred_combined)


def test_combine_ignore_none():

    pred1 = Predictions(
        y_pred=[[(1., 1, 1, 1)], None])
    pred2 = Predictions(
        y_pred=[[(1., 1, 1, 1)], [(1., 3, 3, 1)]])
    pred3 = Predictions(
        y_pred=[[(1., 3, 3, 1)], [(1., 3, 3, 1)]])

    y_pred_combined = Predictions.combine([pred1, pred2, pred3]).y_pred

    # first item: 2 out of 3 match -> confidence of 2/3
    # second item: ignore None, so 2 out of 2 match -> confidence of 2/2 = 1
    expected = np.empty(2, dtype=object)
    expected[:] = [[(2. / 3, 1, 1, 1)], [(1, 3, 3, 1)]]
    assert_allclose(expected[0], y_pred_combined[0])
    assert_allclose(expected[1], y_pred_combined[1])

    # corner case: no overlap in folds
    pred1 = Predictions(
        y_pred=[[(1., 1, 1, 1)], None, None])
    pred2 = Predictions(
        y_pred=[None, [(1., 3, 3, 1)], None])
    pred3 = Predictions(
        y_pred=[None, None, [(1., 3, 3, 1)]])

    y_pred_combined = Predictions.combine([pred1, pred2, pred3]).y_pred

    expected = [[(1., 1, 1, 1)], [(1., 3, 3, 1)], [(1., 3, 3, 1)]]
    assert len(expected) == len(y_pred_combined)
    [assert_array_equal(x, y) for x, y in zip(y_pred_combined, expected)]


def test_combine_zero_conf():

    pred1 = Predictions(
        y_pred=[[(0, 1, 1, 1)], [(0, 1, 1, 1)]])
    pred2 = Predictions(
        y_pred=[[(1, 1.1, 1.1, 1)], [(0, 1, 1, 1)]])

    y_pred_combined = Predictions.combine([pred1, pred2]).y_pred

    expected = [[(0.5, 1.1, 1.1, 1)], [(0., 1, 1, 1)]]
    [assert_array_equal(x, y) for x, y in zip(y_pred_combined, expected)]
