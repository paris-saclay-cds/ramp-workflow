"""Testing for detection predictions (rampwf.prediction.detection)."""

import numpy as np
from numpy.testing import assert_array_equal

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
