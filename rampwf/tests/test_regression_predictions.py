"""Testing for regression predictions (rampwf.regression.multiclass)."""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import numpy as np
from rampwf.prediction_types.regression import Predictions
from numpy.testing import assert_array_equal


def test_init():
    y_pred = [0.7, 0.1, 0.2]
    predictions = Predictions(y_pred=y_pred)
    assert_array_equal(predictions.y_pred, y_pred)


def test_init_empty():
    predictions = Predictions(shape=3)
    assert_array_equal(predictions.y_pred, np.array([np.nan, np.nan, np.nan]))
    predictions = Predictions(shape=(4,))
    assert_array_equal(predictions.y_pred, np.array([
        np.nan, np.nan, np.nan, np.nan,
    ]))
    # multi-target regression
    predictions = Predictions(shape=(4, 3))
    assert_array_equal(predictions.y_pred, np.array([
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ]))


def test_set_valid_in_train():
    y_pred = [0.7, 0.1, 0.2]
    predictions = Predictions(y_pred=y_pred)
    cv_y_pred = [0.6, 0.3]
    cv_predictions = Predictions(y_pred=cv_y_pred)
    updated_y_pred = [0.6, 0.1, 0.3]
    updated_predictions = Predictions(y_pred=updated_y_pred)
    predictions.set_valid_in_train(cv_predictions, [0, 2])
    assert_array_equal(predictions.y_pred, updated_predictions.y_pred)


def test_set_valid_in_train2():
    predictions = Predictions(shape=(4,))
    y_pred = [0.7, 0.1, 0.2]
    predictions_valid = Predictions(y_pred=y_pred)
    predictions.set_valid_in_train(predictions_valid, [0, 2, 3])
    assert_array_equal(predictions.y_pred, np.array([
        0.7, np.nan, 0.1, 0.2]))
