import os
import numpy as np
import tempfile
import os.path as op
from rampwf.predictions.regression import Predictions
from numpy.testing import assert_equal, assert_array_equal


def test_init():
    y_pred = [0.7, 0.1, 0.2]
    predictions = Predictions(y_pred=y_pred)
    assert_array_equal(predictions.y_pred, y_pred)


def test_save_and_load():
    y_pred = [0.7, 0.1, 0.2]
    predictions = Predictions(y_pred=y_pred)
    f_name = op.join(tempfile.gettempdir(), 'predictions.npy')
    predictions.save(f_name)
    loaded_predictions = Predictions(f_name=f_name)
    assert_array_equal(predictions.y_pred, loaded_predictions.y_pred)
    assert_equal(predictions.n_samples, 3)
    os.remove(f_name)


def test_set_valid_in_train():
    y_pred = [0.7, 0.1, 0.2]
    predictions = Predictions(y_pred=y_pred)
    cv_y_pred = [0.6, 0.3]
    cv_predictions = Predictions(y_pred=cv_y_pred)
    updated_y_pred = [0.6, 0.1, 0.3]
    updated_predictions = Predictions(y_pred=updated_y_pred)
    predictions.set_valid_in_train(cv_predictions, [0, 2])
    assert_array_equal(predictions.y_pred, updated_predictions.y_pred)


def test_init_to_nan():
    predictions = Predictions(n_samples=3)
    assert_array_equal(predictions.y_pred, np.array([np.nan, np.nan, np.nan]))
