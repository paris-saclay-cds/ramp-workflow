"""Testing for generative regression predictions """


from rampwf.prediction_types import make_generative_regression
import numpy as np
from numpy.testing import assert_array_equal
from ...utils import MAX_MDN_PARAMS


def test_init_empty():
    # check the shapes of initialized empty predictions for
    # generative regression
    max_dists = 1
    Predictions = make_generative_regression(
        max_dists, label_names=[1])
    predictions = Predictions(n_samples=2)
    assert_array_equal(
        predictions.y_pred, np.array(
            [[np.nan] * (1 + (2 + MAX_MDN_PARAMS))] * 2))
    max_dists = 4
    Predictions = make_generative_regression(max_dists, label_names=[1, 2])
    predictions = Predictions(n_samples=2)
    assert_array_equal(
        predictions.y_pred, np.array(
            [[np.nan] * 2 * (1 + (2 + MAX_MDN_PARAMS) * max_dists)] * 2))
