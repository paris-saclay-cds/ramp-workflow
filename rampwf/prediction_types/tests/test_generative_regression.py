"""Testing for generative regression predictions """


from rampwf.prediction_types import make_generative_regression
import numpy as np
from numpy.testing import assert_array_equal


def test_init_empty():
    NBINS = 1
    Predictions = make_generative_regression(NBINS)
    predictions = Predictions(n_samples=2)
    assert_array_equal(predictions.y_pred, np.array([[[np.nan, np.nan, np.nan]], [[np.nan, np.nan, np.nan]]]))

    Predictions = make_generative_regression(NBINS, label_names=[1, 2])
    predictions = Predictions(n_samples=2)
    assert_array_equal(predictions.y_pred, np.array([[[np.nan, np.nan, np.nan],
                                                      [np.nan, np.nan, np.nan]],
                                                     [[np.nan, np.nan, np.nan],
                                                      [np.nan, np.nan, np.nan]]]))

