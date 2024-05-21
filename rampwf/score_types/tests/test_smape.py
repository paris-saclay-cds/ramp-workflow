import numpy as np

from rampwf.score_types.smape import SMAPE


def test_zeros():
    a = np.zeros(2)
    b = np.zeros(2)
    score = SMAPE()(a, b)
    assert score == 0


def test_smape():
    a = np.array([0, 1, 2])
    b = np.array([-1, 3, 2])
    score = SMAPE()(a, b)
    assert score == 100
