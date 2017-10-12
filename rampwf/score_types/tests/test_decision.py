import pytest

from rampwf.score_types import AverageDetectionPrecision


def test_average_precision():
    ap = AverageDetectionPrecision()

    # perfect match
    y_true = [[(1, 1, 1), (3, 3, 1)]]
    y_pred = [[(1, 1, 1, 1), (1, 3, 3, 1)]]
    assert ap(y_true, y_pred) == 1

    # imperfect match
    y_true = [[(1, 1, 1), (3, 3, 1), (7, 7, 1), (9, 9, 1)]]
    y_pred = [[(1, 1, 1, 1), (1, 5, 5, 1)]]
    assert ap(y_true, y_pred) == pytest.approx(3. / 2 / 11, rel=1e-5)
    # would be 0.125 (1 / 8) exact method

    y_true = [[(1, 1, 1), (3, 3, 1), (7, 7, 1), (9, 9, 1)]]
    y_pred = [[(1, 1, 1.2, 1.2), (1, 3, 3, 1)]]
    assert ap(y_true, y_pred) == pytest.approx(6. / 11, rel=1e-5)
    # would be 0.5 with exact method

    # no match
    y_true = [[(1, 1, 1)]]
    y_pred = [[(1, 3, 3, 1)]]
    assert ap(y_true, y_pred) == 0
