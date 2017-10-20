import pytest

from rampwf.score_types import AverageDetectionPrecision
from rampwf.score_types.detection import precision_recall_curve_greedy

def test_average_precision():
    ap = AverageDetectionPrecision()

    # perfect match
    y_true = [[(1, 1, 1), (3, 3, 1)]]
    y_pred = [[(1, 1, 1, 1), (1, 3, 3, 1)]]
    assert ap(y_true, y_pred) == 1

    # imperfect match
    y_true = [[(1, 1, 1), (3, 3, 1), (7, 7, 1), (9, 9, 1)]]
    y_pred = [[(1, 1, 1, 1), (1, 5, 5, 1)]]
    assert ap(y_true, y_pred) == 0.125

    y_true = [[(1, 1, 1), (3, 3, 1), (7, 7, 1), (9, 9, 1)]]
    y_pred = [[(1, 1, 1.2, 1.2), (1, 3, 3, 1)]]
    assert ap(y_true, y_pred) == 0.5

    # no match
    y_true = [[(1, 1, 1)]]
    y_pred = [[(1, 3, 3, 1)]]
    assert ap(y_true, y_pred) == 0

    # bigger example
    y_true = [[(1, 1, 1), (3, 3, 1)], [(1, 1, 1), (3, 3, 1)]]
    y_pred = [[(0.9, 1, 1, 1), (0.7, 5, 5, 1), (0.5, 8, 8, 1)],
              [(0.8, 1, 1, 1), (0.6, 3, 3, 1), (0.4, 5, 5, 1)]]

    conf, ps, rs = precision_recall_curve_greedy(y_true, y_pred)
    assert conf.tolist() == [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    assert ps.tolist() == [1, 1, 2/3, 3/4, 3/5, 3/6]
    assert rs.tolist() == [1/4, 2/4, 2/4, 3/4, 3/4, 3/4]
    assert ap(y_true, y_pred) == 11/16  # 0.5 * 1 + 0.25 * 3/4 + 0.25 * 0
