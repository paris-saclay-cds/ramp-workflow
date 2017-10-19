from __future__ import division

import math
import pytest
# import numpy as np

# from rampwf.score_types import AverageDetectionPrecision
from rampwf.score_types.detection import ospa_single
from rampwf.score_types.detection import ospa
from rampwf.score_types.detection import scp_single
from rampwf.score_types.detection import precision
from rampwf.score_types.detection import recall
from rampwf.score_types.detection import mad_center
from rampwf.score_types.detection import mad_radius


x = [(1, 1, 1)]
x2 = [(1, 1, 2)]
y = [(1, 3, 1)]
z = x + y
minipatch = [0, 2, 0, 2]


def test_scp_single():
    shape = (10, 10)
    # Perfect match
    assert scp_single(x, x, shape) == (0, 1, 1)
    # No match
    assert scp_single(x, y, shape) == (2, 1, 1)
    assert scp_single(x, x2, shape)[0] > 0
    # 1 match, 1 miss
    assert scp_single(x, z, shape) == (1, 1, 2)
    # 1 empty, 1 not
    assert scp_single(x, [], shape) == (1, 1, 0)
    assert scp_single([], x, shape) == (1, 0, 1)
    # 2 empty arrays
    assert scp_single([], [], shape) == (0, 0, 0)

    # Both inside minipatch
    assert scp_single(x, x, shape, minipatch=minipatch) == (0, 1, 1)
    # One mismatch
    assert scp_single(x, y, shape, minipatch=minipatch) == (1, 1, 0)
    # One too big
    assert scp_single(x, z, shape, minipatch=minipatch) == (0, 1, 1)
    assert scp_single(x, [], shape, minipatch=minipatch) == (1, 1, 0)
    assert scp_single([], x, shape, minipatch=minipatch) == (1, 0, 1)
    assert scp_single([], [], shape, minipatch=minipatch) == (0, 0, 0)

    # object outside of the image is discarded, although it actually is a
    # bad detection
    scp_single(x, [(20, 20, 1)], shape) == (1, 1, 0)


def test_ospa_single():
    # Perfect match
    assert ospa_single(x, x) == (2, 2, 2)
    # No match
    assert ospa_single(x, y) == (0, 2, 2)
    # assert ospa_single(x, x2) > 0
    # Miss or wrong radius
    assert ospa_single(x, z) == (2, 2, 3)
    # An empty array with a non empty one
    assert ospa_single(x, []) == (0, 0, 1)
    assert ospa_single([], x) == (0, 0, 1)
    assert ospa_single(z, []) == (0, 0, 2)
    # Two empty arrays should match
    assert ospa_single([], []) == (0, 0, 0)

    assert ospa_single(x, x, minipatch=minipatch) == (2, 2, 2)
    assert ospa_single(x, y, minipatch=minipatch) == (0, 1, 1)
    assert ospa_single(x, z, minipatch=minipatch) == (2, 2, 2)
    assert ospa_single(x, z, minipatch=minipatch) == (2, 2, 2)
    assert ospa_single(x, [], minipatch=minipatch) == (0, 0, 1)
    assert ospa_single([], x, minipatch=minipatch) == (0, 0, 1)
    assert ospa_single([], [], minipatch=minipatch) == (0, 0, 0)


def test_ospa():
    # Perfect match
    assert ospa([x], [x]) == 1
    assert ospa([x, x], [x, x]) == 1
    assert ospa([x, x], [x, y]) == 0.5
    assert ospa([x, x], [x, z]) == 4 / 5
    # No match
    assert ospa([x], [y]) == 0
    # assert ospa([x], [x]2) > 0
    assert ospa([x], [z]) == 2 / 3
    # Miss or wrong radius
    # An empty array with a non empty one
    assert ospa([x], [[]]) == 0
    assert ospa([[]], [x]) == 0
    assert ospa([z], [[]]) == 0
    # Two empty arrays should match
    assert ospa([[]], [[]]) == 0

    assert ospa([x], [x], minipatch=minipatch) == 2 / 2
    assert ospa([x, x], [x, x], minipatch=minipatch) == (2 + 2) / (2 + 2)
    assert ospa([x, x], [x, y], minipatch=minipatch) == (2 + 0) / (2 + 1)
    assert ospa([x, x], [x, z], minipatch=minipatch) == (2 + 2) / (2 + 2)
    assert ospa([x, x], [x, []], minipatch=minipatch) == (2 + 0) / (2 + 1)
    assert ospa([x, []], [x, x], minipatch=minipatch) == (2 + 0) / (2 + 1)
    assert ospa([x, []], [x, []], minipatch=minipatch) == (2 + 0) / (2 + 0)


def test_precision_recall():
    # perfect match
    y_true = [[(1, 1, 1), (3, 3, 1)]]
    y_pred = [[(1, 1, 1), (3, 3, 1)]]
    assert precision(y_true, y_pred) == 1
    assert recall(y_true, y_pred) == 1
    assert mad_radius(y_true, y_pred) == 0
    assert mad_center(y_true, y_pred) == 0

    # partly perfect match
    y_true = [[(1, 1, 1), (3, 3, 1), (7, 7, 1), (9, 9, 1)]]
    y_pred = [[(1, 1, 1), (5, 5, 1)]]
    assert precision(y_true, y_pred) == 0.5
    assert recall(y_true, y_pred) == 0.25
    assert mad_radius(y_true, y_pred) == 0
    assert mad_center(y_true, y_pred) == 0

    # imperfect match
    y_true = [[(1, 1, 1), (3, 3, 1), (7, 7, 1), (9, 9, 1)]]
    y_pred = [[(1, 1.2, 1.2), (3, 3, 1)]]
    assert precision(y_true, y_pred) == 1
    assert recall(y_true, y_pred) == 0.5
    assert mad_radius(y_true, y_pred) == pytest.approx(0.1)
    assert mad_center(y_true, y_pred) == pytest.approx(0.1)

    # no match
    y_true = [[(1, 1, 1)]]
    y_pred = [[(3, 3, 1)]]
    assert precision(y_true, y_pred) == 0
    assert recall(y_true, y_pred) == 0
    assert math.isnan(mad_radius(y_true, y_pred))
    assert math.isnan(mad_center(y_true, y_pred))


# def test_average_precision():
#     ap = AverageDetectionPrecision()
#     # perfect match
#     y_true = [[(1, 1, 1), (3, 3, 1)]]
#     y_pred = [[(1, 1, 1, 1), (1, 3, 3, 1)]]
#     assert ap(y_true, y_pred) == 1

#     # imperfect match
#     y_true = [[(1, 1, 1), (3, 3, 1), (7, 7, 1), (9, 9, 1)]]
#     y_pred = [[(1, 1, 1, 1), (1, 5, 5, 1)]]
#     assert ap(y_true, y_pred) == pytest.approx(3. / 2 / 11, rel=1e-5)
#     # would be 0.125 (1 / 8) exact method

#     y_true = [[(1, 1, 1), (3, 3, 1), (7, 7, 1), (9, 9, 1)]]
#     y_pred = [[(1, 1, 1.2, 1.2), (1, 3, 3, 1)]]
#     assert ap(y_true, y_pred) == pytest.approx(6. / 11, rel=1e-5)
#     # would be 0.5 with exact method

#     # no match
#     y_true = [[(1, 1, 1)]]
#     y_pred = [[(1, 3, 3, 1)]]
#     assert ap(y_true, y_pred) == 0
