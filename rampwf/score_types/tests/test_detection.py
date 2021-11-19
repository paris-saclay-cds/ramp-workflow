import math

import numpy as np

import pytest

from rampwf.score_types.detection.scp import scp_single
from rampwf.score_types.detection.ospa import ospa, ospa_single
from rampwf.score_types import DetectionAveragePrecision
from rampwf.score_types.detection.precision_recall import precision, recall
from rampwf.score_types.detection.precision_recall import mad_center
from rampwf.score_types.detection.precision_recall import mad_radius
from rampwf.score_types.detection.iou import cc_iou, cc_intersection
from rampwf.score_types.detection.scp import project_circle, circle_maps
from rampwf.score_types.detection.average_precision import (
    precision_recall_curve_greedy)


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


def test_average_precision():
    ap = DetectionAveragePrecision()

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
    y_true = [[], []]
    y_pred = [[], []]
    assert ap(y_true, y_pred) == 0
    y_true = [[(1, 1, 1)], []]
    y_pred = [[], []]
    assert ap(y_true, y_pred) == 0

    # only empty predictions
    y_true = [[(1, 1, 1)]]
    y_pred = [[(1, 3, 3, 1)]]
    assert ap(y_true, y_pred) == 0

    # bigger example
    y_true = [[(1, 1, 1), (3, 3, 1)], [(1, 1, 1), (3, 3, 1)]]
    y_pred = [[(0.9, 1, 1, 1), (0.7, 5, 5, 1), (0.5, 8, 8, 1)],
              [(0.8, 1, 1, 1), (0.6, 3, 3, 1), (0.4, 5, 5, 1)]]

    conf, ps, rs = precision_recall_curve_greedy(y_true, y_pred)
    assert conf.tolist() == [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    assert ps.tolist() == [1, 1, 2 / 3, 3 / 4, 3 / 5, 3 / 6]  # noqa
    assert rs.tolist() == [1 / 4, 2 / 4, 2 / 4, 3 / 4, 3 / 4, 3 / 4]  # noqa
    assert ap(y_true, y_pred) == 11 / 16  # 0.5 * 1 + 0.25 * 3/4 + 0.25 * 0


# # test circles


circle = (1, 1, 1)
x = [circle]


def test_project_circle():
    shape = (10, 10)
    assert project_circle(circle, image=np.zeros(shape)).shape == shape
    assert project_circle(circle, shape=shape).shape == shape

    classic = project_circle(circle, shape=shape)
    assert classic.min() == 0
    assert classic.max() == 1

    negative = project_circle(circle, shape=shape, negative=True)
    assert negative.min() == -1
    assert negative.max() == 0

    normalized = project_circle(circle, shape=shape, normalize=True)
    assert normalized.min() == 0
    assert normalized.sum() == 1

    normalized_neg = project_circle(circle, shape=shape,
                                    normalize=True, negative=True)
    assert normalized_neg.max() == 0
    assert normalized_neg.sum() == -1

    with pytest.raises(ValueError):
        project_circle(circle)
        project_circle(circle, image=None, shape=None)


def test_circle_map():
    shape = (10, 10)
    map_true, map_pred = circle_maps([], [], shape)
    assert map_true.max() == 0
    assert map_pred.max() == 0
    assert map_true.sum() == 0
    assert map_pred.sum() == 0
    map_true, map_pred = circle_maps(x, [], shape)
    assert map_true.sum() == 1
    assert map_pred.sum() == 0
    map_true, map_pred = circle_maps([], x, shape)
    assert map_true.sum() == 0
    assert map_pred.sum() == 1
    map_true, map_pred = circle_maps(x, x, shape)
    assert map_true.sum() == 1
    assert map_pred.sum() == 1


# # test IOU


def test_cc_iou():
    circle1 = (0, 0, 1)
    circle2 = (0, 4, 1)
    circle3 = (1, 1, 2)
    circle1_2 = (0, 0, 2)
    assert cc_iou(circle1, circle1) - 1 < 1e-6
    assert cc_iou(circle1, circle2) < 1e-6
    assert cc_iou(circle2, circle1) < 1e-6
    assert cc_iou(circle1, circle3) - math.pi < 1e-6
    assert cc_iou(circle3, circle1) - math.pi < 1e-6
    assert cc_iou(circle1_2, circle1) == 0.25
    assert cc_iou(circle1, circle1_2) == 0.25


def test_cc_intersection():
    # Zero distance
    assert cc_intersection(0, 1, 2) - 4 * math.pi < 1e-6

    # Zero radius
    assert cc_intersection(1, 0, 1) == 0
    assert cc_intersection(1, 1, 0) == 0

    # High distance
    assert cc_intersection(4, 1, 2) == 0

    # Classic test
    assert cc_intersection(1, 1, 2) - math.pi < 1e-6

    with pytest.raises(ValueError):
        cc_intersection(-1, 1, 1)
        cc_intersection(1, -1, 1)
        cc_intersection(1, 1, -1)

    # floating point corner case
    assert cc_intersection(11.0, 6.0, 4.999999999999999) == 0


def test_cc_intersection_completely_overlapping():
    cc_intersection(2, 4, 1) - math.pi < 1e-6
