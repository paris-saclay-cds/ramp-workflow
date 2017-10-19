from __future__ import division

import math
import numpy as np
from scipy.optimize import linear_sum_assignment


def _select_minipatch_tuples(y_list, minipatch):
    """
    Mask over a list selecting the tuples that lie in the minipatch

    Parameters
    ----------
    y_list : list of (row, col, radius)
        List of true or predicted objects
    minipatch : [row_min, row_max, col_min, col_max]
        Bounds of the internal scoring patch

    Returns
    -------
    y_list_cut : list of bool
        List of booleans corresponding to whether the circle is in
        the minipatch or not

    """
    if len(y_list) == 0:
        return np.array([], dtype=bool)

    row_min, row_max, col_min, col_max = minipatch

    rows, cols, _ = np.asarray(y_list).T

    y_list_cut = ((rows >= row_min) &
                  (rows < row_max) &
                  (cols >= col_min) &
                  (cols < col_max))

    return y_list_cut


def _filter_minipatch_tuples(y_list, minipatch):
    """
    Mask over a list selecting the tuples that lie in the minipatch

    Parameters
    ----------
    y_list : list of (row, col, radius)
        List of true or predicted objects
    minipatch : [row_min, row_max, col_min, col_max]
        Bounds of the internal scoring patch

    Returns
    -------
    y_list_filtered : list of (row, col, radius)
        List of filtered tuples corresponding to the scoring region

    """
    in_minipatch = _select_minipatch_tuples(y_list, minipatch)

    return [tupl for (tupl, cond) in zip(y_list, in_minipatch) if cond]


def _match_tuples(y_true, y_pred):
    """
    Given set of true and predicted (x, y, r) tuples.

    Determine the best possible match.

    Parameters
    ----------
    y_true, y_pred : list of tuples

    Returns
    -------
    (idxs_true, idxs_pred, ious)
        idxs_true, idxs_pred : indices into y_true and y_pred of matches
        ious : corresponding IOU value of each match

        The length of the 3 arrays is identical and the minimum of the length
        of y_true and y_pred
    """
    n_true = len(y_true)
    n_pred = len(y_pred)

    iou_matrix = np.empty((n_true, n_pred))

    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = cc_iou(y_true[i], y_pred[j])

    idxs_true, idxs_pred = linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]
    return idxs_true, idxs_pred, ious


def _count_matches(y_true, y_pred, matches, iou_threshold=0.5):
    """
    Count the number of matches.

    Parameters
    ----------
    y_true, y_pred : list of list of tuples
    matches : optional, output of _match_tuples
    iou_threshold : float

    Returns
    -------
    (n_true, n_pred_all, n_pred_correct):
        Number of true craters
        Number of total predicted craters
        Number of correctly predicted craters

    """
    val_numbers = []

    for y_true_p, y_pred_p, match_p in zip(y_true, y_pred, matches):
        n_true = len(y_true_p)
        n_pred = len(y_pred_p)

        _, _, ious = match_p
        p = (ious > iou_threshold).sum()

        val_numbers.append((n_true, n_pred, p))

    n_true, n_pred_all, n_pred_correct = np.array(val_numbers).sum(axis=0)

    return n_true, n_pred_all, n_pred_correct


def _locate_matches(y_true, y_pred, matches, iou_threshold=0.5):
    """
    Given list of list of matching craters.

    Return contiguous array of all
    craters x, y, r.

    Parameters
    ----------
    y_true, y_pred : list of list of tuples
    matches : optional, output of _match_tuples
    iou_threshold : float

    Returns
    -------
    loc_true, loc_pred
        Each is 2D array (n_patches, 3) with x, y, r columns

    """
    loc_true = []
    loc_pred = []

    for y_true_p, y_pred_p, matches_p in zip(y_true, y_pred, matches):

        for idx_true, idx_pred, iou_val in zip(*matches_p):
            if iou_val >= iou_threshold:
                loc_true.append(y_true_p[idx_true])
                loc_pred.append(y_pred_p[idx_pred])

    if loc_true:
        return np.array(loc_true), np.array(loc_pred)
    else:
        return np.empty((0, 3)), np.empty((0, 3))


def cc_iou(circle1, circle2):
    """
    Intersection over Union (IoU) between two circles.

    Parameters
    ----------
    circle1 : tuple of floats
        first circle parameters (x_pos, y_pos, radius)
    circle2 : tuple of floats
        second circle parameters (x_pos, y_pos, radius)

    Returns
    -------
    float
        ratio between area of intersection and area of union

    """
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    d = math.hypot(x2 - x1, y2 - y1)

    area_intersection = cc_intersection(d, r1, r2)
    area_union = math.pi * (r1 * r1 + r2 * r2) - area_intersection

    return area_intersection / area_union


def cc_intersection(dist, rad1, rad2):
    """
    Area of intersection between two circles.

    Parameters
    ----------
    dist : positive float
        distance between circle centers
    rad1 : positive float
        radius of first circle
    rad2 : positive float
        radius of second circle

    Returns
    -------
    intersection_area : positive float
        area of intersection between circles

    References
    ----------
    http://mathworld.wolfram.com/Circle-CircleIntersection.html

    """
    if dist < 0:
        raise ValueError("Distance between circles must be positive")
    if rad1 < 0 or rad2 < 0:
        raise ValueError("Circle radius must be positive")

    if dist == 0 or (dist <= abs(rad2 - rad1)):
        return min(rad1, rad2) ** 2 * math.pi

    if dist > rad1 + rad2 or rad1 == 0 or rad2 == 0:
        return 0

    rad1_sq = rad1 * rad1
    rad2_sq = rad2 * rad2

    circle1 = rad1_sq * math.acos((dist * dist + rad1_sq - rad2_sq) /
                                  (2 * dist * rad1))
    circle2 = rad2_sq * math.acos((dist * dist + rad2_sq - rad1_sq) /
                                  (2 * dist * rad2))
    intersec = 0.5 * math.sqrt((-dist + rad1 + rad2) * (dist + rad1 - rad2) *
                               (dist - rad1 + rad2) * (dist + rad1 + rad2))
    intersection_area = circle1 + circle2 + intersec

    return intersection_area


def _filter_y_pred(y_pred, conf_threshold):
    return [[detected_object[1:] for detected_object in y_pred_patch
             if detected_object[0] > conf_threshold]
            for y_pred_patch in y_pred]


def mask_detection_curve(y_true, y_pred, conf_thresholds):
    """
    Calculate mask detection score for different confidence thresholds.

    Parameters
    ----------
    y_true : list of list of tuples
        Tuples are of form (x, y, radius).
    y_pred : list of list of tuples
        Tuples are of form (x, y, radius, confidence).
    conf_thresholds : array-like
        The confidence threshold for which to calculate the
        precision and recall.

    Returns
    -------
    ms : array with score for each confidence threshold

    """
    from .scp import scp_single

    ms = []

    for conf_threshold in conf_thresholds:
        y_pred_above_confidence = _filter_y_pred(y_pred, conf_threshold)
        ms.append(scp_single(y_true, y_pred_above_confidence))

    return np.array(ms)


def ospa_curve(y_true, y_pred, conf_thresholds):
    """
    Calculate OSPA score for different confidence thresholds.

    Parameters
    ----------
    y_true : list of list of tuples
        Tuples are of form (x, y, radius).
    y_pred : list of list of tuples
        Tuples are of form (x, y, radius, confidence).
    conf_thresholds : array-like
        The confidence threshold for which to calculate the
        precision and recall.

    Returns
    -------
    os : array with OSPA score for each confidence threshold

    """
    from .ospa import ospa

    os = []

    for conf_threshold in conf_thresholds:
        y_pred_above_confidence = _filter_y_pred(y_pred, conf_threshold)
        os.append(ospa(y_true, y_pred_above_confidence))

    return np.array(os)
