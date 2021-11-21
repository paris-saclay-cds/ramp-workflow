import numpy as np
from scipy.optimize import linear_sum_assignment

from .iou import cc_iou


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


def _filter_y_pred(y_pred, conf_threshold):
    """
    Given a list of list of predicted craters return those
    with a confidence value above given threshold

    Parameters
    ----------
    y_pred : list of list of tuples
    conf_threshold : float

    Returns
    -------
    y_pred_filtered : list of list of tuples

    """
    return [[detected_object[1:]
             for detected_object in y_pred_patch
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
        Tuples are of form (confidence, x, y, radius).
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
        Tuples are of form (confidence, x, y, radius).
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
