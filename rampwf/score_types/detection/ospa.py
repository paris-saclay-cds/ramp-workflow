import numpy as np
from sklearn.utils import indices_to_mask

from .base import DetectionBaseScoreType
from .util import _select_minipatch_tuples, _match_tuples


class OSPA(DetectionBaseScoreType):
    """
    Optimal Subpattern Assignment (OSPA) metric for IoU score.

    This metric provides a coherent way to compute the miss-distance
    between the detection and alignment of objects. Among all
    combinations of true/predicted pairs, if finds the best alignment
    to minimise the distance, and still takes into account missing
    or in-excess predicted values through a cardinality score.

    The lower the value the smaller the distance.

    Arguments
    ---------
    name : str, optional
        Method name
    precision : int, optional
        Rounding precision for the score (default is 2)
    conf_threshold : float, optional
        Confidence threshold value use for the Average Precision
        measurement (default is 0.5)
    minipatch : [row_min, row_max, col_min, col_max], optional
        Bounds of the internal scoring patch (default is None)

    References
    ----------
    http://www.dominic.schuhmacher.name/papers/ospa.pdf

    """
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='ospa', precision=3, conf_threshold=0.5,
                 minipatch=None):
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch

    def detection_score(self, y_true, y_pred):
        """
        Compute the OSPA score

        Parameters
        ----------
        y_true, y_pred : list of list of tuples

        Returns
        -------
        float: distance between input arrays

        """
        ospa_score = ospa(y_true, y_pred, self.minipatch)
        return 1 - ospa_score


def ospa(y_true, y_pred, minipatch=None):
    """
    Optimal Subpattern Assignment (OSPA) metric for IoU score.

    This metric provides a coherent way to compute the miss-distance
    between the detection and alignment of objects. Among all
    combinations of true/predicted pairs, if finds the best alignment
    to minimise the distance, and still takes into account missing
    or in-excess predicted values through a cardinality score.

    The lower the value the smaller the distance.

    Parameters
    ----------
    y_true, y_pred : list of list of tuples
    minipatch : [row_min, row_max, col_min, col_max], optional
        Bounds of the internal scoring patch (default is None)

    Returns
    -------
    float: distance between input arrays

    References
    ----------
    http://www.dominic.schuhmacher.name/papers/ospa.pdf

    """
    ospas = np.array([ospa_single(t, p, minipatch)
                      for t, p in zip(y_true, y_pred)])

    ious, n_count, n_total = ospas.sum(axis=0)

    if n_total == 0:
        return 0

    return ious / n_total


def ospa_single(y_true, y_pred, minipatch=None):
    """
    OSPA score on single patch. See docstring of `ospa` for more info.

    Parameters
    ----------
    y_true, y_pred : ndarray of shape (3, x)
        arrays of (x, y, radius)
    minipatch : [row_min, row_max, col_min, col_max], optional
        Bounds of the internal scoring region (default is None)

    Returns
    -------
    (iou_sum, n_pred, n_total)
        float - sum of ious of matched entries
        int - number of matched entries
        int - total number of entries

    """
    n_true = len(y_true)
    n_pred = len(y_pred)

    # No craters and none found
    if n_true == 0 and n_pred == 0:
        return 0, 0, 0

    # Mask of entries that lie within the minipatch
    if minipatch is not None:
        true_in_minipatch = _select_minipatch_tuples(y_true, minipatch)
        pred_in_minipatch = _select_minipatch_tuples(y_pred, minipatch)
    else:
        true_in_minipatch = np.ones(len(y_true)).astype(bool)
        pred_in_minipatch = np.ones(len(y_pred)).astype(bool)

    n_minipatch = true_in_minipatch.sum() + pred_in_minipatch.sum()

    # No craters and some found or existing craters but non found
    if n_true == 0 or n_pred == 0:
        return 0, 0, n_minipatch

    # First matching
    id_true, id_pred, ious = _match_tuples(y_true, y_pred)

    # For each set of entries (true and pred) create an array with
    # the iou corresponding to each object
    iou_true = np.zeros(len(y_true))
    iou_true[id_true] = ious
    iou_pred = np.zeros(len(y_pred))
    iou_pred[id_pred] = ious

    # Mask of matched entries
    true_matched = indices_to_mask(id_true, n_true)
    pred_matched = indices_to_mask(id_pred, n_pred)

    # Counting
    true_count = true_matched & true_in_minipatch
    pred_count = pred_matched & pred_in_minipatch

    # IoU computation on the final list
    iou_global = iou_true[true_count].sum() + iou_pred[pred_count].sum()
    n_count = true_count.sum() + pred_count.sum()

    return iou_global, n_count, n_minipatch
