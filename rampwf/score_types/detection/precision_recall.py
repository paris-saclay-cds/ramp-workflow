"""
Precision and recall (and location and size precision) based on a best
match (given a specific confidence threshold).

"""
import numpy as np

from .base import DetectionBaseIOUScoreType
from .util import (
    _match_tuples, _count_matches, _locate_matches, _filter_minipatch_tuples)


class DetectionPrecision(DetectionBaseIOUScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
    _name = 'precision'

    def detection_score(self, y_true, y_pred):
        return precision(y_true, y_pred, minipatch=self.minipatch,
                         iou_threshold=self.iou_threshold)


class DetectionRecall(DetectionBaseIOUScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
    _name = 'recall'

    def detection_score(self, y_true, y_pred):
        return recall(y_true, y_pred, minipatch=self.minipatch,
                      iou_threshold=self.iou_threshold)


class MADCenter(DetectionBaseIOUScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf
    _name = 'mad_center'

    def detection_score(self, y_true, y_pred):
        return mad_center(y_true, y_pred, minipatch=self.minipatch,
                          iou_threshold=self.iou_threshold)


class MADRadius(DetectionBaseIOUScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf
    _name = 'mad_radius'

    def detection_score(self, y_true, y_pred):
        return mad_radius(y_true, y_pred, minipatch=self.minipatch,
                          iou_threshold=self.iou_threshold)


def precision(y_true, y_pred, matches=None, iou_threshold=0.5,
              minipatch=None):
    """
    Precision score (fraction of correct predictions).

    Parameters
    ----------
    y_true, y_pred : list of list of tuples
    matches : optional, output of _match_tuples
    iou_threshold : float
        Threshold to determine match
    minipatch : [row_min, row_max, col_min, col_max], optional
        Bounds of the internal scoring patch (default is None)

    Returns
    -------
    precision_score : float [0 - 1]

    """
    # precision is calculated relative to the number of predicted objects,
    # so we filter for the scoring region based on the predictions
    if minipatch is not None:
        y_pred = [_filter_minipatch_tuples(y_patch, minipatch)
                  for y_patch in y_pred]

    if matches is None:
        matches = [_match_tuples(yp_true, yp_pred)
                   for yp_true, yp_pred in zip(y_true, y_pred)]

    n_true, n_pred_all, n_pred_correct = _count_matches(
        y_true, y_pred, matches, iou_threshold=iou_threshold)

    return n_pred_correct / n_pred_all


def recall(y_true, y_pred, matches=None, iou_threshold=0.5,
           minipatch=None):
    """
    Recall score (fraction of true objects that are predicted).

    Parameters
    ----------
    y_true, y_pred : list of list of tuples
    matches : optional, output of _match_tuples
    iou_threshold : float
        Threshold to determine match
    minipatch : [row_min, row_max, col_min, col_max], optional
        Bounds of the internal scoring patch (default is None)

    Returns
    -------
    recall_score : float [0 - 1]

    """
    # recall is calculated relative to the number of true objects,
    # so we filter for the scoring region based on the true labels
    if minipatch is not None:
        y_true = [_filter_minipatch_tuples(y_patch, minipatch)
                  for y_patch in y_true]

    if matches is None:
        matches = [_match_tuples(yp_true, yp_pred)
                   for yp_true, yp_pred in zip(y_true, y_pred)]

    n_true, n_pred_all, n_pred_correct = _count_matches(
        y_true, y_pred, matches, iou_threshold=iou_threshold)

    return n_pred_correct / n_true


def mad_radius(y_true, y_pred, matches=None, iou_threshold=0.5,
               minipatch=None):
    """
    Relative Mean absolute deviation (MAD) of the radius.

    Parameters
    ----------
    y_true, y_pred : list of list of tuples
    matches : optional, output of _match_tuples
    iou_threshold : float
        Threshold to determine match
    minipatch : [row_min, row_max, col_min, col_max], optional
        Bounds of the internal scoring patch (default is None)

    Returns
    -------
    mad_radius : float > 0
    """
    if minipatch is not None:
        y_true = [_filter_minipatch_tuples(y_patch, minipatch)
                  for y_patch in y_true]

    if matches is None:
        matches = [_match_tuples(yp_true, yp_pred)
                   for yp_true, yp_pred in zip(y_true, y_pred)]

    loc_true, loc_pred = _locate_matches(
        y_true, y_pred, matches, iou_threshold=iou_threshold)

    _, _, rad_true = loc_true.T
    _, _, rad_pred = loc_pred.T

    if len(rad_true) == 0:
        return np.nan

    return np.abs((rad_pred - rad_true) / rad_true).mean()


def mad_center(y_true, y_pred, matches=None, iou_threshold=0.5,
               minipatch=None):
    """
    Relative Mean absolute deviation (MAD) of the center.

    (relative to the radius of the true object).

    Parameters
    ----------
    y_true, y_pred : list of list of tuples
    matches : optional, output of _match_tuples
    iou_threshold : float
        Threshold to determine match
    minipatch : [row_min, row_max, col_min, col_max], optional
        Bounds of the internal scoring patch (default is None)

    Returns
    -------
    mad_center : float > 0
    """
    if minipatch is not None:
        y_true = [_filter_minipatch_tuples(y_patch, minipatch)
                  for y_patch in y_true]

    if matches is None:
        matches = [_match_tuples(yp_true, yp_pred)
                   for yp_true, yp_pred in zip(y_true, y_pred)]

    loc_true, loc_pred = _locate_matches(
        y_true, y_pred, matches, iou_threshold=iou_threshold)

    x_true, y_true, rad_true = loc_true.T
    x_pred, y_pred, _ = loc_pred.T

    if len(x_true) == 0:
        return np.nan

    d = np.sqrt((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2)

    return np.abs(d / rad_true).mean()
