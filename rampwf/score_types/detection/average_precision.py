from __future__ import division

import numpy as np

from .base import BaseScoreType
from .util import _filter_y_pred
from .precision_recall import precision, recall


class AverageDetectionPrecision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
    _name = 'average_precision'

    def __init__(self, name=_name, precision=3, iou_threshold=0.5):
        if name is None:
            self.name = '{name}(IOU={iou_threshold})'.format(
                name=self._name, iou_threshold=iou_threshold)
        else:
            self.name = name
        self.precision = precision
        self.iou_threshold = iou_threshold

    def __call__(self, y_true, y_pred):
        y_pred_conf = [detected_object[0]
                       for single_detection in y_pred
                       for detected_object in single_detection]
        min_conf, max_conf = np.min(y_pred_conf), np.max(y_pred_conf)
        conf_thresholds = np.linspace(min_conf, max_conf, 20)
        ps, rs = precision_recall_curve(y_true, y_pred, conf_thresholds,
                                        iou_threshold=self.iou_threshold)
        return average_precision_interpolated(ps, rs)


def precision_recall_curve(y_true, y_pred, conf_thresholds, iou_threshold=0.5):
    """
    Calculate precision and recall for different confidence thresholds.

    Parameters
    ----------
    y_true : list of list of tuples
        Tuples are of form (x, y, radius).
    y_pred : list of list of tuples
        Tuples are of form (x, y, radius, confidence).
    conf_thresholds : array-like
        The confidence threshold for which to calculate the
        precision and recall.
    iou_threshold : float
        Threshold to determine match.

    Returns
    -------
    ps, rs : arrays with precisions, recalls for each confidence threshold

    """
    ps = []
    rs = []

    for conf_threshold in conf_thresholds:
        y_pred_above_confidence = _filter_y_pred(y_pred, conf_threshold)
        ps.append(precision(
            y_true, y_pred_above_confidence, iou_threshold=iou_threshold))
        rs.append(recall(
            y_true, y_pred_above_confidence, iou_threshold=iou_threshold))

    return np.array(ps), np.array(rs)


def average_precision_interpolated(ps, rs):
    """
    The Average Precision (AP) score.

    Calculation based on the 11-point interpolation of the precision-recall
    curve (method described for Pascal VOC challenge,
    http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf).

    TODO: they changed this in later:
    http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf

    https://stackoverflow.com/questions/36274638/map-metric-in-object-detection-and-computer-vision

    Parameters
    ----------
    ps, rs : arrays of same length with corresponding precision / recall scores

    Returns
    -------
    ap : int [0 - 1]
        Average precision score

    """
    ps = np.asarray(ps)
    rs = np.asarray(rs)

    p_at_r = []

    for r in np.arange(0, 1.1, 0.1):
        p = np.array(ps)[np.array(rs) >= r]
        if p.size:
            p_at_r.append(np.nanmax(p))
        else:
            p_at_r.append(0)

    ap = np.mean(p_at_r)
    return ap
