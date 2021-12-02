import numpy as np

from .base import BaseScoreType
from .util import _filter_y_pred
from .iou import cc_iou
from .precision_recall import precision, recall


class DetectionAveragePrecision(BaseScoreType):
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
        _, ps, rs = precision_recall_curve_greedy(
            y_true, y_pred, iou_threshold=self.iou_threshold)
        return average_precision_exact(ps, rs)


AverageDetectionPrecision = DetectionAveragePrecision


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


def _add_id(y):
    """
    Helper function to flatten and add id column to list of lists.

    Since the list of lists is flattened into a single array, empty lists
    do not result in an entry in the final array.

    """
    y_new = []

    for i, y_patch in enumerate(y):
        if len(y_patch):
            tmp = np.asarray(y_patch)
            tmp = np.insert(tmp, 0, i, axis=1)
            y_new.append(tmp)

    if y_new:
        return np.vstack(y_new)
    else:
        return np.array([[]])


def precision_recall_curve_greedy(y_true, y_pred, iou_threshold=0.5):
    """
    Calculate precision and recall incrementally based on the predictions
    sorted by confidence (not calculating an exact optimal match for each
    confidence level).

    Parameters
    ----------
    y_true : list of lists of (x, y, r) tuples
    y_pred : list of lists of (conf, x, y, r) tuples
    iou_threshold : float [0 - 1], default 0.5

    Returns
    -------
    Three arrays:

    confidence_values
        The flattened and sorted confidence values of y_pred
    precision
        The corresponding precision values
    recall
        The corresponding recall values

    """
    # flatten y_pred into single array and add column with img id
    y_pred2 = _add_id(y_pred)
    if not y_pred2.size:
        return np.array([]), np.array([]), np.array([])

    # Sorting predicted objects by decreasing confidence
    y_pred2_sorted = y_pred2[np.argsort(y_pred2[:, 1])[::-1], :]

    # array to store whether a match is observed or not for each prediction
    res = np.empty(len(y_pred2_sorted), dtype='bool')

    # object to keep track of matches: (img id, object index) pairs
    matched = set([])
    confs = []

    for i, pred in enumerate(y_pred2_sorted):
        patch_id, conf, row, col, rad = pred

        y_patch = y_true[int(patch_id)]
        n_true = len(y_patch)

        if n_true > 0:
            ious = np.empty(n_true)

            for j in range(n_true):
                ious[j] = cc_iou(y_patch[j], (row, col, rad))

            i_max = np.argmax(ious)
            if ((ious[i_max] > iou_threshold)
                    and (patch_id, i_max) not in matched):
                res[i] = True
                # add match identifier to set to later check
                # we don't have a duplicate match
                matched.add((patch_id, i_max))
            else:
                res[i] = False
        else:
            res[i] = False

        confs.append(conf)

    n_true_total = np.sum([len(x) for x in y_true])

    recall = np.cumsum(res) / n_true_total
    precision = np.cumsum(res) / np.arange(1, len(res) + 1)

    return np.array(confs), precision, recall


def average_precision_interpolated(ps, rs):
    """
    The Average Precision (AP) score.

    Calculation based on the 11-point interpolation of the precision-recall
    curve (method described for Pascal VOC challenge,
    http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf).

    TODO: they changed this in later:
    http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf

    https://stackoverflow.com/questions/36274638/map-metric-in-object-detection-and-computer-vision  # noqa

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


def average_precision_exact(ps, rs):
    # from https://github.com/amdegroot/ssd.pytorch/blob/ce4c994db0ee11f82aabb4fdb3499dc970156db5/eval.py#L182-L213  # noqa

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rs, [1.]))
    mpre = np.concatenate(([0.], ps, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
