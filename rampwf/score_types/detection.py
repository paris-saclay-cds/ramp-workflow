from __future__ import division

import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.utils import indices_to_mask
from .base import BaseScoreType


class DetectionBaseScoreType(BaseScoreType):
    """Common abstract base type for detection scores.

    Implements `__call__` by selecting all prediction detections with
    a confidence higher than `conf_threshold`. It assumes that the child
    class implements `detection_score`.
    """

    conf_threshold = 0.5

    def __call__(self, y_true, y_pred, conf_threshold=None):
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        y_pred_above_confidence = [
            [detected_object[1:] for detected_object in single_detection
             if detected_object[0] > conf_threshold]
            for single_detection in y_pred]
        return self.detection_score(y_true, y_pred_above_confidence)


class SCP(DetectionBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, shape, name='scp', precision=2, conf_threshold=0.5,
                 minipatch=None):
        self.shape = shape
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch

    def detection_score(self, y_true, y_pred):
        """
        Score based on a matching by reprojection of craters on mask-map.

        True craters are projected positively, predicted craters negatively,
        so they can cancel out. Then the sum of the absolute value of the
        residual map is taken.

        The best score value for a perfect match is 0.
        The worst score value for a given patch is the sum of all crater
        instances in both `y_true` and `y_pred`.

        Parameters
        ----------
        y_true : list of list of tuples (x, y, radius)
            List of coordinates and radius of actual craters for set of patches
        y_pred : list of list of tuples (x, y, radius)
            List of coordinates and radius of predicted craters for set of
            patches

        Returns
        -------
        float : score for a given patch, the lower the better

        """
        scps = np.array(
            [scp_single(t, p, self.shape, self.minipatch)
             for t, p in zip(y_true, y_pred)])
        return np.sum(scps[:, 0]) / np.sum(scps[:, 1:3])


class OSPA(DetectionBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='ospa', precision=2, conf_threshold=0.5,
                 minipatch=None):
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch

    def detection_score(self, y_true, y_pred):
        ospa_score = ospa(y_true, y_pred, self.minipatch)
        return 1 - ospa_score


class AverageDetectionPrecision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='average_precision', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        conf_thresholds = np.linspace(0.0, 1, 50)
        ps, rs = precision_recall_curve(y_true, y_pred, conf_thresholds)
        return average_precision_interpolated(ps, rs)


class DetectionPrecision(DetectionBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='precision', precision=2, conf_threshold=0.5,
                 minipatch=None):
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch

    def detection_score(self, y_true, y_pred):
        return precision(y_true, y_pred, minipatch=self.minipatch)


class DetectionRecall(DetectionBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='recall', precision=2, conf_threshold=0.5,
                 minipatch=None):
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch

    def detection_score(self, y_true, y_pred):
        return recall(y_true, y_pred, minipatch=self.minipatch)


class MADCenter(DetectionBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='mad_center', precision=2, conf_threshold=0.5,
                 minipatch=None):
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch

    def detection_score(self, y_true, y_pred):
        return mad_center(y_true, y_pred, minipatch=self.minipatch)


class MADRadius(DetectionBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='mad_radius', precision=2, conf_threshold=0.5,
                 minipatch=None):
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch

    def detection_score(self, y_true, y_pred):
        return mad_radius(y_true, y_pred, minipatch=self.minipatch)


def scp_single(y_true, y_pred, shape, minipatch=None):
    """
    L1 distance between superposing bounding box cylinder or prism maps.

    True craters are projected positively, predicted craters negatively,
    so they can cancel out. Then the sum of the absolute value of the
    residual map is taken.

    The best score value for a perfect match is 0.
    The worst score value for a given patch is the sum of all crater
    instances in both `y_true` and `y_pred`.

    Parameters
    ----------
    y_true : list of tuples (x, y, radius)
        List of coordinates and radius of actual craters in a patch
    y_pred : list of tuples (x, y, radius)
        List of coordinates and radius of craters predicted in the patch
    shape : tuple of int
        Shape of the main patch
    minipatch : list of int, optional
        Bounds of the internal scoring patch (default is None)

    Returns
    -------
    float : score for a given patch, the lower the better

    """
    map_true, map_pred = circle_maps(y_true, y_pred, shape)
    if minipatch is not None:
        map_true = map_true[
            minipatch[0]:minipatch[1], minipatch[2]:minipatch[3]]
        map_pred = map_pred[
            minipatch[0]:minipatch[1], minipatch[2]:minipatch[3]]
    # Sum all the pixels
    score = np.abs(map_true - map_pred).sum()
    n_true = map_true.sum()
    n_pred = map_pred.sum()
    return score, n_true, n_pred


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
    minipatch : list of int, optional
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

    return ious * n_count / n_total


def ospa_single(y_true, y_pred, minipatch=None):
    """
    OSPA score on single patch. See docstring of `ospa` for more info.

    Parameters
    ----------
    y_true, y_pred : ndarray of shape (3, x)
        arrays of (x, y, radius)
    minipatch : list of int, optional
        Bounds of the internal scoring patch (default is None)

    Returns
    -------
    (iou_sum, n_pred, n_total)
        float - sum of ious of matched entries
        int - sum of matched entries
        int - sum of all entries

    """
    n_true = len(y_true)
    n_pred = len(y_pred)
    n_total = n_true + n_pred

    # No craters and none found
    if n_true == 0 and n_pred == 0:
        return 2, 2, 2

    # No craters and some found or existing craters but non found
    if n_true == 0 or n_pred == 0:
        return 0, 0, n_total

    # First matching
    id_true, id_pred, ious = _match_tuples(y_true, y_pred)

    # Mask of matched entries
    true_matched = indices_to_mask(id_true, n_true)
    pred_matched = indices_to_mask(id_pred, n_pred)

    # Mask of entries that lie within the minipatch
    if minipatch is not None:
        true_in_minipatch = _select_minipatch_tuples(y_true, minipatch)
        pred_in_minipatch = _select_minipatch_tuples(y_pred, minipatch)
    else:
        true_in_minipatch = np.ones_like(y_true).astype(bool)
        pred_in_minipatch = np.ones_like(y_true).astype(bool)

    # Counting
    true_count = true_matched & true_in_minipatch
    pred_count = pred_matched & pred_in_minipatch

    # IoU computation on the final list
    iou_global = ious[true_count].sum() + ious[pred_count].sum()
    n_count = true_count.sum() + pred_count.sum()

    return iou_global, n_count, n_total


def _select_minipatch_tuples(y_list, minipatch):
    """
    Mask over a list selecting the tuples that lie in the minipatch

    Parameters
    ----------
    y_list : list of tuples
        Full list of labels and predictions
    minipatch : list of int
        Bounds of the internal scoring patch

    Returns
    -------
    y_list_cut : list of bool
        List of booleans corresponding to whether the circle is in
        the minipatch or not

    """
    row_min, row_max, col_min, col_max = minipatch

    y_list = np.asarray(y_list)

    y_list_cut = ((y_list[0] >= col_min) & (y_list[0] < col_max) &
                  (y_list[1] >= row_min) & (y_list[1] < row_max))

    return y_list_cut


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
        p = (ious >= iou_threshold).sum()

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

    Returns
    -------
    precision_score : float [0 - 1]
    """
    if matches is None:
        matches = [_match_tuples(t, p, minipatch=minipatch)
                   for t, p in zip(y_true, y_pred)]

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

    Returns
    -------
    recall_score : float [0 - 1]
    """
    if matches is None:
        matches = [_match_tuples(t, p, minipatch=minipatch)
                   for t, p in zip(y_true, y_pred)]

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

    Returns
    -------
    mad_radius : float > 0
    """
    if matches is None:
        matches = [_match_tuples(t, p, minipatch=minipatch)
                   for t, p in zip(y_true, y_pred)]

    loc_true, loc_pred = _locate_matches(
        y_true, y_pred, matches, iou_threshold=iou_threshold)

    return np.abs((loc_pred[:, 2] - loc_true[:, 2]) / loc_true[:, 2]).mean()


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

    Returns
    -------
    mad_center : float > 0
    """
    if matches is None:
        matches = [_match_tuples(t, p, minipatch=minipatch)
                   for t, p in zip(y_true, y_pred)]

    loc_true, loc_pred = _locate_matches(
        y_true, y_pred, matches, iou_threshold=iou_threshold)

    d = np.sqrt((loc_pred[:, 0] - loc_true[:, 0]) ** 2 + (
        loc_pred[:, 1] - loc_true[:, 1]) ** 2)

    return np.abs(d / loc_true[:, 2]).mean()


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
    from .mask import mask_detection

    ms = []

    for conf_threshold in conf_thresholds:
        y_pred_above_confidence = _filter_y_pred(y_pred, conf_threshold)
        ms.append(mask_detection(y_true, y_pred_above_confidence))

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


def project_circle(circle, image=None, shape=None,
                   normalize=True, negative=False):
    """
    Project circles on an image.

    Parameters
    ----------
    circle : array-like
        x, y, radius
    image : array-like, optional
        image on which to project the circle
    shape : tuple of ints, optional
        shape of the image on which to project the circle
    normalize : bool, optional (default is `True`)
        normalize the total surface of the circle to unity
    negative : bool, optional (default is `False`)
        subtract the circle instead of adding it

    Returns
    -------
    array-like : image with projected circle

    """
    if image is None:
        if shape is None:
            raise ValueError("Either `image` or `shape` must be defined")
        else:
            image = np.zeros(shape)
    else:
        shape = image.shape
    x, y, radius = circle
    coords = circle_coords(x, y, radius, shape=shape)
    value = 1
    if normalize:
        value /= coords[0].size
    if negative:
        value = -value
    image[coords] += value
    return image


def circle_maps(y_true, y_pred, shape):
    """
    Create a map to compare true and predicted craters.

    The craters (circles) are projected on the map with a coefficient
    chosen so its sum is normalized to unity.

    True and predicted craters are projected with a different sign,
    so that good predictions tend to cancel out the true craters.

    Parameters
    ----------
    y_pred, y_true : array-like of shape (3, X)
        list of circle positions (x, y, radius)
    shape : tuple of ints, optional
        shape of image

    Returns
    -------
    array-like : image with projected true and predicted circles

    """
    map_true = np.zeros(shape)
    map_pred = np.zeros(shape)

    # Add true craters positively
    for circle in y_true:
        map_true = project_circle(
            circle, map_true, shape=shape, normalize=True)

    # Add predicted craters negatively
    for circle in y_pred:
        map_pred = project_circle(
            circle, map_pred, shape=shape, normalize=True)

    return map_true, map_pred


#
# Copyright scikit-image below
#
def _ellipse_in_shape(shape, center, radii, rotation=0.):
    """Generate coordinates of points within ellipse bounded by shape.

    Parameters
    ----------
    shape :  iterable of ints
        Shape of the input image.  Must be length 2.
    center : iterable of floats
        (row, column) position of center inside the given shape.
    radii : iterable of floats
        Size of two half axes (for row and column)
    rotation : float, optional
        Rotation of the ellipse defined by the above, in radians
        in range (-PI, PI), in contra clockwise direction,
        with respect to the column-axis.
    Returns
    -------
    rows : iterable of ints
        Row coordinates representing values within the ellipse.
    cols : iterable of ints
        Corresponding column coordinates representing values within the ellipse
    """
    r_lim, c_lim = np.ogrid[0:float(shape[0]), 0:float(shape[1])]
    r_org, c_org = center
    r_rad, c_rad = radii
    rotation %= np.pi
    sin_alpha, cos_alpha = np.sin(rotation), np.cos(rotation)
    r, c = (r_lim - r_org), (c_lim - c_org)
    distances = ((r * cos_alpha + c * sin_alpha) / r_rad) ** 2 \
        + ((r * sin_alpha - c * cos_alpha) / c_rad) ** 2
    return np.nonzero(distances < 1)


def ellipse(r, c, r_radius, c_radius, shape=None, rotation=0.):
    """Generate coordinates of pixels within ellipse.

    Parameters
    ----------
    r, c : double
        Centre coordinate of ellipse.
    r_radius, c_radius : double
        Minor and major semi-axes. ``(r/r_radius)**2 + (c/c_radius)**2 = 1``.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for ellipses which exceed the
        image size.
        By default the full extent of the ellipse are used.
    rotation : float, optional (default 0.)
        Set the ellipse rotation (rotation) in range (-PI, PI)
        in contra clock wise direction, so PI/2 degree means swap ellipse axis
    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of ellipse.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    Examples
    --------
    >>> from skimage.draw import ellipse
    >>> img = np.zeros((10, 12), dtype=np.uint8)
    >>> rr, cc = ellipse(5, 6, 3, 5, rotation=np.deg2rad(30))
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    Notes
    -----
    The ellipse equation::
        ((x * cos(alpha) + y * sin(alpha)) / x_radius) ** 2 +
        ((x * sin(alpha) - y * cos(alpha)) / y_radius) ** 2 = 1
    Note that the positions of `ellipse` without specified `shape` can have
    also, negative values, as this is correct on the plane. On the other hand
    using these ellipse positions for an image afterwards may lead to appearing
    on the other side of image, because ``image[-1, -1] = image[end-1, end-1]``
    >>> rr, cc = ellipse(1, 2, 3, 6)
    >>> img = np.zeros((6, 12), dtype=np.uint8)
    >>> img[rr, cc] = 1
    >>> img
    array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]], dtype=uint8)
    """
    center = np.array([r, c])
    radii = np.array([r_radius, c_radius])
    # allow just rotation with in range +/- 180 degree
    rotation %= np.pi

    # compute rotated radii by given rotation
    r_radius_rot = abs(r_radius * np.cos(rotation)) \
        + c_radius * np.sin(rotation)
    c_radius_rot = r_radius * np.sin(rotation) \
        + abs(c_radius * np.cos(rotation))
    # The upper_left and lower_right corners of the smallest rectangle
    # containing the ellipse.
    radii_rot = np.array([r_radius_rot, c_radius_rot])
    upper_left = np.ceil(center - radii_rot).astype(int)
    lower_right = np.floor(center + radii_rot).astype(int)

    if shape is not None:
        # Constrain upper_left and lower_right by shape boundary.
        upper_left = np.maximum(upper_left, np.array([0, 0]))
        lower_right = np.minimum(lower_right, np.array(shape[:2]) - 1)

    shifted_center = center - upper_left
    bounding_shape = lower_right - upper_left + 1

    rr, cc = _ellipse_in_shape(bounding_shape, shifted_center, radii, rotation)
    rr.flags.writeable = True
    cc.flags.writeable = True
    rr += upper_left[0]
    cc += upper_left[1]
    return rr, cc


def circle_coords(r, c, radius, shape=None):
    """Generate coordinates of pixels within circle.

    Parameters
    ----------
    r, c : double
        Centre coordinate of circle.
    radius : double
        Radius of circle.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for circles that exceed the image
        size. If None, the full extent of the circle is used.
    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of circle.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    Examples
    --------
    >>> from skimage.draw import circle
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = circle(4, 4, 5)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    return ellipse(r, c, radius, radius, shape)
