"""Multiclass predictions.

``y_pred`` should be two dimensional (n_samples x n_classes).

"""
import itertools

import numpy as np

from .base import BasePrediction
from ..score_types.detection.iou import cc_iou
from ..score_types.detection.util import _match_tuples


class Predictions(BasePrediction):

    iou_threshold = 0.5

    def __init__(self, y_pred=None, y_true=None, n_samples=None):
        if y_pred is not None:
            self.y_pred = y_pred
        elif y_true is not None:
            self.y_pred = y_true
        elif n_samples is not None:
            self.y_pred = np.empty(n_samples, dtype=object)
        else:
            raise ValueError(
                'Missing init argument: y_pred, y_true, or n_samples')
        self.check_y_pred_dimensions()

    def check_y_pred_dimensions(self):
        # XXX should check that prediction is an array of lists or Nones.
        pass

    @classmethod
    def combine(cls, predictions_list, index_list=None):

        if index_list is None:  # we combine the full list
            index_list = range(len(predictions_list))
        y_comb_list = [predictions_list[i].y_pred for i in index_list]

        matches = []

        for a, b in itertools.combinations(y_comb_list, 2):
            idx1, idx2, ious = _match_tuples([(x, y, r) for (c, x, y, r) in a],
                                             [(x, y, r) for (c, x, y, r) in b])

            idx1 = idx1[ious > cls.iou_threshold]
            idx2 = idx2[ious > cls.iou_threshold]

            for i1, i2 in zip(idx1, idx2):
                combined = (np.asarray(a[i1]) + np.array(b[i2])) / 2
                matches.append(combined)

        combined = _greedy_nms(matches)

        combined_predictions = cls(y_pred=combined)
        return combined_predictions

    def set_valid_in_train(self, predictions, test_is):
        """Set a cross-validation slice."""
        self.y_pred[test_is] = predictions.y_pred

    @property
    def valid_indexes(self):
        """Return valid indices (e.g., a cross-validation slice)."""
        return self.y_pred != np.empty(len(self.y_pred), dtype=np.object)


def make_detection():
    return Predictions


def _greedy_nms(y_pred, iou_threshold=0.45):
    y_pred = np.asarray(y_pred)

    boxes_left = np.copy(y_pred)
    # This is where we store the boxes that make it through the
    # non-maximum suppression
    maxima = []

    # While there are still boxes left to compare...
    while boxes_left.shape[0] > 0:
        # ...get the index of the next box with the highest confidence...
        maximum_index = np.argmax(boxes_left[:, 0])
        # ...copy that box and...
        maximum_box = np.copy(boxes_left[maximum_index])
        # ...append it to `maxima` because we'll definitely keep it
        maxima.append(maximum_box)
        # Now remove the maximum box from `boxes_left`
        boxes_left = np.delete(boxes_left, maximum_index, axis=0)
        # If there are no boxes left after this step, break. Otherwise...
        if boxes_left.shape[0] == 0:
            break
        # ...compare (IoU) the other left over boxes to the maximum box...
        similarities = np.array([cc_iou(b[1:], maximum_box[1:]) for b in
                                 boxes_left])
        # ...so that we can remove the ones that overlap too much
        # with the maximum box
        boxes_left = boxes_left[similarities <= iou_threshold]

    return np.array(maxima)
