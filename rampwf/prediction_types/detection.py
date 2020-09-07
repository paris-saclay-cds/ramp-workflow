"""Multiclass predictions.

``y_pred`` should be two dimensional (n_samples x n_classes).

"""
import itertools

import numpy as np
from scipy import sparse

from .base import BasePrediction
from ..score_types.detection.iou import cc_iou
from ..score_types.detection.util import _match_tuples


class Predictions(BasePrediction):

    iou_threshold = 0.5

    def __init__(self, y_pred=None, y_true=None, n_samples=None,
                 fold_is=None):
        if y_pred is not None:
            if fold_is is not None:
                y_pred = y_pred[fold_is]
            self.y_pred = np.array(y_pred)
        elif y_true is not None:
            if fold_is is not None:
                y_true = y_true[fold_is]
            self.y_pred = np.array(y_true)
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
    def combine(cls, predictions_list, index_list=None, greedy=False):

        if index_list is None:  # we combine the full list
            index_list = range(len(predictions_list))
        y_comb_list = [predictions_list[i].y_pred for i in index_list]

        n_preds = len(y_comb_list[0])

        y_preds_combined = np.empty(n_preds, dtype=object)

        for i in range(n_preds):

            preds_list = [preds[i] for preds in y_comb_list
                          if preds[i] is not None]

            if len(preds_list) == 1:
                # no overlap in the different prediction sets -> simply take
                # the single one that is not None
                preds_combined = preds_list[0]
            else:
                preds_combined, _ = combine_predictions(
                    preds_list, cls.iou_threshold, greedy=greedy)

            y_preds_combined[i] = preds_combined

        combined_predictions = cls(y_pred=y_preds_combined)
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


def create_adjacency_matrix_from_edge_list(nodes, matches):
    # code based on nx.to_scipy_sparse_matrix

    nlen = len(nodes)
    index = dict(zip(nodes, range(nlen)))

    row, col, data = zip(*((index[u], index[v], 1)
                         for u, v in matches
                         if u in index and v in index))

    # symmetrize matrix
    d = data + data
    r = row + col
    c = col + row

    M = sparse.coo_matrix((d, (r, c)), shape=(nlen, nlen), dtype='int8')

    return M


def get_connected_components(nodes, matrix):
    from scipy.sparse.csgraph import connected_components
    ncon, labels = connected_components(matrix, directed=False)
    a_nodes = np.empty(len(nodes), dtype=object)
    a_nodes[:] = nodes
    return [a_nodes[labels == i] for i in range(ncon)]


def combine_predictions(preds_list, iou_threshold, greedy=False):
    """
    Combine multiple sets of predictions (of different models)
    for a single patch.


    """
    n_preds = len(preds_list)

    # for all pairs of the predictions: combine its detections and keep
    # those combinations that have a IOU > threshold
    matches = []
    matches_combined = []

    for mod1, mod2 in itertools.combinations(range(len(preds_list)), 2):

        pred1 = preds_list[mod1]
        pred2 = preds_list[mod2]

        idx1, idx2, ious = _match_tuples(
            [(x, y, r) for (c, x, y, r) in pred1],
            [(x, y, r) for (c, x, y, r) in pred2])

        idx1 = idx1[ious > iou_threshold]
        idx2 = idx2[ious > iou_threshold]

        # add each matching combination to list as (model_idx, pred_idx)
        # where:
        # model_idx is the index into preds_list
        # pred_idx is the index into preds_list[model_idx]
        for i1, i2 in zip(idx1, idx2):
            comb = (np.asarray(pred1[i1]) + np.array(pred2[i2])) / 2
            matches.append(((mod1, i1), (mod2, i2)))
            matches_combined.append(comb)

    if greedy:
        combined = _greedy_nms(matches_combined)
        return combined, matches

    # if matches is empty, return directly
    if not matches:
        return [], matches

    # the matches (set of (model_idx, pred_idx) tuples) are the nodes of
    # our graph:
    # - TODO: remove redundant edges/nodes
    # - convert to adjacency matrix
    # - get connected components of matrix
    nodes = sorted(set([x for y in matches for x in y]))
    M = create_adjacency_matrix_from_edge_list(nodes, matches)
    match_groups = get_connected_components(nodes, M)

    preds_combined = []

    # each connected set of detections is combined into a single (c,x,y,r)
    for group in match_groups:

        # convert list of (model_idx, pred_idx) tuples into actual
        # predictions
        preds = []
        for mod, idx in group:
            preds.append(preds_list[mod][idx])

        preds = np.array(preds)

        # combine in a single (c, x, y, r)
        if preds[:, 0].sum() == 0:
            # corner case of all confidence being zero: just taking mean
            # because weighted average will fail
            pred_combined = np.mean(preds, axis=0)
        else:
            # (x, y, r) are averaged weighted with the confidence
            pred_combined = np.average(preds[:, 1:], weights=preds[:, 0],
                                       axis=0)

            # the confidence is averaged taking into account missing
            # predictions
            conf = preds[:, 0].sum() / n_preds
            pred_combined = np.insert(pred_combined, 0, conf)

        preds_combined.append(pred_combined)

    return preds_combined, matches
