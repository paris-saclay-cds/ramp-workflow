import numpy as np
from databoard.base_prediction import BasePrediction


class Predictions(BasePrediction):

    def __init__(self, labels=None, y_pred=None, y_pred_labels=None,
                 y_pred_indexes=None, y_true=None, n_samples=None):
        self.labels = labels
        if y_pred is not None:
            self.y_proba = np.array(y_pred)
        elif y_pred_labels is not None:
            self._init_from_pred_labels(y_pred_labels)
        elif y_true is not None:
            self._init_from_pred_labels(y_true)
        elif n_samples is not None:
            self.y_proba = np.empty(
                (n_samples, len(self.labels)), dtype=np.float64)
            self.y_proba.fill(np.nan)
        else:
            raise ValueError('Missing init argument: y_pred, y_pred_labels, '
                             'y_pred_indexes, y_true, f_name, or n_samples)')
        shape = self.y_proba.shape
        if len(shape) != 2:
            raise ValueError('Multiclass y_proba should be 2-dimensional, '
                             'instead it is {}-dimensional'.format(len(shape)))
        # if shape[1] != len(labels):
        #    raise ValueError('Vectors in multiclass y_proba should be '
        #                     '{}-dimensional, instead they are {}-dimensional'.
        #                     format(len(labels), shape[1]))

    def _init_from_pred_labels(self, y_pred_labels):
        type_of_label = type(self.labels[0])
        self.y_proba = np.zeros(
            (len(y_pred_labels), len(self.labels)), dtype=np.float64)
        for ps_i, label_list in zip(self.y_proba, y_pred_labels):
            # converting single labels to list of labels, assumed below
            if type(label_list) != np.ndarray and type(label_list) != list:
                label_list = [label_list]
            label_list = list(map(type_of_label, label_list))
            for label in label_list:
                ps_i[self.labels.index(label)] = 1.0 / len(label_list)

    def set_valid_in_train(self, predictions, test_is):
        self.y_proba[test_is] = predictions.y_proba

    @property
    def valid_indexes(self):
        return ~np.isnan(self.y_proba[:, 0])

    @property
    def y_pred(self):
        return self.y_proba

    @property
    def y_pred_label_index(self):
        """Multi-class y_pred is the index of the predicted label."""
        return np.argmax(self.y_proba, axis=1)

    @property
    def y_pred_label(self):
        return self.labels[self.y_pred_label_index]
