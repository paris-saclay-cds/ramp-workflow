import numpy as np
from .base import BasePrediction


class Predictions(BasePrediction):

    def __init__(self, labels=None, y_pred=None, y_true=None, shape=None):
        self.labels = labels
        if y_pred is not None:
            self.y_pred = np.array(y_pred)
        elif y_true is not None:
            self.y_pred = np.array(y_true)
        elif shape is not None:
            self.y_pred = np.empty(shape, dtype=float)
            self.y_pred.fill(np.nan)
        else:
            raise ValueError("Missing init argument: y_pred, y_true, f_name "
                             "or n_samples")

    def set_valid_in_train(self, predictions, test_is):
        """Set a cross validation slice."""
        self.y_pred[test_is] = predictions.y_pred

    @property
    def valid_indexes(self):
        return ~np.isnan(self.y_pred)
