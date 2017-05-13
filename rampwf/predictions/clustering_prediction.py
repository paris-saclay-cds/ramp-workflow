import numpy as np
from .base_prediction import BasePrediction


class Predictions(BasePrediction):

    def __init__(self, labels=None, y_pred=None, y_true=None, f_name=None,
                 n_samples=None):
        self.labels = labels
        if y_pred is not None:
            self.y_pred = np.array(y_pred)
        elif y_true is not None:
            self.y_pred = np.array(y_true)
        elif f_name is not None:
            self.y_pred = np.load(f_name)
        elif n_samples is not None:
            self.y_pred = np.empty((n_samples, 2))
            self.y_pred.fill(np.nan)
        else:
            raise ValueError("Missing init argument: y_pred, y_true, f_name "
                             "or n_samples")

    def set_valid_in_train(self, predictions, test_is):
        self.y_pred[test_is] = predictions.y_pred

    @property
    def valid_indexes(self):
        return ~np.isnan(self.y_pred[:, 1])

    @property
    def y_pred_comb(self):
        """Return an array which can be combined by taking means."""
        return self.y_pred

    @property
    def n_samples(self):
        return self.y_pred.shape[0]

    # def combine(self, indexes=[]):
        # Not yet used

        # usually the class contains arrays corresponding to predictions
        # for a set of different data points. Here we assume that it is
        # a list of predictions produced by different functions on the same
        # data point. We return a single PrdictionType

        # Just saving here in case we want to go back there how to
        # combine based on simply ranks, k = len(indexes)
        # n = len(y_preds[0])
        # n_ones = n * k - y_preds[indexes].sum() # number of zeros
        # if len(indexes) == 0:  # we combine the full list
        #    indexes = range(len(self.y_probas_array))
        # combined_y_preds = self.y_preds_array.mean()
        # combined_prediction = PredictionType(combined_y_preds)
        # return combined_prediction
