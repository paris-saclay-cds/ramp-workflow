"""Mixed classification/regression predictions.

The first shape[1] - 1 columns are classification posteriors, the
last column is a regression.

Alternatively, we would need to define workflows with multiple outputs,
and let scores handle them separately.

"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import numpy as np
from .base import BasePrediction


def _combined_init(self, y_pred=None, y_true=None, n_samples=None):
    self.predictions_list = []
    if y_pred is not None:
        start = 0
        for Predictions in self.Predictions_list:
            end = start + Predictions.n_columns
            predictions = Predictions(y_pred=y_pred[:, start:end])
            self.predictions_list.append(predictions)
            start += Predictions.n_columns
    elif y_true is not None:
        start = 0
        for Predictions in self.Predictions_list:
            n_columns = 1
            if Predictions.n_columns_true != 0:
                n_columns = Predictions.n_columns_true
            end = start + n_columns
            predictions = Predictions(y_true=y_true[:, start:end])
            self.predictions_list.append(predictions)
            start += n_columns
    elif n_samples is not None:
        for Predictions in self.Predictions_list:
            n_columns = Predictions.n_columns
            predictions = Predictions(n_samples=n_samples)
            self.predictions_list.append(predictions)
    else:
        raise ValueError(
            'Missing init argument: y_pred, y_true, or n_samples')
    self.n_columns = np.array(
        [Predictions.n_columns for Predictions in self.Predictions_list]).sum()
    self.n_columns_true = np.array(
        [Predictions.n_columns_true
         for Predictions in self.Predictions_list]).sum()


def _set_valid_in_train(self, predictions, test_is):
    """Set a cross-validation slice."""
    for to_predictions, from_predictions in zip(
            self.predictions_list, predictions.predictions_list):
        to_predictions.y_pred[test_is] = from_predictions.y_pred


@property
def _y_pred(self):
    return np.concatenate(
        [predictions.y_pred for predictions in self.predictions_list], axis=1)


def make_combined(Predictions_list):
    Predictions = type(
        'Predictions',
        (BasePrediction,),
        {'n_columns': 0,
         'n_columns_true': 0,
         'Predictions_list': Predictions_list,
         '__init__': _combined_init,
         'set_valid_in_train': _set_valid_in_train,
         'y_pred': _y_pred,
         })
    return Predictions
