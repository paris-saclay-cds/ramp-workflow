"""Base wrapper of scikit-learn-style numpy array predictions.

Functionalities:
- Hiding implementation details of prediction types (e.g., number of
  dimensions, semantics of columns).
- Providing a numpy array view through ``y_pred``. The ``y_pred`` view is
  used in ``rampwf.score``s as input, and in the default implementation
  of combining
- Handling cross-validation slices through ``set_valid_in_train``.
- Handling nan's in CV bagging by ``valid_indexes``.
- Combining ``Prediction``s (for CV bagging and ensembling). The default is
  to take the (nan)mean of the ``y_pred``s, but it can be overridden in
  derived classes.
Derived classes should all implement ``Prediction`` because we implement
polymorphism through importing ``Prediction`` from the particular file.
``Prediction``s can be asymmetric: ground truth (``y_true``) and predictions
are all stored in these classes. All ``rampwf.score_type``s accept
ground truth as the first argument and prediction as the second.
When constructors are called with a shape, they make an empty ``Prediction``.
This is to store combined ``Prediction``s, so the shape should be the same
as predictions' shape (in case of asymmetric ground truth/prediction).
"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import numpy as np


class BasePrediction(object):

    def __init__(self, y_pred):
        self.y_pred = y_pred

    def __str__(self):
        return 'y_pred = {}'.format(self.y_pred)

    @property
    def valid_indexes(self):
        """Return valid indices (e.g., a cross-validation slice)."""
        return ~np.isnan(self.y_pred)

    def set_valid_in_train(self, predictions, test_is):
        """Set a cross-validation slice."""
        self.y_pred[test_is] = predictions.y_pred

    @classmethod
    def combine(cls, predictions_list, index_list=None):
        """Combine predictions in predictions_list[index_list].

        The default implemented here is by taking the mean of their y_pred
        views. It can be overridden in derived classes.

        E.g. for regression it is the actual
        predictions, and for classification it is the probability array (which
        should be calibrated if we want the best performance). Called both for
        combining one submission on cv folds (a single model that is trained on
        different folds) and several models on a single fold.

        Parameters
        ----------
        predictions_list : list of instances of BasePrediction
            Each element of the list is an instance of BasePrediction with the
            same length and type.
        index_list : None | list of integers
            The subset of predictions to be combined. If None, the full set is
            combined.

        Returns
        -------
        combined_predictions : instance of cls
            A predictions instance containing the combined predictions.
        """
        if index_list is None:  # we combine the full list
            index_list = range(len(predictions_list))

        y_comb_list = np.array(
            [predictions_list[i].y_pred for i in index_list])

        y_comb = np.nanmean(y_comb_list, axis=0)
        combined_predictions = cls(
            labels=predictions_list[0].labels, y_pred=y_comb)
        return combined_predictions
