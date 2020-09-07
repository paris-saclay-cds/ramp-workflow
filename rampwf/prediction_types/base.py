"""Base wrapper of scikit-learn-style numpy array predictions.

Functionalities:
- Hiding implementation details of prediction types (e.g., number of
  dimensions, semantics of columns).
- Providing a numpy array view through ``y_pred``. The ``y_pred`` view is
  used in ``rampwf.score``s as input, and in the default implementation
  of combining.
- Handling cross-validation slices through ``set_valid_in_train`` and
  ``set_slice``.
- Handling NaN's in CV bagging by ``valid_indexes``.
- Combining ``Prediction``s (for CV bagging and ensembling). The default is
  to take the (nan)mean of the ``y_pred``s, but it can be overridden in
  derived classes.
Derived classes should all implement ``Predictions`` because we implement
polymorphism through importing ``Predictions`` from the particular file.
``Predictions``s can be asymmetric: ground truth (``y_true``) and predictions
are all stored in these classes. All ``rampwf.score_type``s accept
ground truth as the first argument and prediction as the second.
When constructors are called with a shape, they make an empty ``Predictions``.
This is to store combined ``Predictions``s, so the shape should be the same
as predictions' shape (in case of asymmetric ground truth/prediction).
"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import numpy as np
import warnings


class BasePrediction(object):
    def __str__(self):
        return 'y_pred = {}'.format(self.y_pred)

    @property
    def valid_indexes(self):
        """Return valid indices (e.g., a cross-validation slice).

        When combining Predictions on different cross validation slices,
        we start with an empty y_pred. Each time a fold is added, some
        entries become valid. Invalid entries are those that are not
        predicted by any folds, i.e., those that remain NaN.
        """
        if len(self.y_pred.shape) == 1:
            return ~np.isnan(self.y_pred)
        elif len(self.y_pred.shape) == 2:
            return ~np.isnan(self.y_pred[:, 0])
        else:
            raise ValueError('y_pred.shape > 2 is not implemented')

    def set_valid_in_train(self, predictions, test_is):
        """Set a cross-validation slice."""
        self.y_pred[test_is] = predictions.y_pred

    def set_slice(self, valid_indexes):
        """Collapsing y_pred to a cross-validation slice.

        So scores do not need to deal with masks.
        """
        self.y_pred = self.y_pred[valid_indexes]

    def check_y_pred_dimensions(self):
        if self.n_columns == 0 and len(self.y_pred.shape) != 1:
            raise ValueError(
                'Wrong y_pred dimensions: y_pred should be 1D, '
                'instead its shape is {}'.format(self.y_pred.shape))
        if self.n_columns > 0:
            if len(self.y_pred.shape) != 2 or\
                    self.y_pred.shape[1] != self.n_columns:
                raise ValueError(
                    'Wrong y_pred dimensions: y_pred should be 2D '
                    'with {} columns, instead its shape is {}'.format(
                        self.n_columns, self.y_pred.shape))

    @classmethod
    def combine(cls, predictions_list, index_list=None):
        """Combine predictions in predictions_list[index_list].

        The default implemented here is by taking the mean of their y_pred
        views. It can be overridden in derived classes.

        E.g. for regression it is the actual
        predictions, and for classification it is the probability array (which
        should be calibrated if we want the best performance). Called both for
        combining one submission on cv folds (a single model that is trained on
        different folds) and several models on a single fold (blending).

        Parameters
        ----------
        predictions_list : list of instances of Base
            Each element of the list is an instance of Base with the
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
        # I expect to see RuntimeWarnings in this block
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            y_comb = np.nanmean(y_comb_list, axis=0)
        combined_predictions = cls(y_pred=y_comb)
        return combined_predictions
