import numpy as np


class BasePrediction(object):

    def __init__(self, y_pred):
        self.y_pred = y_pred

    def __str__(self):
        return "y_pred = ".format(self.y_pred)

    @property
    def n_samples(self):
        return len(self.y_pred)

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
