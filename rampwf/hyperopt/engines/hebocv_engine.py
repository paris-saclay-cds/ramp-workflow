from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import numpy as np
from .generic_engine import GenericEngine


class HEBOCVEngine(GenericEngine):

    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.converted_hyperparams_ = []

        for h in self.hyperparameters:
            self.converted_hyperparams_.append({
                "name": h.name,
                "type": "cat",
                "categories": h.values
            })

        self.space = DesignSpace().parse(self.converted_hyperparams_)
        self._opt = HEBO(self.space)
        self._mean = 0

    def next_hyperparameter_indices(self, df_scores, n_folds, problem):
        """Return the next hyperparameter indices to try.

        Parameters:
            df_scores : pandas DataFrame
                It represents the results of the experiments that have been
                run so far.
        Return:
            next_value_indices : list of int
                The indices in corresponding to the values to try in
                hyperparameters.
        """
        # First finish incomplete cv's.
        hyperparameter_names = [h.name for h in self.hyperparameters]
        df_n_folds = df_scores.groupby(hyperparameter_names).count()
        incomplete_folds = df_n_folds[(df_n_folds['fold_i'] % n_folds > 0)]

        if len(incomplete_folds) > 0:
            fold_i, next_value_indices = self.finish_incomplete_cvs(incomplete_folds, n_folds)
        # Otherwise select hyperparameter values from those that haven't
        # been selected yet, using also prior
        else:
            fold_i = 0
            next_value_indices = []
            self.next = self._opt.suggest(n_suggestions=1)
            for h in self.hyperparameters:
                next_value_indices.append(np.where(h.values == self.next.iloc[0][h.name])[0][0])

        return fold_i, next_value_indices

    def pass_feedback(self, fold_i, n_folds, df_scores, score_name):

        self._mean += df_scores.loc['valid', score_name]
        if fold_i == n_folds - 1:
            self._opt.observe(self.next, np.asarray([self._mean / n_folds]).reshape(-1, 1))
            self._mean = 0
