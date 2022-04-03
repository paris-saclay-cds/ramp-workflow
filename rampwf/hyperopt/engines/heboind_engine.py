from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import numpy as np
from .generic_engine import GenericEngine


class HEBOINDEngine(GenericEngine):
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        print("hypers", self.hyperparameters)
        self.converted_hyperparams_ = []

        for h in self.hyperparameters:
            self.converted_hyperparams_.append({
                "name": h.name,
                "type": "cat",
                "categories": h.values,
            })
        print("converted", self.converted_hyperparams_)
        self.space = DesignSpace().parse(self.converted_hyperparams_)
        self._opt = HEBO(self.space)

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
        fold_i = len(df_scores) % n_folds
        next_value_indices = []
        self.next = self._opt.suggest(n_suggestions=1)
        for h in self.hyperparameters:
            next_value_indices.append(np.where(h.values == self.next.iloc[0][h.name])[0][0])
        return fold_i, next_value_indices

    def pass_feedback(self, fold_i, n_folds, df_scores, score_name):

        score_ = df_scores.loc['valid', score_name]
        self._opt.observe(self.next, np.asarray([score_]).reshape(-1, 1))