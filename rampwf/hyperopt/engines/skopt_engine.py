from skopt.space import Categorical
from skopt import Optimizer
import numpy as np

class SKOptEngine(object):
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.converted_hyperparams_ = []

        for h in self.hyperparameters:
            print(h.values)
            self.converted_hyperparams_.append(
                Categorical(h.values, name=h.name)
            )
        self._opt = Optimizer(
            dimensions=self.converted_hyperparams_,
            random_state=1,
            base_estimator='gp'
        )
        self._mean = 0


    def next_hyperparameter_indices(self, df_scores, n_folds):
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
            incomplete_folds = incomplete_folds.reset_index()
            next_values = incomplete_folds.iloc[0][
                [h.name for h in self.hyperparameters]].values
            next_value_indices = [
                h.get_index(v) for h, v
                in zip(self.hyperparameters, next_values)]
            # for some reason iloc converts int to float
            fold_i = int(incomplete_folds.iloc[0]['fold_i']) % n_folds
        # Otherwise select hyperparameter values from those that haven't
        # been selected yet, using also prior
        else:
            fold_i = 0
            next_value_indices = []
            self.next = self._opt.ask()
            for idx, h in enumerate(self.hyperparameters):
                next_value_indices.append(np.where(h.values == self.next[idx])[0][0])
        return fold_i, next_value_indices

    def pass_feedback(self, fold_i, n_folds, df_scores, score_name):
        self._opt.tell(self.next, -df_scores.loc['valid', score_name])

