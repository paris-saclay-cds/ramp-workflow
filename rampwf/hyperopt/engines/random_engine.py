import numpy as np
from .generic_engine import GenericEngine


class RandomEngine(GenericEngine):
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters

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
            fold_i, next_value_indices = \
                self.finish_incomplete_cvs(incomplete_folds, n_folds)
        # Otherwise select hyperparameter values from those that haven't
        # been selected yet, using also prior
        else:
            fold_i = 0
            next_value_indices = []
            df_scores_local = df_scores.copy()
            for h in self.hyperparameters:
                # unnormalized but positive prior
                prior = np.clip(h.prior, 1e-15, None)
                # How many times each hyperparameter value was tried, given the
                # selected values next_value_indices so far
                frequencies = np.zeros(len(prior))
                if len(df_scores_local) > 0:
                    for i, v in np.ndenumerate(h.values):
                        frequencies[i] = len(
                            df_scores_local[df_scores_local[h.name] == v])
                # How many times each hyperparameter value was not tried, given
                # the selected values next_value_indices so far, in this round
                # of full grid search
                frequencies = max(frequencies) - frequencies
                prior *= frequencies
                if prior.sum() <= 0:
                    prior = np.ones(len(prior))
                prior /= prior.sum()
                selected_index = np.random.choice(
                    range(len(h.values)), p=prior)
                # keep only experiments that used the selected values so far
                df_scores_local = df_scores_local[(
                    df_scores_local[h.name] == h.values[selected_index])]
                next_value_indices.append(selected_index)
        return fold_i, next_value_indices

    def pass_feedback(self, fold_i, n_folds, df_scores, score_name):
        pass
