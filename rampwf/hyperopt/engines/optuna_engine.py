import optuna
from optuna.samplers import TPESampler
from .generic_engine import GenericEngine


class OptunaIndEngine(GenericEngine):
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        sampler = TPESampler(**TPESampler.hyperopt_parameters())
        self.study = optuna.create_study(direction="maximize", sampler=sampler)
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
        self.trial = self.study.ask()
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
            for h in self.hyperparameters:
                next = self.trial.suggest_int(h.name, 0, len(h.values) - 1)
                next_value_indices.append(next)

        return fold_i, next_value_indices

    def pass_feedback(self, fold_i, n_folds, df_scores, score_name):
        self.study.tell(self.trial, df_scores.loc['valid', score_name])
