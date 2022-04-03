from abc import ABC, abstractmethod

class GenericEngine(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def next_hyperparameter_indices(self, df_scores, n_folds):
        pass

    @abstractmethod
    def pass_feedback(self, fold_i, n_folds, df_scores, score_name):
        pass

    def finish_incomplete_cvs(self, incomplete_folds, n_folds, problem):
        incomplete_folds = incomplete_folds.reset_index()
        next_values = incomplete_folds.iloc[0][
            [h.name for h in self.hyperparameters]].values
        next_value_indices = [
            h.get_index(v) for h, v
            in zip(self.hyperparameters, next_values)]
        # for some reason iloc converts int to float
        fold_i = int(incomplete_folds.iloc[0]['fold_i']) % n_folds
        return fold_i, next_value_indices



