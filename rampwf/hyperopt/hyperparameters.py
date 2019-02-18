class Hyperparameter(object):
    """Discrete grid hyperparameters.

    Attributes:
        name: the name of the hyper-parameter variable, used in user interface,
            both for specifying the grid of values and getting the report on
            an experiment.
        values: the list of hyperparameter values
        n_values: the number of hyperparameter values
        prior: a list of probabilities
            positivity and summing to 1 are not checked, hyperparameter
            optimizers should do that when using the list
    """

    def __init__(self, name, values, prior=None):
        self.name = name
        self.values = values
        if prior is None:
            self.prior = [1. / self.n_values] * self.n_values
        else:
            if len(prior) != len(values):
                raise ValueError(
                    'len(values) == {} != {} == len(prior)'.format(
                        len(values), len(prior)))
            self.prior = prior

    @property
    def n_values(self):
        return len(self.values)
