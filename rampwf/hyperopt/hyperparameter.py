import re
import os
import shutil
import numpy as np
from tempfile import mkdtemp
from ..utils import (
    assert_read_problem, import_file, run_submission_on_cv_fold)

HYPERPARAMS_REPL_REGEX = re.compile(
    '# RAMP START HYPERPARAMETERS.*# RAMP END HYPERPARAMETERS', re.S)


class Hyperparameter(object):
    """Discrete grid hyperparameter.

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

    def __init__(self, default=None, values=None, prior=None):
        self.name = ''
        self.workflow_element_name = ''
        if default is None and values is None:
            raise ValueError('Either default or values must be defined.')
        if values is None:
            self.values = [default]
        else:
            if len(values) < 1:
                raise ValueError(
                    'Values needs to contain at least one element.')
            self.values = values
        if default is None:
            self.default_index = 0
        else:
            if default not in self.values:
                raise ValueError('Default must be among values.')
            else:
                self.default_index = self.values.index(default)

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

    @property
    def default(self):
        return self.values[self.default_index]

    @property
    def default_repr(self):
        if isinstance(self.default, str):
            return '\'{}\''.format(self.default)
        else:
            return self.default

    @property
    def values_repr(self):
        s = '['
        for v in self.values:
            if isinstance(v, str):
                s += '\'{}\', '.format(v)
            else:
                s += '{}, '.format(v)
        s += ']'
        return s

    def __int__(self):
        return int(self.values[self.default_index])

    def __float__(self):
        return float(self.values[self.default_index])

    def __str__(self):
        return str(self.values[self.default_index])


class RandomEngine(object):
    """Discrete grid hyperparameter space.

    Attributes:
        hyperparameters: a list of Hyperparameters
        engine: a hyperopt engine
        ramp_kit_dir: the directory where the ramp kit is found
        submission_dir: the directory where the submission to be optimized
            is found
    """

    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def next_hyperparameter_indices(self):
        next_value_indices = []
        for h in self.hyperparameters:
            prior = np.clip(h.prior, 1e-15, 1 - 1e-15)
            prior /= prior.sum()
            next_value_indices.append(
                np.random.choice(range(len(h.values)), p=prior))
        return next_value_indices


class HyperparameterExperiment(object):
    """Discrete grid hyperparameter space.

    Attributes:
        hyperparameters: a list of Hyperparameters
        engine: a hyperopt engine
        ramp_kit_dir: the directory where the ramp kit is found
        submission_dir: the directory where the submission to be optimized
            is found
    """

    def __init__(self, hyperparameters, engine, ramp_kit_dir, submission_dir):
        self.hyperparameters = hyperparameters
        self.engine = engine
        self.ramp_kit_dir = ramp_kit_dir
        self.submission_dir = submission_dir
        self.columns = ['round', 'fold_i', 'n_train', 'n_test', 'score']
        self.hyperparameter_names = [h.name for h in hyperparameters]
        self.columns[4:4] = self.hyperparameter_names

    def run(self, n_iter):
        problem = assert_read_problem(self.ramp_kit_dir)
        X_train, y_train = problem.get_train_data(path=self.ramp_kit_dir)
        cv = list(problem.get_cv(X_train, y_train))
        hypers_per_element = {}
        for h in self.hyperparameters:
            if h.workflow_element_name not in hypers_per_element.keys():
                hypers_per_element[h.workflow_element_name] = [h]
            else:
                hypers_per_element[h.workflow_element_name].append(h)
        for i in range(n_iter):
            # Getting new hyper values
            next_value_indices = self.engine.next_hyperparameter_indices()
            for h, i in zip(self.hyperparameters, next_value_indices):
                h.default_index = i
            # Dumping new files
            output_dir_name = mkdtemp()
            for wen, hs in hypers_per_element.items():
                hyper_section = '# RAMP START HYPERPARAMETERS\n'
                for h in hs:
                    hyper_section += '{} = Hyperparameter({}'.format(
                        h.name, h.default_repr)
                    hyper_section += ', values={})\n'.format(h.values_repr)
                hyper_section += '# RAMP END HYPERPARAMETERS'
                f_name = os.path.join(self.submission_dir, wen + '.py')
                with open(f_name) as f:
                    content = f.read()
                    content = HYPERPARAMS_REPL_REGEX.sub(
                        hyper_section, content)
                output_f_name = os.path.join(output_dir_name, wen + '.py')
                with open(output_f_name, "w") as f:
                    f.write(content)
            # Calling the training script
            _, _, df_scores = run_submission_on_cv_fold(
                problem, module_path=output_dir_name, fold=cv[0],
                X_train=X_train, y_train=y_train)
            shutil.rmtree(output_dir_name)


def init_hyperopt(ramp_kit_dir, submission):
    problem = assert_read_problem(ramp_kit_dir)
    hyperopt_submission = submission + '_hyperopt'
    submission_dir = os.path.join(ramp_kit_dir, 'submissions', submission)
    hyperopt_submission_dir = os.path.join(
        ramp_kit_dir, 'submissions', hyperopt_submission)
    if os.path.exists(hyperopt_submission_dir):
        shutil.rmtree(hyperopt_submission_dir)
    shutil.copytree(submission_dir, hyperopt_submission_dir)
    hyperparameters = []
    for wen in problem.workflow.element_names:
        workflow_element = import_file(hyperopt_submission_dir, wen)
        for object_name in dir(workflow_element):
            o = getattr(workflow_element, object_name)
            if type(o) == Hyperparameter:
                o.name = object_name
                o.workflow_element_name = wen
                hyperparameters.append(o)
    engine = RandomEngine(hyperparameters)
    hyperparameter_experiment = HyperparameterExperiment(
        hyperparameters, engine, ramp_kit_dir, hyperopt_submission_dir)
    return hyperparameter_experiment
