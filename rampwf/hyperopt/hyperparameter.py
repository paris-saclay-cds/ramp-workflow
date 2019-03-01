import re
import os
import shutil
import numpy as np
import pandas as pd
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

    def __init__(self, dtype, default=None, values=None, prior=None):
        self.name = ''
        self.workflow_element_name = ''
        self.dtype = dtype
        if default is None and values is None:
            raise ValueError('Either default or values must be defined.')
        if values is None:
            self.values = np.array([default], dtype=self.dtype)
        else:
            if len(values) < 1:
                raise ValueError(
                    'Values needs to contain at least one element.')
            self.values = np.array(values, dtype=self.dtype)
        if default is None:
            self.default_index = 0
        else:
            if default not in self.values:
                raise ValueError('Default must be among values.')
            else:
                self.set_default(default)

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
        if self.dtype == 'object':
            return '\'{}\''.format(self.default)
        else:
            return self.default

    @property
    def values_repr(self):
        s = '['
        for v in self.values:
            if self.dtype == 'object':
                s += '\'{}\', '.format(v)
            else:
                s += '{}, '.format(v)
        s += ']'
        return s

    def set_default(self, default):
        self.default_index = list(self.values).index(default)

    def __int__(self):
        return int(self.values[self.default_index])

    def __float__(self):
        return float(self.values[self.default_index])

    def __str__(self):
        return str(self.values[self.default_index])

    def python_repr(self):
        repr = '{} = Hyperparameter(\n'.format(self.name)
        repr += '\tdtype={}'.format(str(self.dtype))
        repr += ', default={}'.format(self.default_repr)
        repr += ', values={})\n'.format(self.values_repr)
        return repr


class RandomEngine(object):
    """Random search hyperopt engine.

    Attributes:
        hyperparameters: a list of Hyperparameters
    """

    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def next_hyperparameter_indices(self, df_scores):
        next_value_indices = []
        df_scores_local = df_scores.copy()
        for h in self.hyperparameters:
            prior = np.clip(h.prior, 1e-15, 1 - 1e-15)
            frequencies = np.ones(len(prior))
            if len(df_scores_local) > 0:
                for i, v in np.ndenumerate(h.values):
                    frequencies[i] = len(
                        df_scores_local[df_scores_local[h.name] == v])
            frequencies = max(frequencies) - frequencies
            prior *= frequencies
            prior /= prior.sum()
            selected_index = np.random.choice(range(len(h.values)), p=prior)
            df_scores_local = df_scores_local[(
                df_scores_local[h.name] == h.values[selected_index])]
            next_value_indices.append(selected_index)
        return next_value_indices


class HyperparameterExperiment(object):
    """A hyperparameter optimization experiment.

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
        self.problem = assert_read_problem(ramp_kit_dir)
        self.X_train, self.y_train = self.problem.get_train_data(
            path=ramp_kit_dir)
        self.cv = list(self.problem.get_cv(self.X_train, self.y_train))
        self.submission_dir = submission_dir
        self.columns = ['fold_i', 'n_train', 'n_test', 'score']
        self.hyperparameter_names = [h.name for h in hyperparameters]
        self.columns[4:4] = self.hyperparameter_names
        self.df_scores_ = None
        self.df_summary_ = None

    def get_next_df_scores(self, module_path, fold):
        _, _, df_scores = run_submission_on_cv_fold(
            self.problem, module_path=module_path, fold=fold,
            X_train=self.X_train, y_train=self.y_train)
        return df_scores

    def run(self, n_iter):
        hyperopt_output_path = os.path.join(
            self.submission_dir, 'hyperopt_output')
        if not os.path.exists(hyperopt_output_path):
            os.makedirs(hyperopt_output_path)

        hypers_per_element = {
            wen: [] for wen in self.problem.workflow.element_names}
        for h in self.hyperparameters:
            hypers_per_element[h.workflow_element_name].append(h)
        score_names = [s.name for s in self.problem.score_types]
        scores_columns = ['fold_i']
        scores_columns += self.hyperparameter_names
        scores_columns += ['train_' + name for name in score_names]
        scores_columns += ['valid_' + name for name in score_names]
        scores_columns += ['train_time', 'valid_time', 'n_train', 'n_valid']
        dtypes = ['int'] + [h.dtype for h in self.hyperparameters] +\
            ['float'] * 2 * len(score_names) + ['float'] * 2 + ['int'] * 2
        self.df_scores_ = pd.DataFrame(columns=scores_columns)
        for column, dtype in zip(scores_columns, dtypes):
            self.df_scores_[column] = self.df_scores_[column].astype(dtype)
        for i in range(n_iter):
            # Getting new hyper values
            next_value_indices = self.engine.next_hyperparameter_indices(
                self.df_scores_)
            for h, i in zip(self.hyperparameters, next_value_indices):
                h.default_index = i
            # Dumping new files
            output_dir_name = mkdtemp()
            for wen, hs in hypers_per_element.items():
                hyper_section = '# RAMP START HYPERPARAMETERS\n'
                for h in hs:
                    hyper_section += h.python_repr()
                hyper_section += '# RAMP END HYPERPARAMETERS'
                f_name = os.path.join(self.submission_dir, wen + '.py')
                with open(f_name) as f:
                    content = f.read()
                    content = HYPERPARAMS_REPL_REGEX.sub(
                        hyper_section, content)
                output_f_name = os.path.join(output_dir_name, wen + '.py')
                with open(output_f_name, 'w') as f:
                    f.write(content)
            # Calling the training script
            for fold_i, fold in enumerate(self.cv):
                df_scores = self.get_next_df_scores(output_dir_name, fold)
                row = {'fold_i': fold_i}
                for h in self.hyperparameters:
                    row[h.name] = h.default
                for name in score_names:
                    row['train_' + name] = df_scores.loc['train'][name]
                    row['valid_' + name] = df_scores.loc['valid'][name]
                row['train_time'] = float(df_scores.loc['train']['time'])
                row['valid_time'] = float(df_scores.loc['valid']['time'])
                row['n_train'] = len(fold[0])
                row['n_valid'] = len(fold[1])
                self.df_scores_ = self.df_scores_.append(
                    row, ignore_index=True)
            shutil.rmtree(output_dir_name)
        summary_groupby = self.df_scores_.groupby(
            self.hyperparameter_names)
        means = summary_groupby.mean().drop(columns=['fold_i'])
        stds = summary_groupby.std().drop(columns=['fold_i'])
        counts = summary_groupby.count()[['n_train']].rename(
            columns={'n_train': 'n_folds'})
        self.df_summary_ = pd.merge(
            means, stds, left_index=True, right_index=True,
            suffixes=('_m', '_s'))
        self.df_summary_ = pd.merge(
            counts, self.df_summary_, left_index=True, right_index=True)
        print self.df_summary_
        summary_fname = os.path.join(hyperopt_output_path, 'summary.csv')
        self.df_summary_.to_csv(summary_fname)
        official_scores = self.df_summary_[
            'valid_' + self.problem.score_types[0].name + '_m']
        best_defaults = official_scores.idxmax()
        print best_defaults
        for bd, h in zip(best_defaults, self.hyperparameters):
            h.set_default(bd)
        output_dir_name = self.submission_dir
        for wen, hs in hypers_per_element.items():
            hyper_section = '# RAMP START HYPERPARAMETERS\n'
            for h in hs:
                hyper_section += h.python_repr()
            hyper_section += '# RAMP END HYPERPARAMETERS'
            f_name = os.path.join(self.submission_dir, wen + '.py')
            with open(f_name) as f:
                content = f.read()
                content = HYPERPARAMS_REPL_REGEX.sub(
                    hyper_section, content)
            output_f_name = os.path.join(output_dir_name, wen + '.py')
            with open(output_f_name, 'w') as f:
                f.write(content)


def init_hyperopt(ramp_kit_dir, ramp_submission_dir, submission, engine_name):
    problem = assert_read_problem(ramp_kit_dir)
    hyperopt_submission = submission + '_hyperopt'
    hyperopt_submission_dir = os.path.join(
        ramp_submission_dir, hyperopt_submission)
    submission_dir = os.path.join(
        ramp_submission_dir, submission)
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
    if engine_name == 'random':
        engine = RandomEngine(hyperparameters)
    else:
        raise ValueError('{} is not a valide engine name'.format(engine_name))
    hyperparameter_experiment = HyperparameterExperiment(
        hyperparameters, engine, ramp_kit_dir, hyperopt_submission_dir)
    return hyperparameter_experiment


def run_hyperopt(ramp_kit_dir, ramp_data_dir, ramp_submission_dir,
                 submission, engine_name, n_iter, save_best=True,
                 is_cleanup=False):
    hyperparameter_experiment = init_hyperopt(
        ramp_kit_dir, ramp_submission_dir, submission, engine_name)
    hyperparameter_experiment.run(n_iter)
    if is_cleanup:
        shutil.rmtree(hyperparameter_experiment.submission_dir)
