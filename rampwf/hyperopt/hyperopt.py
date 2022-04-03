"""Hyperparameter optiomization for ramp-kits."""
import re
import os
import shutil
import numpy as np
import pandas as pd
from ray import tune

from tempfile import mkdtemp
from ..utils import (
    assert_read_problem, import_module_from_source, run_submission_on_cv_fold)

from .engines import HEBOCVEngine, HEBOINDEngine, RandomEngine, SKOptEngine, OptunaIndEngine, TuneEngine

HYPERPARAMS_SECTION_START = '# RAMP START HYPERPARAMETERS'
HYPERPARAMS_SECTION_END = '# RAMP END HYPERPARAMETERS'
HYPERPARAMS_REPL_REGEX = re.compile('{}.*{}'.format(
    HYPERPARAMS_SECTION_START, HYPERPARAMS_SECTION_END), re.S)


class Hyperparameter(object):
    """Discrete grid hyperparameter.

    Represented by a list of values, a default value, the name of the
    hyperparameter (specified by the user in the workflow element), the
    name of the workflow element in which the hyperparemeter appears, and an
    optional prior probability vector.

    Attributes:
        name : string
            The name of the hyperparameter variable, used in user interface,
            both for specifying the grid of values and getting the report on
            an experiment. Initialized to '' then set in set_names, to the
            name the user chose for the variable in the workflow element.
        workflow_element_name : string
            The name of the workflow element in which the hyperparameter is
            used. Initialized to '' then set in set_names.
        dtype : string
            The dtype of the hyperparameter.
        default_index: int
            The index in values of the current value of the hyperparameter.
        values: numpy array of any dtype
            The list of hyperparameter values.
        prior: numpy array of float
            A list of reals that the hyperopt can use as a prior probability
            over values. Positivity and summing to one are not checked,
            hyperparameter optimizers should do that when using the list
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
                message = 'Default must be among values.\n'
                message += f'default: {default}\n'
                message += f'values: {self.values}'
                raise ValueError(message)
            else:
                self.set_default(default)

        if prior is None:
            self.prior = np.array([1. / self.n_values] * self.n_values)
        else:
            if len(prior) != len(values):
                raise ValueError(
                    'len(values) == {} != {} == len(prior)'.format(
                        len(values), len(prior)))
            self.prior = prior

    @property
    def n_values(self):
        """The number of hyperparameter values.

        Return:
            n_values : int
                The number of hyperparameter values len(values)
        """
        return len(self.values)

    @property
    def default(self):
        """The current value of the hyperparameter.

        Return:
            default : any dtype
                The current value of the hyperparameter values[default_index].
        """
        return self.values[self.default_index]

    @property
    def default_repr(self):
        """The string representation of the default value.

        It can be used to output the default value into a python file. For
        object types it adds '', otherwise it's the string representation of
        the default value.

        Return:
            default_repr : str
                The string representation of the default value.
        """
        if self.dtype == 'object':
            return '\'{}\''.format(self.default)
        else:
            return str(self.default)

    @property
    def values_repr(self):
        """The string representation of the list of values.

        It can be used to output the list of values into a python file. For
        object types it adds '' around the values, otherwise it's the list of
        string representations of the values in brackets.

        Return:
            values_repr : list of str
                The string representation of the list of values.
        """
        s = '['
        for v in self.values:
            if self.dtype == 'object':
                s += '\'{}\', '.format(v)
            else:
                s += '{}, '.format(v)
        s += ']'
        return s

    @property
    def python_repr(self):
        """The string representation of the hyperparameter.

        It can be used to output the hyperparameter definition into a python
        file:
        <name> = Hyperparameter(
            dtype=<dtype>, default=<default>, values=[<values>])

        Return:
            python_repr : str
                The string representation of the hyperparameter.
        """
        repr = '{} = Hyperparameter(\n'.format(self.name)
        repr += "\tdtype='{}'".format(str(self.dtype))
        repr += ', default={}'.format(self.default_repr)
        repr += ', values={})\n'.format(self.values_repr)
        return repr

    def set_names(self, name, workflow_element_name):
        """Set the name and workflow element name.

        Used when a hyperparameter object is loaded from a workflow element.

        Parameters:
            name : str
                The name of the hyperparameter, declared by the user in the
                workflow element.
            workflow_element_name : str
                The name of the workflow element in which the hyperparameter
                is defined.

        """
        self.name = name
        self.workflow_element_name = workflow_element_name

    def get_index(self, value):
        """Get the index of a value.

        Parameters:
            value : any dtype
                The value to look for.
        """
        return list(self.values).index(value)

    def set_default(self, default):
        """Set the default value.

        Parameters:
            default : any dtype
                The new default value.
        """
        self.default_index = self.get_index(default)

    def __int__(self):
        """Cast the default value into an integer.

        It can be used in the workflow element for an integer hyperparameter.

        Return:
            int(default) : int
                The integer representation of the default value.
        """
        return int(self.default)

    def __float__(self):
        """Cast the default value into an float.

        It can be used in the workflow element for an float hyperparameter.

        Return:
            float(default) : float
                The float representation of the default value.
        """
        return float(self.default)

    def __str__(self):
        """Cast the default value into a string.

        It can be used in the workflow element for a string hyperparameter.

        Return:
            str(default) : str
                The string representation of the default value.
        """
        return str(self.default)


def parse_hyperparameters(module_path, workflow_element_name):
    """Parse hyperparameters in a workflow element.

    Load the module, take all Hyperparameter objects, and set the name of each
    to the name of the hyperparameter the user chose and the workflow element
    name of each to workflow_element_name.

    Parameters:
        module_path : str
            The path to the submission directory.
        workflow_element_name : string
            The name of the workflow element.
    Return:
        hyperparameters : list of instances of Hyperparameter
    """
    hyperparameters = []
    workflow_element = import_module_from_source(
        os.path.join(module_path, workflow_element_name + '.py'),
        workflow_element_name
    )
    for object_name in dir(workflow_element):
        o = getattr(workflow_element, object_name)
        if type(o) == Hyperparameter:
            o.set_names(object_name, workflow_element_name)
            hyperparameters.append(o)
    return hyperparameters


def parse_all_hyperparameters(module_path, workflow):
    """Parse hyperparameters in a submission.

    Load all the the modules, take all Hyperparameter objects, and set the name
    of each to the name of the hyperparameter the user chose and the workflow
    element name of each to the corresponding workflow_element_name.

    Parameters:
        module_path : str
            The path to the submission directory.
        workflow_element_name : string
            The name of the workflow element.
    Return:
        hyperparameters : list of instances of Hyperparameter
    """
    hyperparameters = []
    for wen in workflow.element_names:
        hyperparameters += parse_hyperparameters(module_path, wen)
    return hyperparameters


def write_hyperparameters(submission_dir, output_submission_dir,
                          hypers_per_workflow_element):
    """Write hyperparameters in a submission.

    Read workflow elements from submission_dir, replace the hyperparameter
    section with the hyperparameters in the hypers_per_workflow_element
    dictionary (with new hyperparamter values set by, e.g, a hyperopt engine),
    then write the new workflow elements into output_submission_dir (which
    can be a temporary directory or submission_dir itself when the function
    is called to replace the hyperparameters in the input submission with the
    best hyperparameters.)

    Parameters:
        submission_dir : str
            The path to the submission directory from which the submission is
            read.
        output_submission_dir : str
            The path to the output submission directory into which the
            submission with the new hyperparameter values is written.
        hypers_per_workflow_element : dictionary
            Each key is a workflow element name and each value is a list of
            Hyperparameter instances, representing the hyperparemters in
            the workflow element.
    """
    for wen, hs in hypers_per_workflow_element.items():
        hyper_section = '{}\n'.format(HYPERPARAMS_SECTION_START)
        for h in hs:
            hyper_section += h.python_repr
        hyper_section += HYPERPARAMS_SECTION_END
        f_name = os.path.join(submission_dir, wen + '.py')
        with open(f_name) as f:
            content = f.read()
            content = HYPERPARAMS_REPL_REGEX.sub(hyper_section, content)
        output_f_name = os.path.join(output_submission_dir, wen + '.py')
        with open(output_f_name, 'w') as f:
            f.write(content)


class HyperparameterOptimization(object):
    """A hyperparameter optimization.

    Attributes:
        hyperparameters: a list of Hyperparameters
        engine: a hyperopt engine
        ramp_kit_dir: the directory where the ramp kit is found
        submission_dir: the directory where the submission to be optimized
            is found
    """

    def __init__(self, hyperparameters, engine, ramp_kit_dir,
                 submission_dir, data_label):
        self.hyperparameters = hyperparameters
        self.engine = engine
        self.problem = assert_read_problem(ramp_kit_dir)
        if data_label is not None:
            self.X_train, self.y_train = self.problem.get_train_data(
                path=ramp_kit_dir, data_label=data_label)
            self.X_test, self.y_test = self.problem.get_test_data(
                path=ramp_kit_dir, data_label=data_label)
        else:
            self.X_train, self.y_train = self.problem.get_train_data(
                path=ramp_kit_dir)
            self.X_test, self.y_test = self.problem.get_test_data(
                path=ramp_kit_dir)
        self.cv = list(self.problem.get_cv(self.X_train, self.y_train))
        self.submission_dir = submission_dir
        self.hyperparameter_names = [h.name for h in hyperparameters]
        self.hyperparameters_indices = [h.name + "_i" for h in hyperparameters]
        self.score_names = [s.name for s in self.problem.score_types]
        self.df_summary_ = None
        self.fold_i = 0

        # Set up hypers_per_workflow_element dictionary: keys are
        # workflow element names, values are lists are hypers belonging
        # to the workflow element
        self.hypers_per_workflow_element = {
            wen: [] for wen in self.problem.workflow.element_names}
        for h in self.hyperparameters:
            self.hypers_per_workflow_element[h.workflow_element_name].append(h)

        # Set up df_scores_ which will contain one row per experiment
        scores_columns = ['fold_i']
        scores_columns += self.hyperparameter_names + self.hyperparameters_indices
        scores_columns += ['train_' + name for name in self.score_names]
        scores_columns += ['valid_' + name for name in self.score_names]
        scores_columns += ['train_time', 'valid_time', 'n_train', 'n_valid']
        dtypes = ['int'] + [h.dtype for h in self.hyperparameters] +\
                 ['int'] * len(self.hyperparameters) + \
            ['float'] * 2 * len(self.score_names) + ['float'] * 2 + ['int'] * 2
        self.df_scores_ = pd.DataFrame(columns=scores_columns)
        for column, dtype in zip(scores_columns, dtypes):
            self.df_scores_[column] = self.df_scores_[column].astype(dtype)

        self.df_best_scores_ = pd.DataFrame(columns=['valid_' + name for name in self.score_names])


    def _run_next_experiment(self, module_path, fold_i):
        _, _, df_scores = run_submission_on_cv_fold(
            self.problem, module_path=module_path, fold=self.cv[fold_i],
            X_train=self.X_train, y_train=self.y_train, X_test=self.X_test, y_test=self.y_test)
        return df_scores

    def _update_df_scores(self, df_scores, fold_i, test):
        row = {'fold_i': fold_i}
        for h in self.hyperparameters:
            row[h.name] = h.default
            row[h.name + "_i"] = h.default_index
        for name in self.score_names:
            row['train_' + name] = df_scores.loc['train'][name]
            row['valid_' + name] = df_scores.loc['valid'][name]
            if test:
                row['test_' + name] = df_scores.loc['test'][name]
        row['train_time'] = float(df_scores.loc['train']['time'])
        row['valid_time'] = float(df_scores.loc['valid']['time'])
        row['valid_time'] = float(df_scores.loc['test']['time'])

        row['n_train'] = len(self.cv[fold_i][0])
        row['n_valid'] = len(self.cv[fold_i][1])
        row['n_test'] = len(self.X_test[0])


        self.df_scores_ = self.df_scores_.append(row, ignore_index=True)


    def _make_and_save_summary(self, hyperopt_output_path):
        summary_fname = os.path.join(hyperopt_output_path, 'summary.csv')
        self.df_scores_.to_csv(summary_fname)

    def _save_best_model(self):
        official_scores = self.df_summary_[
            'valid_' + self.problem.score_types[0].name + '_m']
        print("official scores", official_scores)
        if self.problem.score_types[0].is_lower_the_better:
            best_defaults = official_scores.idxmin()
        else:
            best_defaults = official_scores.idxmax()
        print('Best hyperparameters: ', best_defaults)
        try:
            for bd, h in zip(best_defaults, self.hyperparameters):
                h.set_default(bd)
        except(TypeError):
            # single hyperparameter
            self.hyperparameters[0].set_default(best_defaults)
        # Overwrite the submission with the best hyperparameter values
        write_hyperparameters(
            self.submission_dir, self.submission_dir,
            self.hypers_per_workflow_element)


    def run_tune(self, hyperopt_output_path):

        def easy_objective(config):
            next_value_indices = []
            for idx, h in enumerate(self.hyperparameters):
                next_value_indices.append(np.where(h.values == config[h.name])[0][0])
                # Writing submission files with new hyperparameter values
            for h, i in zip(self.hyperparameters, next_value_indices):
                h.default_index = i
            output_submission_dir = mkdtemp()
            os.chdir(self.current_dir)
            write_hyperparameters(
                self.submission_dir, output_submission_dir,
                self.hypers_per_workflow_element)
            # Calling the training script.
            self.fold_i = (self.fold_i + 1) % len(self.cv)
            df_scores = self._run_next_experiment(
                output_submission_dir, self.fold_i)
            shutil.rmtree(output_submission_dir)
            self._update_df_scores(df_scores, self.fold_i, False)
            tune.report(accuracy=df_scores.loc['valid', self.score_names[0]])

        self.converted_hyperparams_ = dict()

        for h in self.hyperparameters:
            print(h.values)
            self.converted_hyperparams_[h.name] = tune.choice(h.values)

        self.current_dir = os.getcwd()
        analysis = tune.run(
        easy_objective,
        metric="accuracy",
        mode="max",
        num_samples=self.n_iter,
        name="hebo_exp_with_warmstart",
        config=self.converted_hyperparams_)

        print("analysis", analysis.results_df)
        summary_fname = os.path.join(hyperopt_output_path, 'summary.csv')
        analysis.results_df.to_csv(summary_fname)



    def run(self, n_iter, test):
        # Create hyperopt output directory

        mean = 0
        hyperopt_output_path = os.path.join(
            self.submission_dir, 'hyperopt_output')
        if not os.path.exists(hyperopt_output_path):
            os.makedirs(hyperopt_output_path)

        if isinstance(self.engine, TuneEngine):
            self.n_iter = n_iter
            self.run_tune(hyperopt_output_path)
            return

        for i_iter in range(n_iter):
            # Getting new hyperparameter values from engine
            fold_i, next_value_indices =\
                self.engine.next_hyperparameter_indices(
                    self.df_scores_, len(self.cv), self.problem)
            # Updating hyperparameters
            for h, i in zip(self.hyperparameters, next_value_indices):
                h.default_index = i
            # Writing submission files with new hyperparameter values
            output_submission_dir = mkdtemp()
            write_hyperparameters(
                self.submission_dir, output_submission_dir,
                self.hypers_per_workflow_element)
            # Calling the training script.

            df_scores = self._run_next_experiment(
                output_submission_dir, fold_i)
            self.engine.pass_feedback(fold_i, len(self.cv), df_scores, self.problem.score_types[0].name)
            self._update_df_scores(df_scores, fold_i, test)
            shutil.rmtree(output_submission_dir)
            print(f'Done {i_iter + 1} / {n_iter}.')
        self._make_and_save_summary(hyperopt_output_path)
        scores_columns = ['valid_' + name for name in self.score_names]
        for score in scores_columns:
            self.df_scores_[score + "_max"] = self.df_scores_[score].rolling(n_iter, min_periods=1).max()


def init_hyperopt(ramp_kit_dir, ramp_submission_dir, submission,
                  engine_name, data_label, label):
    problem = assert_read_problem(ramp_kit_dir)
    if data_label is None:
        hyperopt_submission = submission + '_hyperopt'
    else:
        hyperopt_submission = submission + '_' + data_label + '_hyperopt' \
            if not label else submission + '_' + data_label + '_' + engine_name + '_hyperopt'
    hyperopt_submission_dir = os.path.join(
        ramp_submission_dir, hyperopt_submission)
    submission_dir = os.path.join(
        ramp_submission_dir, submission)
    if os.path.exists(hyperopt_submission_dir):
        shutil.rmtree(hyperopt_submission_dir)
    shutil.copytree(submission_dir, hyperopt_submission_dir)
    hyperparameters = parse_all_hyperparameters(
        hyperopt_submission_dir, problem.workflow)
    if engine_name == 'random':
        engine = RandomEngine(hyperparameters)
    elif engine_name == 'hebo_cv':
        engine = HEBOCVEngine(hyperparameters)
    elif engine_name == "hebo_ind":
        engine = HEBOINDEngine(hyperparameters)
    elif engine_name == "optuna":
        engine = OptunaEngine(hyperparameters)
    elif engine_name == "skopt":
        engine = SKOptEngine(hyperparameters)
    elif engine_name == "tune":
        engine = TuneEngine()
    else:
        raise ValueError('{} is not a valid engine name'.format(engine_name))
    hyperparameter_experiment = HyperparameterOptimization(
        hyperparameters, engine, ramp_kit_dir,
        hyperopt_submission_dir, data_label)

    return hyperparameter_experiment


def run_hyperopt(ramp_kit_dir, ramp_data_dir, ramp_submission_dir, data_label,
                 submission, engine_name, n_iter, save_best, test, label):
    hyperparameter_experiment = init_hyperopt(
        ramp_kit_dir, ramp_submission_dir, submission, engine_name, data_label, label)
    hyperparameter_experiment.run(n_iter, test)
    if not save_best:
        shutil.rmtree(hyperparameter_experiment.submission_dir)
