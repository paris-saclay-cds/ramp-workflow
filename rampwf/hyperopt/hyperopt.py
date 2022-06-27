"""Hyperparameter optiomization for ramp-kits."""
import re
import os
import shutil
import numpy as np
import pandas as pd

from tempfile import mkdtemp
from ..utils import (
    assert_read_problem, import_module_from_source, run_submission_on_cv_fold)

from .engines import RandomEngine

HYPERPARAMS_SECTION_START = '# RAMP START HYPERPARAMETERS'
HYPERPARAMS_SECTION_END = '# RAMP END HYPERPARAMETERS'
HYPERPARAMS_REPL_REGEX = re.compile('{}.*{}'.format(
    HYPERPARAMS_SECTION_START, HYPERPARAMS_SECTION_END), re.S)
CONST_MESSAGE = "missing module install it using pip install"


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
        if self.dtype in ['object', 'str']:
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
            if self.dtype in ['object', 'str']:
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

    def _run_next_experiment(self, module_path, fold_i):
        _, _, df_scores = run_submission_on_cv_fold(
            self.problem, module_path=module_path, fold=self.cv[fold_i],
            X_train=self.X_train, y_train=self.y_train,
            X_test=self.X_test, y_t est=self.y_test)
        return df_scores

    def _update_df_scores(self, df_scores, fold_i, test):
        row = {'fold_i': fold_i}
        for h in self.hyperparameters:
            row[h.name] = h.default
            row[h.name + '_i'] = h.default_index
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
        self.df_scores_['fold_i'] = self.df_scores_['fold_i'].astype(int)
        for h in self.hyperparameters:
            col = h.name + '_i'
            self.df_scores_[col] = self.df_scores_[col].astype(int)

    def _make_and_save_summary(self, hyperopt_output_path):
        summary_fname = os.path.join(hyperopt_output_path, 'summary.csv')
        self.df_scores_.to_csv(summary_fname)

    def _load_summary(self, hyperopt_output_path):
        summary_fname = os.path.join(hyperopt_output_path, 'summary.csv')
        self.df_scores_ = pd.read_csv(summary_fname, index_col=0)

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

    def run_tune(self, n_trials, test):
        from ray import tune
        is_lower_the_better = self.problem.score_types[0].is_lower_the_better
        engine_mode = 'min' if is_lower_the_better else 'max'

        hyperopt_output_path = os.path.join(
            self.submission_dir, 'hyperopt_output')
        if not os.path.exists(hyperopt_output_path):
            os.makedirs(hyperopt_output_path)

        config = {h.name: tune.randint(0, len(h.values))
                  for h in self.hyperparameters}
        
        def objective(config, run_params=None):
            for h in run_params['hyperparameters']:
                h.default_index = config[h.name]
            output_submission_dir = mkdtemp()
            os.chdir(run_params['current_dir'])
            write_hyperparameters(
                self.submission_dir, output_submission_dir,
                run_params['hypers_per_workflow_element'])
            # Calling the training script.
            valid_scores = np.zeros(len(self.cv))
            df_scores_list = []
            for fold_i in range(len(self.cv)):
                df_scores = self._run_next_experiment(
                    output_submission_dir, fold_i)
                sn = self.score_names[0]
                valid_scores[fold_i] = df_scores.loc['valid', sn]
                df_scores_list.append(df_scores)
            shutil.rmtree(output_submission_dir)
            tune.report(
                valid_score=valid_scores.mean(),
                df_scores_list=df_scores_list,
            )

        run_params = {
            'hyperparameters': 
                self.hyperparameters, 'current_dir': os.getcwd(),
            'submission_dir': self.submission_dir,
            'hypers_per_workflow_element': self.hypers_per_workflow_element
        }
        results = tune.run(
            tune.with_parameters(objective, run_params=run_params),
            max_concurrent_trials=1,
            metric='valid_score',
            mode=engine_mode,
            num_samples=int(n_trials / len(self.cv)),
            name=self.engine.name,
            search_alg=self.engine.ray_engine,
            config=config
        )

        for _, row in results.results_df.iterrows():
            for h in self.hyperparameters:
                h.default_index = int(row[f'config.{h.name}'])
            for fold_i, df_scores in enumerate(row['df_scores_list']):
                self._update_df_scores(df_scores, fold_i, test=test)

        summary_fname = os.path.join(hyperopt_output_path, 'summary.csv')
        self.df_scores_.to_csv(summary_fname)


    def run(self, n_trials, test, resume = False):
        # Create hyperopt output directory

        mean = 0
        hyperopt_output_path = os.path.join(
            self.submission_dir, 'hyperopt_output')
        if not os.path.exists(hyperopt_output_path):
            os.makedirs(hyperopt_output_path)

        start_iter = 0
        if resume:
            self._load_summary(hyperopt_output_path)
            start_iter = len(self.df_scores_)
        start = pd.Timestamp.now()
        for i_iter in range(start_iter, n_trials):
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
            sn = self.problem.score_types[0].name
            self.engine.pass_feedback(
                fold_i, len(self.cv), df_scores, sn)
            self._update_df_scores(df_scores, fold_i, test)
            shutil.rmtree(output_submission_dir)
            now = pd.Timestamp.now()
            eta = start + (now - start) / (i_iter + 1 - start_iter)\
                * (n_trials - start_iter)
            print(f'Done {i_iter + 1} / {n_trials} at {now}. ETA = {eta}.')
            self._make_and_save_summary(hyperopt_output_path)
        scores_columns = ['valid_' + name for name in self.score_names]
        for score in scores_columns:
            self.df_scores_[score + "_max"] = \
                self.df_scores_[score].rolling(n_trials, min_periods=1).max()

class RayEngine:
    # n_trials is only needed by zoopt at init time
    def __init__(self, engine_name, n_trials=None):
        self.name = engine_name
        if engine_name[4:] == 'zoopt':
            try:
                from ray.tune.suggest.zoopt import ZOOptSearch
                self.ray_engine = ZOOptSearch( # gets stuck
                    algo="Asracos",  # only support ASRacos currently
                    budget=n_trials,  # with grid_size it got stuck
                )
            except:
                self.raise_except("zoopt")
        if engine_name[4:] == 'ax':
            try:
                from ray.tune.suggest.ax import AxSearch
                self.ray_engine = AxSearch()
            except:
                self.raise_except("ax-platform sqlalchemy")
        elif engine_name[4:] == 'blend_search':
            try:
                from ray.tune.suggest.flaml import BlendSearch
                self.ray_engine = BlendSearch()
            except:
                self.raise_except("flaml")
        elif engine_name[4:] == 'cfo':
            try:
                from ray.tune.suggest.flaml import CFO
                self.ray_engine = CFO()
            except:
                self.raise_except("flaml")
        elif engine_name[4:] == 'skopt':
            try:
                from ray.tune.suggest.skopt import SkOptSearch
                self.ray_engine = SkOptSearch()
            except:
                self.raise_except("scikit-optimize")
        elif engine_name[4:] == 'hyperopt':
            try:
                from ray.tune.suggest.hyperopt import HyperOptSearch
                self.ray_engine = HyperOptSearch()
            except:
                self.raise_except("hyperopt")
        elif engine_name[4:] == 'bayesopt':
            try:
                from ray.tune.suggest.bayesopt import BayesOptSearch
                self.ray_engine = BayesOptSearch()
            except:
                self.raise_except("bayesian-optimization")
        elif engine_name[4:] == 'bohb':
            try:
                from ray.tune.suggest.bohb import TuneBOHB
                self.ray_engine = TuneBOHB()
            except:
                self.raise_except("hpbandster")
        elif engine_name[4:] == 'nevergrad':
            try:
                from ray.tune.suggest.nevergrad import NevergradSearch
                import nevergrad as ng
                self.ray_engine = NevergradSearch(
                    ray_engine=ng.optimizers.OnePlusOne
                )
            except:
                self.raise_except("nevergrad")
        elif engine_name[4:] == 'hebo':
            try:
                from ray.tune.suggest.hebo import HEBOSearch
                self.ray_engine = HEBOSearch()
            except:
                self.raise_except("hebo")
        elif engine_name[4:] == 'optuna':
            try:
                from ray.tune.suggest.optuna import OptunaSearch
                self.ray_engine = OptunaSearch()
            except:
                self.raise_except("optuna")
        else:
            raise ValueError(
                f'Engine {engine_name[4:]} not found in Ray Tune')

    def raise_except(message):
        raise EnvironmentError(CONST_MESSAGE + message)


def init_hyperopt(ramp_kit_dir, ramp_submission_dir, submission,
                  engine_name, data_label, label, resume, n_trials=None):
    # n_trials is only needed by ray_zoopt at init time
    problem = assert_read_problem(ramp_kit_dir)
    if data_label is None:
        hyperopt_submission = submission + '_hyperopt'
    else:
        hyperopt_submission = submission + '_' + data_label + '_hyperopt'\
            if not label else submission + '_' + data_label + '_'\
                + engine_name + '_hyperopt'
    hyperopt_submission_dir = os.path.join(
        ramp_submission_dir, hyperopt_submission)
    submission_dir = os.path.join(
        ramp_submission_dir, submission)
    if not resume:
        if os.path.exists(hyperopt_submission_dir):
            shutil.rmtree(hyperopt_submission_dir)
        shutil.copytree(submission_dir, hyperopt_submission_dir)
    hyperparameters = parse_all_hyperparameters(
        hyperopt_submission_dir, problem.workflow)
    if engine_name == 'random':
        engine = RandomEngine(hyperparameters)
    elif engine_name.startswith('ray_'):
        engine = RayEngine(engine_name, n_trials)
    else:
        raise ValueError(f'{engine_name} is not a valid engine name')
    hyperparameter_experiment = HyperparameterOptimization(
        hyperparameters, engine, ramp_kit_dir,
        hyperopt_submission_dir, data_label)

    return hyperparameter_experiment


def run_hyperopt(ramp_kit_dir, ramp_data_dir, ramp_submission_dir, data_label,
                 submission, engine_name, n_trials, save_best, test, label,
                 resume):
    hyperparameter_experiment = init_hyperopt(
        ramp_kit_dir, ramp_submission_dir, submission, engine_name,
        data_label, label, resume)
    if engine_name.startswith('ray_'):
        hyperparameter_experiment.run_tune(n_trials, test)
    else:
        hyperparameter_experiment.run(n_trials, test, resume)
    if not save_best:
        shutil.rmtree(hyperparameter_experiment.submission_dir)

