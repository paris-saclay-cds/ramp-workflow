# coding: utf-8

"""Provide utils to test ramp-kits."""
from __future__ import print_function

import os
import imp
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd

from .combine import blend_on_fold
from .io import load_y_pred, set_state
from .pretty_print import print_title, print_df_scores
from .notebook import execute_notebook, convert_notebook
from .scoring import round_df_scores, mean_score_matrix
from .submission import (bag_submissions, run_submission_on_cv_fold,
                         run_submission_on_full_train)


def load_module(name, path):
    """Load module from path. If path is directory then  it will be appended
    `name` + '.py' (if `name` doesn't end ith '.py').

    Parameters:
    -----------
    name: str
        name to give the loaded module. If path is directory, theb this
        parameter will also be used as the name of the python source file
        under `path` to load source from (appended with `.py` if necessary)
    path: str
        path to a file/folder to load module from

    Returns:
    loaded_module: module
        module loaded from source.
    """
    if os.path.isdir(path):
        source_file = name if name.endswith(".py") else name + ".py"
        path = os.path.join(path, source_file)
    loader = SourceFileLoader(name, path)
    return loader.load_module()

class Problem(object):

    def __init__(self, ramp_kit_dir, name='problem', data_dir=''):
        """

        Parameters:
        -----------0
        name: str
            name of the module file, default: 'problem'
        ramp_kit_dir: str
            name of ramp kit dir, default: 'starting_kit'
        data_dir: str
            path to a directory that contain train and test data. It should contain
            a subdirectory called 'data'. Default: '', means that 'data' is located
            under the `ramp_kit_dir`.
        """
        self._problem_module = load_module(name, ramp_kit_dir)
        self._ramp_data_dir =  data_dir

    @property
    def title(self):
        return self._problem_module.problem_title

    @property
    def train_data(self):
        X_train, y_train = self._problem_module.get_train_data(self._ramp_data_dir)
        return X_train, y_train

    @property
    def test_data(self):
        X_test, y_test = self._problem_module.get_test_data(self._ramp_data_dir)
        return X_test, y_test

    @property
    def data(self):
        X_train, y_train = self.train_data
        X_test, y_test = self.test_data
        return X_train, y_train, X_test, y_test

    @property
    def cv(self):
        X_train, y_train = self.train_data
        cv = list(self._problem_module.get_cv(X_train, y_train))
        return cv

    @property
    def score_types(self):
        return self._problem_module.score_types

    @property
    def workflow(self):
        return self._problem_module.workflow

    @property
    def Predictions(self):
        return self._problem_module.Predictions


def assert_notebook(ramp_kit_dir='.'):
    print('----------------------------')
    convert_notebook(ramp_kit_dir)
    execute_notebook(ramp_kit_dir)


def assert_read_problem(ramp_kit_dir='.'):
    # giving a random name to the module so it passes looped tests
    module_name = str(int(1000000000 * np.random.rand()))
    problem = imp.load_source(module_name,
                              os.path.join(ramp_kit_dir, 'problem.py'))
    return problem


def assert_title(ramp_kit_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    print_title('Testing {}'.format(problem.problem_title))


def assert_data(ramp_kit_dir='.', ramp_data_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    print_title('Reading train and test files from {}/data ...'.format(
        ramp_data_dir))
    X_train, y_train = problem.get_train_data(path=ramp_data_dir)
    X_test, y_test = problem.get_test_data(path=ramp_data_dir)
    return X_train, y_train, X_test, y_test


def assert_cv(ramp_kit_dir='.', ramp_data_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    X_train, y_train = problem.get_train_data(path=ramp_data_dir)
    print_title('Reading cv ...')
    cv = list(problem.get_cv(X_train, y_train))
    return cv


def assert_score_types(ramp_kit_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    score_types = problem.score_types
    return score_types


def assert_submission(ramp_kit_dir='.', ramp_data_dir='.',
                      ramp_submission_dir='submissions',
                      submission='starting_kit', is_pickle=False,
                      save_output=False, retrain=False):
    """Helper to test a submission from a ramp-kit.

    Parameters
    ----------
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit to be tested for submission.
    ramp_data_dir : str, default='.'
        The directory of the data.
    ramp_submission_dir : str, default='./submissions'
        The directory of the submissions.
    submission : str, default='starting_kit'
        The name of the submission to be tested.
    is_pickle : bool, default is False
        Whether to pickle the model or not.
    save_y_preds : bool, default is False
        Whether to store the predictions.
    retrain : bool, default is False
        Whether to train the model on the full training set and test on the
        test set.
    """
    problem = assert_read_problem(ramp_kit_dir)
    assert_title(ramp_kit_dir)
    X_train, y_train, X_test, y_test = assert_data(ramp_kit_dir, ramp_data_dir)
    cv = assert_cv(ramp_kit_dir, ramp_data_dir)
    score_types = assert_score_types(ramp_kit_dir)

    # module_path = os.path.join(ramp_kit_dir, 'submissions', submission)
    submission_path = os.path.join(ramp_submission_dir, submission)
    print_title('Training {} ...'.format(submission_path))

    training_output_path = ''
    if is_pickle or save_output:
        # creating <submission_path>/<submission>/training_output dir
        training_output_path = os.path.join(submission_path, 'training_output')
        if not os.path.exists(training_output_path):
            os.makedirs(training_output_path)

    # saving predictions for CV bagging after the CV loop
    predictions_valid_list = []
    predictions_test_list = []
    df_scores_list = []

    for fold_i, fold in enumerate(cv):
        fold_output_path = ''
        if is_pickle or save_output:
            # creating <submission_path>/<submission>/training_output/fold_<i>
            fold_output_path = os.path.join(
                training_output_path, 'fold_{}'.format(fold_i))
            if not os.path.exists(fold_output_path):
                os.makedirs(fold_output_path)
        print_title('CV fold {}'.format(fold_i))

        predictions_valid, predictions_test, df_scores = \
            run_submission_on_cv_fold(
                problem, submission_path, X_train, y_train, X_test, y_test,
                score_types, is_pickle, save_output, fold_output_path,
                fold, ramp_data_dir)
        if save_output:
            filename = os.path.join(fold_output_path, 'scores.csv')
            df_scores.to_csv(filename)
        df_scores_rounded = round_df_scores(df_scores, score_types)
        print_df_scores(df_scores_rounded, indent='\t')

        # saving predictions for CV bagging after the CV loop
        df_scores_list.append(df_scores)
        predictions_valid_list.append(predictions_valid)
        predictions_test_list.append(predictions_test)

    print_title('----------------------------')
    print_title('Mean CV scores')
    print_title('----------------------------')
    df_mean_scores = mean_score_matrix(df_scores_list, score_types)
    print_df_scores(df_mean_scores, indent='\t')

    if retrain:
        # We retrain on the full training set
        print_title('----------------------------')
        print_title('Retrain scores')
        print_title('----------------------------')
        run_submission_on_full_train(
            problem, submission_path, X_train, y_train, X_test, y_test,
            score_types, is_pickle, save_output, training_output_path,
            ramp_data_dir)
    bag_submissions(
        problem, cv, y_train, y_test, predictions_valid_list,
        predictions_test_list, training_output_path,
        ramp_data_dir=ramp_data_dir, score_type_index=None,
        save_output=save_output)


def blend_submissions(submissions, ramp_kit_dir='.', ramp_data_dir='.',
                      ramp_submission_dir='.', save_output=False,
                      min_improvement=0.0):
    problem = assert_read_problem(ramp_kit_dir)
    print_title('Blending {}'.format(problem.problem_title))
    X_train, y_train, X_test, y_test = assert_data(ramp_kit_dir, ramp_data_dir)
    cv = assert_cv(ramp_kit_dir, ramp_data_dir)
    valid_is_list = [valid_is for (train_is, valid_is) in cv]
    score_types = assert_score_types(ramp_kit_dir)
    contributivitys = np.zeros(len(submissions))

    combined_predictions_valid_list = []
    foldwise_best_predictions_valid_list = []
    combined_predictions_test_list = []
    foldwise_best_predictions_test_list = []
    for fold_i, valid_is in enumerate(valid_is_list):
        print_title('CV fold {}'.format(fold_i))
        ground_truths_valid = problem.Predictions(y_true=y_train[valid_is])
        predictions_valid_list = []
        predictions_test_list = []
        for submission in submissions:
            module_path = os.path.join(ramp_submission_dir, submission)
            training_output_path = os.path.join(module_path, 'training_output')
            fold_output_path = os.path.join(
                training_output_path, 'fold_{}'.format(fold_i))
            y_pred_train = load_y_pred(
                problem, data_path=ramp_data_dir,
                input_path=fold_output_path, suffix='train')
            y_pred_test = load_y_pred(
                problem, data_path=ramp_data_dir,
                input_path=fold_output_path, suffix='test')
            predictions_valid = problem.Predictions(
                y_pred=y_pred_train[valid_is])
            predictions_valid_list.append(predictions_valid)
            predictions_test = problem.Predictions(y_pred=y_pred_test)
            predictions_test_list.append(predictions_test)

        best_index_list = blend_on_fold(
            predictions_valid_list, ground_truths_valid, score_types[0],
            min_improvement=min_improvement)

        # we share a unit of 1. among the contributive submissions
        unit_contributivity = 1. / len(best_index_list)
        for i in best_index_list:
            contributivitys[i] += unit_contributivity

        combined_predictions_valid_list.append(
            problem.Predictions.combine(predictions_valid_list))
        foldwise_best_predictions_valid_list.append(predictions_valid_list[0])
        combined_predictions_test_list.append(
            problem.Predictions.combine(predictions_test_list))
        foldwise_best_predictions_test_list.append(predictions_test_list[0])

    contributivitys /= len(cv)
    contributivitys_df = pd.DataFrame()
    contributivitys_df['submission'] = np.array(submissions)
    contributivitys_df['contributivity'] = np.round(contributivitys, 3)
    contributivitys_df = contributivitys_df.reset_index()
    contributivitys_df = contributivitys_df.sort_values(
        'contributivity', ascending=False)
    print(contributivitys_df.to_string(index=False))

    training_output_path = os.path.join(ramp_kit_dir, 'training_output')
    if not os.path.exists(training_output_path):
        os.mkdir(training_output_path)
    # bagging the foldwise ensembles
    bag_submissions(
        problem, cv, y_train, y_test, combined_predictions_valid_list,
        combined_predictions_test_list, training_output_path,
        ramp_data_dir=ramp_data_dir, score_type_index=0,
        save_output=save_output, score_table_title='Combined bagged scores',
        score_f_name_prefix='foldwise_best')
    # bagging the foldwise best submissions
    bag_submissions(
        problem, cv, y_train, y_test, foldwise_best_predictions_valid_list,
        foldwise_best_predictions_test_list, training_output_path,
        ramp_data_dir=ramp_data_dir, score_type_index=0,
        save_output=save_output,
        score_table_title='Foldwise best bagged scores',
        score_f_name_prefix='combined')
