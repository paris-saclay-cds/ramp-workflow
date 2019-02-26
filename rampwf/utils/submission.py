# coding: utf-8
"""
Utilities to manage the submissions
"""
import os
import time
import json
from collections import Iterable
from collections import OrderedDict

import numpy as np
import pandas as pd
import cloudpickle as pickle

from .io import save_y_pred, set_state, print_submission_exception
from .combine import get_score_cv_bags
from .pretty_print import print_title, print_df_scores, print_warning
from .scoring import score_matrix, round_df_scores, reorder_df_scores

from . import testing

def timeit(func):
    # TODO: move utility functions to another module
    def run(*args, **kwargs):
        t = time.time()
        ret = func(*args, **kwargs)
        t = time.time() - t
        return t, ret
    return run


class _Fold(object):

    def __init__(self, id, cv_fold, output_dir, save_info=False, pickle_model=False,
                 save_func=np.savez_compressed):
        self.id = id
        self._cv_fold = cv_fold
        self._info = None
        self._save_info = save_info
        self._pickle_model = pickle_model
        self._save_func = save_func
        self._state = None
        self._model = None
        self.predictions_train = None
        self.predictions_valid = None
        self.predictions_test = None
        self._basedir = os.path.join(output_dir, '{}'.format(id))
        if save_info:
            os.makedirs(self._basedir, exist_ok=True)

    def __iter__(self):
        # ensure a behavior similar to cv_fold's
        for entry in self._cv_fold:
            yield entry

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state
        if self._save_info:
            with open(os.path.join(self._basedir, 'state.txt'), 'w') as fp:
                fp.write(new_state)

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, info):
        self._info = info
        if self._save_info:
            with open(os.path.join(self._basedir, "info.json"), "w") as fp:
                json.dump(info, fp)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        if self._pickle_model:
            self._model = pickle_model(self._basedir, model, 'model.pkl')

    @property
    def predictions(self):
        return self.predictions_train, self.predictions_valid, self.predictions_test

    @predictions.setter
    def predictions(self, predictions):
        self.predictions_train, self.predictions_valid, self.predictions_test = predictions
        if self._save_info:
            for suffix, pred in zip(("train", 'valid', 'test'), self.predictions):
                if pred is None:
                    continue
                fname = os.path.join(self._basedir, 'y_pred_{}'.format(suffix))
                self._save_func(fname, y_pred=pred.y_pred)

    @property
    def scores(self):
        return self._scores

    @scores.setter
    def scores(self, scores):
        self._scores = scores
        if self._save_info:
            fname = os.path.join(self._basedir, "scores.csv")
            scores.to_csv(fname)

    def show_scores(self, score_types):
        scores_rounded = round_df_scores(self.scores, score_types)
        print_df_scores(scores_rounded, indent='\t')

    def __str__(self):
        return('{}'.format(self.id))


class Submission(object):

    def __init__(self, ramp_kit_dir, submission='starting_kit',
                 data_dir=''):
        """
        Class to run a train and test a submission.
        This creates a submission for code locted under:
        `ramp_kit_dir`/sbmissions/`submission`

        Parameters:
        -----------
        ramp_kit_dir: str
            path to a directory containing a ramp kit. This directory is
            expected to have a sub-directory called 'submissions' that
            contains user submissions as well as a python module called
            'problem.py'.

        submission: str
            submission name. Should be the name of a sub-directory within
            '$ramp_kit_dir/submissions'.

        data_dir: str
            path to a directory containing data, Default: '', means that
            data is located within the ramp_kit_dir (under a directory called
            'data').
        """
        self._problem = testing.Problem(ramp_kit_dir, name='problem', data_dir=data_dir)
        self._basedir = os.path.join(ramp_kit_dir, "submissions", submission)

    def show_title(self):
        """Show ramp kit title
        """
        print('----------------------------------')
        print(self._problem.title)
        print('----------------------------------')

    def _show_scores(self, df_scores_list):
        """Show cross-validation scores.

        Parameters:
        -----------
        df_scores_list: list
            list of DataFrames representing cross-validation scores.
        """
        print_title('----------------------------')
        print_title('Mean CV scores')
        print_title('----------------------------')
        df_mean_scores = testing.mean_score_matrix(df_scores_list, self._problem.score_types)
        print_df_scores(df_mean_scores, indent='\t')

    def _make_train_valid_data(self, fold):
        """Make tain and validation data and targets using `fold`.

        Parameters:
        -----------
        fold: iterable
            an 2-element iterable containig lists for train and validation
            indices respectively.

        Returns:
        --------
        X_train, y_train, X_valid, y_valid:
            train/validation data and targets.
        """
        train_indices, valid_indices = fold
        X, y = self._problem.train_data
        X_train, y_train = X.iloc[train_indices], y[train_indices]
        X_valid, y_valid = X.iloc[valid_indices], y[valid_indices]
        return X_train, y_train, X_valid, y_valid

    @timeit
    def _train(self, X_train, y_train):
        """Train workflow on the provided data

        Parameters:
        -----------
        X_train: iterable
            train data samples
        y_train: iterable
            train data targers

        Rturns:
        -------
        model:
            trained workflow
        """
        model = self._problem.workflow.train_submission(self._basedir,
                                                        X_train,
                                                        y_train)
        return model

    @timeit
    def _predict(self, model, x):
        """Run workflow prediction on data.

        Parameters:
        -----------
        model:
            trained workflow
        x: iterable
            data used for prediction

        Rturns:
        -------
        predictions: Predictions
            output predictions
        """
        y_pred = self._problem.workflow.test_submission(model, x)
        return self._problem.Predictions(y_pred=y_pred)

    def _run_submission_on_cv_fold(self, fold):
        """Run submission of fold, compute and return predictions and scores.

        Parameters
        ----------
        fold: iterable
            an 2-element iterable containig lists for train and validation
            indices respectively.

        Returns:
        --------
        valid_pred, test_pred, scores, models: tuple
            validation predictions, test predictions (if test data is
            available) and scores.
        """
        X_train, y_train, X_valid, y_valid = self._make_train_valid_data(fold)
        X_test, y_test = self._problem.test_data
        train_time, model = self._train(X_train, y_train)
        fold.state = 'trained'
        fold.model = model

        pred_train_time, predictions_train = self._predict(model, X_train)
        pred_valid_time, predictions_valid = self._predict(model, X_valid)
        y_true_train = self._problem.Predictions(y_true=y_train)
        y_true_valid = self._problem.Predictions(y_true=y_valid)
        fold.state = 'validated'

        ground_truth = OrderedDict([('train', y_true_train),
                                    ('valid', y_true_valid)])
        predictions = OrderedDict([('train', predictions_train),
                                   ('valid', predictions_valid)])

        if y_test is not None:
            y_true_test = self._problem.Predictions(y_true=y_test)
            pred_test_time, predictions_test = self._predict(model, X_test)
            ground_truth['test'] = y_true_test
            predictions['test'] = predictions_test
            fold.state = 'tested'
        else:
            pred_test_time, predictions_test = None, None
        info = {'train_time' : train_time,
                'pred_train_time' : pred_train_time,
                'pred_valid_time' : pred_valid_time,
                'pred_test_time' : pred_test_time,
            }
        fold.info = info
        fold.predictions = (predictions_train, predictions_valid,
                            predictions_test)
        df_scores = score_matrix(self._problem.score_types,
                                 ground_truth=ground_truth,
                                 predictions=predictions)
        fold.scores = df_scores
        fold.state = 'scored'
        return predictions['valid'], predictions.get('test'), df_scores


    def run_cross_validation(self, save_info=False, pickle_models=False):
        """Run cross-validation for submission.

        Parameters:
        -----------
        save_info: bool, default: False
            if True, save fold-related information (training and evaluation
            times, prediction arrays, scores, etc.). For each fold $i, this
            iformation is saved at:
            $ramp_kit/submissions/$submission_name/training_output/fold_$i

        pickle_models: bool, default: False
            If true save trained model for each fold. Pickle location is the
            same as for `save_info`.

        Returns:
        --------
        scores_list: list
            list of DataFrames containing scores for each cross validation
            fold.
        """
        self.show_title()
        cv = self._problem.cv
        df_scores_list = []
        predictions_valid = []
        predictions_test = []

        output_dir = os.path.join( self._basedir, 'training_output')
        for i, cv_fold in enumerate(cv):
            fold_i = 'fold_{}'.format(i)
            fold = _Fold(fold_i, cv_fold, output_dir, save_info, pickle_models)
            print(fold)
            pred_valid, pred_test, df_scores = self._run_submission_on_cv_fold(fold)
            df_scores_list.append(df_scores)
            predictions_valid.append(pred_valid)
            predictions_test.append(pred_test)
            fold.show_scores(self._problem.score_types)
        self._show_scores(df_scores_list)
        return df_scores_list


    def train_full_data(self):
        """train model on full data and return model"""
        X_train, y_train = self._problem.train_data
        # TODO compute and save scores
        return self._train(X_train, y_train)

    def bag_submissions(self):
        pass

def save_submissions(problem, y_pred, data_path='.', output_path='.',
                     suffix='test'):
    """Custom save a prediction vector in file to, e.g., submit to Kaggle.

    If problem.save_submission is implemented, y_pred is passed to it.
    Otherwise nothing happens (the exception is caught silently).

    Parameters
    ----------
    problem : a problem object
        loaded from problem.py, may implement save_y_pred
    y_pred : a prediction vector
        a vector of predictions to be saved
    data_path : str, (default='.')
        the directory of the ramp-kit to be tested for submission, maybe
        needed by problem.save_y_pred for, e.g., merging with an index vector
    output_path : str, (default='.')
        the directory where (typically) y_pred_<suffix>.csv will be saved
    suffix : str, (default='test')
        suffix in (typically) y_pred_<suffix>.csv, can be used in
        problem.save_y_pred to, e.g., save only test predictions
    """
    try:
        # We try using custom made problem.save_submission
        # it may need to re-read the data, e.g., for ids, so we send
        # it the data path. See, e.g.,
        # https://github.com/ramp-kits/kaggle_seguro/blob/master/problem.py
        problem.save_submission(y_pred, data_path, output_path, suffix)
    except AttributeError:
        pass

def pickle_model(fold_output_path, trained_workflow, model_name='model.pkl'):
    """Pickle and reload trained workflow.

    If workflow can't be pickled, print warning and return origina' workflow.

    Parameters
    ----------
    fold_output_path : str
        the path into which the model will be pickled
    trained_workflow : a rampwf.workflow
        the workflow to be pickled
    model_name : str (default='model.pkl')
        the file name of the pickled workflow
    Returns
    -------
    trained_workflow : a rampwf.workflow
        either the input workflow or the pickled and reloaded workflow
    """
    try:
        model_file = os.path.join(fold_output_path, model_name)
        with open(model_file, 'wb') as pickle_file:
            pickle.dump(trained_workflow, pickle_file)
        with open(model_file, 'r') as pickle_file:
            trained_workflow = pickle.load(pickle_file)
    except Exception as e:
        print_warning("Warning: model can't be pickled.")
        print_warning(e)
    return trained_workflow
