# coding: utf-8
"""
Utilities to manage the submissions
"""
import os
import time
from collections import OrderedDict

import numpy as np
import cloudpickle as pickle

from .io import save_y_pred, set_state, print_submission_exception
from .combine import get_score_cv_bags
from .pretty_print import print_title, print_df_scores, print_warning
from .scoring import score_matrix, round_df_scores, score_matrix_from_scores


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


def train_test_submission(problem, module_path, X_train, y_train, X_test,
                          is_pickle, save_y_preds, output_path,
                          model_name='model.pkl', train_is=None):
    """Train and test submission, on cv fold if train_is not none.

    Parameters
    ----------
    problem : problem object
        imp.loaded from problem.py
    module_path : str
        the path of the submission, typically submissions/<submission_name>
    X_train : a list of training instances
        returned by problem.get_train_data
    y_train : a list of training ground truth
        returned by problem.get_train_data
    X_test : a list of testing instances or None
        returned by problem.get_test_data
    is_pickle : boolean
        True if the model should be pickled
    output_path : str
        the path into which the model will be pickled
    save_y_preds : boolean
        True if predictions should be written in files
    model_name : str (default='model.pkl')
        the file name of the pickled workflow
    train_is : a list of integers (default=None)
        training indices from the cross-validation fold, if None, train
        on full set
    Returns
    -------
    a tuple of the form ((y_pred_train, y_pred_test),
                         (train_time, valid_time, test_time))

    y_pred_train : a list of predictions
        on the training (train_train and train_valid) set
    y_pred_test : a list of predictions
        on the test set
    train_time : duration in seconds for training
    valid_time : duration in seconds for validation
    test_time : duration in seconds for testing
    """
    # Train
    t0 = time.time()
    try:
        trained_workflow = problem.workflow.train_submission(
            module_path, X_train, y_train, train_is=train_is)
    except Exception:
        print_submission_exception(save_y_preds, output_path)
        set_state('training_error', save_y_preds, output_path)
        exit(1)
    train_time = time.time() - t0
    set_state('trained', save_y_preds, output_path)
    if is_pickle:
        trained_workflow = pickle_model(
            output_path, trained_workflow, model_name)

    # Validate
    t0 = time.time()
    try:
        y_pred_train = problem.workflow.test_submission(
            trained_workflow, X_train)
    except Exception:
        print_submission_exception(save_y_preds, output_path)
        set_state('validating_error', save_y_preds, output_path)
        exit(1)
    valid_time = time.time() - t0
    set_state('validated', save_y_preds, output_path)

    # Test
    t0 = time.time()
    try:
        if X_test is None:
            y_pred_test = None
        else:
            y_pred_test = problem.workflow.test_submission(
                trained_workflow, X_test)
    except Exception:
        print_submission_exception(save_y_preds, output_path)
        set_state('testing_error', save_y_preds, output_path)
        exit(1)
    test_time = time.time() - t0
    set_state('tested', save_y_preds, output_path)

    return (y_pred_train, y_pred_test), (train_time, valid_time, test_time)


def run_submission_on_cv_fold(problem, module_path, X_train, y_train,
                              X_test, y_test, score_types,
                              is_pickle, save_y_preds, fold_output_path,
                              fold, ramp_data_dir):
    """Run submission, compute and return predictions and scores on cv.

    Parameters
    ----------
    problem : problem object
        imp.loaded from problem.py
    module_path : str
        the path of the submission, typically submissions/<submission_name>
    X_train : a list of training instances
        returned by problem.get_train_data
    y_train : a list of training ground truth
        returned by problem.get_train_data
    X_train : a list of testing instances
        returned by problem.get_test_data
    y_test : a list of testing ground truth or None
        returned by problem.get_test_data
    score_types : a list of score types
        problem.score_types
    is_pickle : boolean
        True if the model should be pickled
    save_y_preds : boolean
        True if predictions should be written in files
    fold_output_path : str
        the path into which the model will be pickled
    fold : pair of lists of integers
        (train_is, valid_is) generated by problem.get_cv
    ramp_data_dir : str
        the directory of the data
    Returns
    -------
    predictions_train_valid : instance of Predictions
        on the validation set
    predictions_test : instance of Predictions
        on the test set
    df_scores : pd.DataFrame
        table of scores (rows = train/valid/test steps, columns = scores)
    """
    train_is, valid_is = fold
    pred, timing = train_test_submission(
        problem, module_path, X_train, y_train, X_test, is_pickle,
        save_y_preds, fold_output_path, train_is=train_is)
    y_pred_train, y_pred_test = pred
    train_time, valid_time, test_time = timing

    predictions_train_train = problem.Predictions(
        y_pred=y_pred_train[train_is])
    ground_truth_train_train = problem.Predictions(
        y_true=y_train[train_is])
    predictions_train_valid = problem.Predictions(
        y_pred=y_pred_train[valid_is])
    ground_truth_train_valid = problem.Predictions(
        y_true=y_train[valid_is])
    if y_test is not None:
        predictions_test = problem.Predictions(y_pred=y_pred_test)
        ground_truth_test = problem.Predictions(y_true=y_test)
        if save_y_preds:
            save_y_pred(
                problem, y_pred_train, data_path=ramp_data_dir,
                output_path=fold_output_path, suffix='train')
            if y_test is not None:
                save_y_pred(
                    problem, y_pred_test, data_path=ramp_data_dir,
                    output_path=fold_output_path, suffix='test')
            with open(os.path.join(fold_output_path, 'train_time'), 'w') as fd:
                fd.write(str(train_time))
            with open(os.path.join(fold_output_path, 'valid_time'), 'w') as fd:
                fd.write(str(valid_time))
            with open(os.path.join(fold_output_path, 'test_time'), 'w') as fd:
                fd.write(str(test_time))
        df_scores = score_matrix(
            score_types,
            ground_truth=OrderedDict([('train', ground_truth_train_train),
                                      ('valid', ground_truth_train_valid),
                                      ('test', ground_truth_test)]),
            predictions=OrderedDict([('train', predictions_train_train),
                                     ('valid', predictions_train_valid),
                                     ('test', predictions_test)]),
        )
        set_state('scored', save_y_preds, fold_output_path)
        return predictions_train_valid, predictions_test, df_scores

    else:
        if save_y_preds:
            save_y_pred(
                problem, y_pred_train, data_path=ramp_data_dir,
                output_path=fold_output_path, suffix='train')
            with open(os.path.join(fold_output_path, 'train_time'), 'w') as fd:
                fd.write(str(train_time))
            with open(os.path.join(fold_output_path, 'valid_time'), 'w') as fd:
                fd.write(str(valid_time))
        df_scores = score_matrix(
            score_types,
            ground_truth=OrderedDict([('train', ground_truth_train_train),
                                      ('valid', ground_truth_train_valid)]),
            predictions=OrderedDict([('train', predictions_train_train),
                                     ('valid', predictions_train_valid)]),
        )
        set_state('scored', save_y_preds, fold_output_path)
        return predictions_train_valid, None, df_scores


def run_submission_on_full_train(problem, module_path, X_train, y_train,
                                 X_test, y_test, score_types,
                                 is_pickle, save_y_preds, output_path,
                                 ramp_data_dir):
    """Run submission, compute predictions, and print scores on full train.

    Parameters
    ----------
    problem : problem object
        imp.loaded from problem.py
    module_path : str
        the path of the submission, typically submissions/<submission_name>
    X_train : a list of training instances
        returned by problem.get_train_data
    y_train : a list of training ground truth
        returned by problem.get_train_data
    X_test : a list of testing instances or None
        returned by problem.get_test_data
    y_test : a list of testing ground truth or None
        returned by problem.get_test_data
    score_types : a list of score types
        problem.score_types
    is_pickle : boolean
        True if the model should be pickled
    save_y_preds : boolean
        True if predictions should be written in files
    output_path : str
        the path into which the model will be pickled
    ramp_data_dir : str
        the directory of the data
    """
    (y_pred_train, y_pred_test), _ = train_test_submission(
        problem, module_path, X_train, y_train, X_test, is_pickle,
        save_y_preds, output_path, model_name='retrained_model.pkl')
    predictions_train = problem.Predictions(y_pred=y_pred_train)
    ground_truth_train = problem.Predictions(y_true=y_train)
    if y_test is not None:
        predictions_test = problem.Predictions(y_pred=y_pred_test)
        ground_truth_test = problem.Predictions(y_true=y_test)

        df_scores = score_matrix(
            score_types,
            ground_truth=OrderedDict([('train', ground_truth_train),
                                      ('test', ground_truth_test)]),
            predictions=OrderedDict([('train', predictions_train),
                                     ('test', predictions_test)]),
        )
        df_scores_rounded = round_df_scores(df_scores, score_types)
        print_df_scores(df_scores_rounded, score_types, indent='\t')

        if save_y_preds:
            save_submissions(
                problem, y_pred_train, data_path=ramp_data_dir,
                output_path=output_path, suffix='retrain_train')
            save_submissions(
                problem, y_pred_test, data_path=ramp_data_dir,
                output_path=output_path, suffix='retrain_test')
    else:
        df_scores = score_matrix(
            score_types,
            ground_truth=OrderedDict([('train', ground_truth_train)]),
            predictions=OrderedDict([('train', predictions_train)]),
        )
        df_scores_rounded = round_df_scores(df_scores, score_types)
        print_df_scores(df_scores_rounded, score_types, indent='\t')

        if save_y_preds:
            save_submissions(
                problem, y_pred_train, data_path=ramp_data_dir,
                output_path=output_path, suffix='retrain_train')


def bag_submissions(problem, cv, y_train, y_test, predictions_valid_list,
                    predictions_test_list, training_output_path,
                    ramp_data_dir='.', score_type_index=0,
                    save_y_preds=False, score_table_title='Bagged scores',
                    score_f_name_prefix=''):
    """CV-bag trained submission.

    Parameters
    ----------
    problem : problem object
        imp.loaded from problem.py
    cv : cross validation object
        coming from get_cv of problem.py
    y_train : a list of training ground truth
        returned by problem.get_train_data
    y_test : a list of testing ground truth or None
        returned by problem.get_test_data
    predictions_valid_list : list of Prediction objects
        returned by run_submission_on_cv_fold
    predictions_test_list : list of Prediction objects or None
        returned by run_submission_on_cv_fold
    training_output_path : str
        submissions/<submission>/training_output
    ramp_data_dir : str
        the directory of the data
    score_type_index : int
        the score type on which we bag
    save_y_preds : boolean
        True if predictions should be written in files
    score_table_title : str
    score_f_name_prefix : str
    """
    print_title('----------------------------')
    print_title(score_table_title)
    print_title('----------------------------')
    valid_is_list = [valid_is for (train_is, valid_is) in cv]
    ground_truths_train = problem.Predictions(y_true=y_train)
    score_type = problem.score_types[score_type_index]
    bagged_valid_predictions, bagged_valid_scores =\
        get_score_cv_bags(
            score_type, predictions_valid_list,
            ground_truths_train, test_is_list=valid_is_list)
    if y_test is not None:
        ground_truths_test = problem.Predictions(y_true=y_test)
        bagged_test_predictions, bagged_test_scores = get_score_cv_bags(
            score_type, predictions_test_list, ground_truths_test)

        df_scores = score_matrix_from_scores(
            [score_type], ['valid', 'test'],
            [[bagged_valid_scores[-1]], [bagged_test_scores[-1]]])
        df_scores_rounded = round_df_scores(df_scores, [score_type])
        print_df_scores(df_scores_rounded, [score_type], indent='\t')

        if save_y_preds:
            # y_pred_bagged_train.csv contains _out of sample_ (validation)
            # predictions, but not for all points (contains nans)
            save_submissions(
                problem, bagged_valid_predictions.y_pred,
                data_path=ramp_data_dir, output_path=training_output_path,
                suffix='{}_bagged_train'.format(score_f_name_prefix))
            save_submissions(
                problem, bagged_test_predictions.y_pred,
                data_path=ramp_data_dir, output_path=training_output_path,
                suffix='{}_bagged_test'.format(score_f_name_prefix))
            # also save the partial combined scores (CV bag learning curves)
            bagged_train_valid_scores_f_name = os.path.join(
                training_output_path,
                '{}_bagged_valid_scores.csv'.format(score_f_name_prefix))
            np.savetxt(bagged_train_valid_scores_f_name, bagged_valid_scores)
            bagged_test_scores_f_name = os.path.join(
                training_output_path,
                '{}_bagged_test_scores.csv'.format(score_f_name_prefix))
            np.savetxt(bagged_test_scores_f_name, bagged_test_scores)
    else:
        df_scores = score_matrix_from_scores(
            [score_type], ['valid'],
            [[bagged_valid_scores[-1]]])
        df_scores_rounded = round_df_scores(df_scores, [score_type])
        print_df_scores(df_scores_rounded, [score_type], indent='\t')

        if save_y_preds:
            # y_pred_bagged_train.csv contains _out of sample_ (validation)
            # predictions, but not for all points (contains nans)
            save_submissions(
                problem, bagged_valid_predictions.y_pred,
                data_path=ramp_data_dir, output_path=training_output_path,
                suffix='{}_bagged_train'.format(score_f_name_prefix))
            # also save the partial combined scores (CV bag learning curves)
            bagged_train_valid_scores_f_name = os.path.join(
                training_output_path,
                '{}_bagged_valid_scores.csv'.format(score_f_name_prefix))
            np.savetxt(bagged_train_valid_scores_f_name, bagged_valid_scores)


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
