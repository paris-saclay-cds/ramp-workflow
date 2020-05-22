# coding: utf-8
"""
Utilities to manage the submissions
"""
import os
import time
import pickle
from collections.abc import Iterable
from collections import OrderedDict

import pandas as pd
import cloudpickle

from .io import save_y_pred, set_state, print_submission_exception
from .combine import get_score_cv_bags
from .pretty_print import print_title, print_df_scores, print_warning
from .scoring import score_matrix, round_df_scores, reorder_df_scores


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
                          is_pickle, save_output, output_path,
                          model_name, train_is):
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
    save_output : boolean
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
            module_path, X_train, y_train, train_is)
    except Exception:
        print_submission_exception(save_output, output_path)
        set_state('training_error', save_output, output_path)
        exit(1)
    train_time = time.time() - t0
    set_state('trained', save_output, output_path)
    if is_pickle:
        trained_workflow = pickle_model(
            output_path, trained_workflow, model_name)

    # Validate
    t0 = time.time()
    try:
        y_pred_train = problem.workflow.test_submission(
            trained_workflow, X_train)
    except Exception:
        print_submission_exception(save_output, output_path)
        set_state('validating_error', save_output, output_path)
        exit(1)
    valid_time = time.time() - t0
    set_state('validated', save_output, output_path)

    # Test
    t0 = time.time()
    try:
        if X_test is None:
            y_pred_test = None
        else:
            y_pred_test = problem.workflow.test_submission(
                trained_workflow, X_test)
    except Exception:
        print_submission_exception(save_output, output_path)
        set_state('testing_error', save_output, output_path)
        exit(1)
    test_time = time.time() - t0
    set_state('tested', save_output, output_path)

    return (y_pred_train, y_pred_test), (train_time, valid_time, test_time)


def run_submission_on_cv_fold(problem, module_path, fold, X_train,
                              y_train, X_test=None, y_test=None,
                              is_pickle=False, save_output=False,
                              fold_output_path='.', ramp_data_dir='.'):
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
    is_pickle : boolean
        True if the model should be pickled
    save_output : boolean
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
    score_types = problem.score_types
    train_is, valid_is = fold
    pred, timing = train_test_submission(
        problem, module_path, X_train, y_train, X_test, is_pickle,
        save_output, fold_output_path, 'model.pkl', train_is)
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
        if save_output:
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
        df_scores['time'] = [train_time, valid_time, test_time]
        set_state('scored', save_output, fold_output_path)
        return predictions_train_valid, predictions_test, df_scores

    else:
        if save_output:
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
        df_scores['time'] = [train_time, valid_time]
        set_state('scored', save_output, fold_output_path)
        return predictions_train_valid, None, df_scores


def run_submission_on_full_train(problem, module_path, X_train, y_train,
                                 X_test, y_test, score_types,
                                 is_pickle, save_output, output_path,
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
    save_output : boolean
        True if predictions should be written in files
    output_path : str
        the path into which the model will be pickled
    ramp_data_dir : str
        the directory of the data
    """
    (y_pred_train, y_pred_test), _ = train_test_submission(
        problem, module_path, X_train, y_train, X_test, is_pickle,
        save_output, output_path, 'retrained_model.pkl', None)
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
        print_df_scores(df_scores_rounded, indent='\t')

        if save_output:
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
        print_df_scores(df_scores_rounded, indent='\t')

        if save_output:
            save_submissions(
                problem, y_pred_train, data_path=ramp_data_dir,
                output_path=output_path, suffix='retrain_train')


def bag_submissions(problem, cv, y_train, y_test, predictions_valid_list,
                    predictions_test_list, training_output_path,
                    ramp_data_dir='.', score_type_index=0,
                    save_output=False, score_table_title='Bagged scores',
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
    score_type_index : int or None.
        The score type on which we bag. If None, all scores will be computed.
    save_output : boolean
        True if predictions should be written in files
    score_table_title : str
    score_f_name_prefix : str
    """
    print_title('----------------------------')
    print_title(score_table_title)
    print_title('----------------------------')
    score_type_index = (slice(None) if score_type_index is None
                        else score_type_index)
    score_types = problem.score_types[score_type_index]
    score_types = (
        [score_types] if not isinstance(score_types, Iterable)
        else score_types)

    # placeholder to store the scores and predictions
    bagged_scores = {}
    scoring_step = ['valid', 'test'] if y_test is not None else ['valid']
    for step in scoring_step:
        # Get either the training or testing infomation depending of the step
        pred_list = (predictions_valid_list if step == 'valid'
                     else predictions_test_list)
        y_step = y_train if step == 'valid' else y_test
        gt_list = problem.Predictions(y_true=y_step)
        # indices of the validation set or all sample for the testing set
        test_idx = ([valid_is for (train_is, valid_is) in cv]
                    if step == 'valid' else None)
        score_dict = {}
        for st in score_types:
            pred, scores = get_score_cv_bags(
                st, pred_list, gt_list, test_is_list=test_idx)
            score_dict[st.name] = {
                key: val for key, val in enumerate(scores)}
        bagged_scores[step] = score_dict
        # the predictions will always be the same for all score and we store
        # only a single instance
        if save_output:
            save_submissions(
                problem, pred.y_pred, data_path=ramp_data_dir,
                output_path=training_output_path,
                suffix='{}_bagged_{}'.format(score_f_name_prefix, step)
            )

    df_scores = pd.concat({step: pd.DataFrame(scores)
                           for step, scores in bagged_scores.items()})
    df_scores.columns = df_scores.columns.rename('score')
    df_scores.index = df_scores.index.rename(['step', 'n_bag'])
    # bagging learning curves can be plotted on this df_scores
    if save_output:
        bagged_scores_filename = os.path.join(
            training_output_path, 'bagged_scores.csv')
        df_scores.to_csv(bagged_scores_filename)

    # prepare the bagged scores which will be printed.
    highest_level = df_scores.index.get_level_values('n_bag').max()
    df_scores = df_scores.loc[(slice(None), highest_level), :]
    df_scores.index = df_scores.index.droplevel('n_bag')
    df_scores = reorder_df_scores(df_scores, score_types)
    df_scores = round_df_scores(df_scores, score_types)
    print_df_scores(df_scores, indent='\t')


def pickle_model(fold_output_path, trained_workflow, model_name='model.pkl'):
    """Pickle and reload trained workflow.

    If workflow can't be pickled, print warning and return original workflow.

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
    msg = "Warning: model can't be pickled."
    model_file = os.path.join(fold_output_path, model_name)
    try:
        with open(model_file, 'wb') as pickle_file:
            cloudpickle.dump(trained_workflow, pickle_file)
    except pickle.PicklingError as e:
        print_warning(msg)
        print_warning(e)
        return trained_workflow
    else:
        # check if dumped trained_workflow can be loaded
        try:
            with open(model_file, 'rb') as pickle_file:
                trained_workflow = cloudpickle.load(pickle_file)
        except Exception as e:
            print_warning(msg)
            print_warning(e)

    return trained_workflow
