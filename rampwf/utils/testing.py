# coding: utf-8

"""Provide utils to test ramp-kits."""
from __future__ import print_function

import os
import imp
from subprocess import call
from os.path import join, abspath
from collections import OrderedDict

import pandas as pd
import numpy as np
from colored import stylize, fg, attr
import cloudpickle as pickle
from .combine import get_score_cv_bags, blend_on_fold

fg_colors = {
    'official_train': 'light_green',
    'official_valid': 'light_blue',
    'official_test': 'red',
    'train': 'dark_sea_green_3b',
    'valid': 'light_slate_blue',
    'test': 'pink_1',
    'title': 'gold_3b',
    'warning': 'grey_46',
}


def _print_title(str):
    print(stylize(str, fg(fg_colors['title']) + attr('bold')))


def _print_warning(str):
    print(stylize(str, fg(fg_colors['warning'])))


def _delete_line_from_file(f_name, line_to_delete):
    with open(f_name, "r+") as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            if line != line_to_delete:
                f.write(line)
        f.truncate()


def execute_notebook(ramp_kit_dir='.'):
    problem_name = abspath(ramp_kit_dir).split('/')[-1]
    print('Testing if the notebook can be executed')
    call(
        'jupyter nbconvert --execute {}/{}_starting_kit.ipynb '.format(
            ramp_kit_dir, problem_name) +
        '--ExecutePreprocessor.kernel_name=$IPYTHON_KERNEL ' +
        '--ExecutePreprocessor.timeout=600', shell=True)


def convert_notebook(ramp_kit_dir='.'):
    problem_name = abspath(ramp_kit_dir).split('/')[-1]
    print('Testing if the notebook can be converted to html')
    call('jupyter nbconvert --to html {}/{}_starting_kit.ipynb'.format(
        ramp_kit_dir, problem_name), shell=True)
    _delete_line_from_file(
        '{}/{}_starting_kit.html'.format(ramp_kit_dir, problem_name),
        '<link rel="stylesheet" href="custom.css">\n')


def assert_notebook(ramp_kit_dir='.'):
    print('----------------------------')
    convert_notebook(ramp_kit_dir)
    execute_notebook(ramp_kit_dir)


def assert_read_problem(ramp_kit_dir='.'):
    # giving a random name to the module so it passes looped tests
    module_name = str(int(1000000000 * np.random.rand()))
    problem = imp.load_source(module_name, join(ramp_kit_dir, 'problem.py'))
    return problem


def assert_title(ramp_kit_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    _print_title('Testing {}'.format(problem.problem_title))


def assert_data(ramp_kit_dir='.', ramp_data_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    _print_title('Reading train and test files from {}/data ...'.format(
        ramp_data_dir))
    X_train, y_train = problem.get_train_data(path=ramp_data_dir)
    X_test, y_test = problem.get_test_data(path=ramp_data_dir)
    return X_train, y_train, X_test, y_test


def assert_cv(ramp_kit_dir='.', ramp_data_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    X_train, y_train = problem.get_train_data(path=ramp_data_dir)
    _print_title('Reading cv ...')
    cv = list(problem.get_cv(X_train, y_train))
    return cv


def assert_score_types(ramp_kit_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    score_types = problem.score_types
    return score_types


def _mean_score_matrix(df_scores_list, score_types):
    u"""Construct a mean ± std score dataframe from a list of score dataframes.

    Parameters
    ----------
    df_scores_list : list of pd.DataFrame
        a list of score data frames to average
    score_types : list of score types
        a list of score types to use

    Returns
    -------
    df_scores : the mean ± std score dataframe
    """
    scores = np.array([df_scores.values for df_scores in df_scores_list])
    meanss = scores.mean(axis=0)
    stdss = scores.std(axis=0)
    # we use unicode no break space so split in _print_df_scores works
    strs = np.array([[
        u'{val}\u00A0±\u00A0{std}'.format(
            val=round(mean, score_type.precision),
            std=round(std, score_type.precision + 1))
        for mean, std, score_type in zip(means, stds, score_types)]
        for means, stds in zip(meanss, stdss)])
    df_scores = pd.DataFrame(
        strs, columns=df_scores_list[0].columns, index=df_scores_list[0].index)
    return df_scores


def _score_matrix_from_scores(score_types, steps, scoress):
    """Construct a score dataframe from a matrix of scores.

    Parameters
    ----------
    score_types : list of score types
        a list of score types to use, score_type.name serves as column index
    steps : a list of strings
        subset of ['train', 'valid', 'test'], serves as row index

    Returns
    -------
    df_scores : the score dataframe
    """
    results = []
    for step, scores in zip(steps, scoress):
        for score_type, score in zip(score_types, scores):
            results.append(
                {'step': str(step), 'score': score_type.name, 'value': score})
    df_scores = pd.DataFrame(results)
    df_scores = df_scores.set_index(['step', 'score'])['value']
    df_scores = df_scores.unstack()
    return df_scores


def _score_matrix(score_types, ground_truth, predictions):
    """Construct a score dataframe by scoring predictions against ground truth.

    Parameters
    ----------
    score_types : list of score types
        a list of score types to use, score_type.name serves as column index
    ground_truth : dict of Predictions
        the ground truth data
    predictions : dict of Predictions
        the predicted data

    Returns
    -------
    df_scores : pd.DataFrame
        table of scores (rows = train/valid/test steps, columns = scores)
    """
    if set(ground_truth.keys()) != set(predictions.keys()):
        raise ValueError(('Predictions and ground truth steps '
                          'do not match:\n'
                          ' * predictions = {} \n'
                          ' * ground_truth = {} ')
                         .format(set(predictions.keys()),
                                 set(ground_truth.keys())))
    steps = ground_truth.keys()
    scoress = [[
        score_type.score_function(ground_truth[step], predictions[step])
        for score_type in score_types] for step in ground_truth]
    return _score_matrix_from_scores(score_types, steps, scoress)


def _round_df_scores(df_scores, score_types):
    """Round scores to the precision set in the score type.

    Parameters
    ----------
    df_scores : pd.DataFrame
        the score dataframe
    score_types : list of score types

    Returns
    -------
    df_scores : the dataframe with rounded scores
    """
    df_scores_copy = df_scores.copy()
    for column, score_type in zip(df_scores_copy, score_types):
        df_scores_copy[column] = [round(score, score_type.precision)
                                  for score in df_scores_copy[column]]
    return df_scores_copy


def _print_df_scores(df_scores, score_types, indent=''):
    """Pretty print the scores dataframe.

    Parameters
    ----------
    df_scores : pd.DataFrame
        the score dataframe
    score_types : list of score types
        a list of score types to use
    indent : str, default=''
        indentation if needed
    """
    try:
        # try to re-order columns/rows in the printed array
        # we may not have all train, valid, test, so need to select
        index_order = np.array(['train', 'valid', 'test'])
        ordered_index = index_order[np.isin(index_order, df_scores.index)]
        df_scores = df_scores.loc[
            ordered_index, [score_type.name for score_type in score_types]]
    except Exception:
        _print_warning("Couldn't re-order the score matrix..")
    with pd.option_context("display.width", 160):
        df_repr = repr(df_scores)
    df_repr_out = []
    for line, color_key in zip(df_repr.splitlines(),
                               [None, None] +
                               list(df_scores.index.values)):
        if line.strip() == 'step':
            continue
        if color_key is None:
            # table header
            line = stylize(line, fg(fg_colors['title']) + attr('bold'))
        if color_key is not None:
            tokens = line.split()
            tokens_bak = tokens[:]
            if 'official_' + color_key in fg_colors:
                # line label and official score bold & bright
                label_color = fg(fg_colors['official_' + color_key])
                tokens[0] = stylize(tokens[0], label_color + attr('bold'))
                tokens[1] = stylize(tokens[1], label_color + attr('bold'))
            if color_key in fg_colors:
                # other scores pale
                tokens[2:] = [stylize(token, fg(fg_colors[color_key]))
                              for token in tokens[2:]]
            for token_from, token_to in zip(tokens_bak, tokens):
                line = line.replace(token_from, token_to)
        line = indent + line
        df_repr_out.append(line)
    print('\n'.join(df_repr_out))


def _save_y_pred(problem, y_pred, data_path='.', output_path='.',
                 suffix='test'):
    """Save a prediction vector in file.

    If problem.save_y_pred is implemented, y_pred is passed to it. Otherwise,
    np.savez_compressed is used on y_pred. If it crashes, a warning is raised.
    The file is (typically) in
    submissions/<submission_name>/training_output/y_pred_<suffix>.npz or
    submissions/<submission_name>/training_output/fold_<i>/y_pred_<suffix>.npz.

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
        the directory where (typically) y_pred_<suffix>.npz will be saved
    suffix : str, (default='test')
        suffix in (typically) y_pred_<suffix>.npz, can be used in
        problem.save_y_pred to, e.g., save only test predictions
    """
    try:
        # We try using custom made problem.save_y_pred
        # obligatory to implement if np.savez_compressed doesn't work on y_pred
        problem.save_y_pred(y_pred, data_path, output_path, suffix)
    except AttributeError:
        # We fall back to numpy savez_compressed
        try:
            y_pred_f_name = join(output_path, 'y_pred_{}'.format(suffix))
            np.savez_compressed(y_pred_f_name, y_pred=y_pred)
        except Exception as e:
            _print_warning(
                "Warning: model can't be saved.\n{}\n".format(e) +
                'Consider implementing custom save_y_pred in problem.py\n')


def _load_y_pred(problem, data_path='.', input_path='.', suffix='test'):
    """Load a file into a prediction vector.

    If problem.load_y_pred is implemented, y_pred is loaded by it. Otherwise,
    np.load is used. If it crashes, the exception is raised. The input file
    is (typically) in
    submissions/<submission_name>/training_output/y_pred_<suffix>.npz or
    submissions/<submission_name>/training_output/fold_<i>/y_pred_<suffix>.npz.

    Parameters
    ----------
    problem : a problem object
        loaded from problem.py, may implement save_y_pred
    data_path : str, (default='.')
        the directory of the ramp-kit to be tested for submission, maybe
        needed by problem.save_y_pred for, e.g., merging with an index vector
    input_path : str, (default='.')
        the directory where (typically) y_pred_<suffix>.npz will be saved
    suffix : str, (default='test')
        suffix in (typically) y_pred_<suffix>.npz
    """
    try:
        # We try using custom made problem.load_y_pred
        # obligatory to implement if np.load doesn't work
        return problem.load_y_pred(data_path, input_path, suffix)
    except AttributeError:
        # We fall back to numpy load
        y_pred_f_name = join(input_path, 'y_pred_{}.npz'.format(suffix))
        return np.load(y_pred_f_name)['y_pred']


def _save_submission(problem, y_pred, data_path='.', output_path='.',
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


def _pickle_model(fold_output_path, trained_workflow, model_name='model.pkl'):
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
        model_file = join(fold_output_path, model_name)
        with open(model_file, 'wb') as pickle_file:
            pickle.dump(trained_workflow, pickle_file)
        with open(model_file, 'r') as pickle_file:
            trained_workflow = pickle.load(pickle_file)
    except Exception as e:
        _print_warning("Warning: model can't be pickled.")
        _print_warning(e)
    return trained_workflow


def _train_test_submission(problem, module_path, X_train, y_train, X_test,
                           is_pickle, output_path,
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
    X_train : a list of testing instances
        returned by problem.get_test_data
    is_pickle : boolean
        True if the model should be pickled
    output_path : str
        the path into which the model will be pickled
    model_name : str (default='model.pkl')
        the file name of the pickled workflow
    train_is : a list of integers (default=None)
        training indices from the cross-validation fold, if None, train
        on full set
    Returns
    -------
    y_pred_train : a list of predictions
        on the training (train_train and train_valid) set
    y_pred_test : a list of predictions
        on the test set
    """
    trained_workflow = problem.workflow.train_submission(
        module_path, X_train, y_train, train_is=train_is)
    if is_pickle:
        trained_workflow = _pickle_model(
            output_path, trained_workflow, model_name)

    y_pred_train = problem.workflow.test_submission(
        trained_workflow, X_train)
    y_pred_test = problem.workflow.test_submission(
        trained_workflow, X_test)
    return y_pred_train, y_pred_test


def _run_submission_on_cv_fold(problem, module_path, X_train, y_train,
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
    y_test : a list of testing ground truth
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
    y_pred_train, y_pred_test = _train_test_submission(
        problem, module_path, X_train, y_train, X_test, is_pickle,
        fold_output_path, train_is=train_is)
    predictions_train_train = problem.Predictions(
        y_pred=y_pred_train[train_is])
    ground_truth_train_train = problem.Predictions(
        y_true=y_train[train_is])
    predictions_train_valid = problem.Predictions(
        y_pred=y_pred_train[valid_is])
    ground_truth_train_valid = problem.Predictions(
        y_true=y_train[valid_is])
    predictions_test = problem.Predictions(y_pred=y_pred_test)
    ground_truth_test = problem.Predictions(y_true=y_test)

    if save_y_preds:
        _save_y_pred(
            problem, y_pred_train, data_path=ramp_data_dir,
            output_path=fold_output_path, suffix='train')
        _save_y_pred(
            problem, y_pred_test, data_path=ramp_data_dir,
            output_path=fold_output_path, suffix='test')

    df_scores = _score_matrix(
        score_types,
        ground_truth=OrderedDict([('train', ground_truth_train_train),
                                  ('valid', ground_truth_train_valid),
                                  ('test', ground_truth_test)]),
        predictions=OrderedDict([('train', predictions_train_train),
                                 ('valid', predictions_train_valid),
                                 ('test', predictions_test)]),
    )
    return predictions_train_valid, predictions_test, df_scores


def _run_submission_on_full_train(problem, module_path, X_train, y_train,
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
    X_train : a list of testing instances
        returned by problem.get_test_data
    y_test : a list of testing ground truth
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
    y_pred_train, y_pred_test = _train_test_submission(
        problem, module_path, X_train, y_train, X_test, is_pickle,
        output_path, model_name='retrained_model.pkl')
    predictions_train = problem.Predictions(y_pred=y_pred_train)
    ground_truth_train = problem.Predictions(y_true=y_train)
    predictions_test = problem.Predictions(y_pred=y_pred_test)
    ground_truth_test = problem.Predictions(y_true=y_test)

    df_scores = _score_matrix(
        score_types,
        ground_truth=OrderedDict([('train', ground_truth_train),
                                  ('test', ground_truth_test)]),
        predictions=OrderedDict([('train', predictions_train),
                                 ('test', predictions_test)]),
    )
    df_scores_rounded = _round_df_scores(df_scores, score_types)
    _print_df_scores(df_scores_rounded, score_types, indent='\t')

    if save_y_preds:
        _save_submission(
            problem, y_pred_train, data_path=ramp_data_dir,
            output_path=output_path, suffix='retrain_train')
        _save_submission(
            problem, y_pred_test, data_path=ramp_data_dir,
            output_path=output_path, suffix='retrain_test')


def _bag_submissions(problem, cv, y_train, y_test, predictions_valid_list,
                     predictions_test_list, training_output_path,
                     ramp_data_dir='.', score_type_index=0,
                     save_y_preds=False, score_table_title='Bagged scores'):
    _print_title('----------------------------')
    _print_title(score_table_title)
    _print_title('----------------------------')
    valid_is_list = [valid_is for (train_is, valid_is) in cv]
    ground_truths_train = problem.Predictions(y_true=y_train)
    ground_truths_test = problem.Predictions(y_true=y_test)
    score_type = problem.score_types[score_type_index]
    bagged_valid_predictions, bagged_valid_scores =\
        get_score_cv_bags(
            score_type, predictions_valid_list,
            ground_truths_train, test_is_list=valid_is_list)
    bagged_test_predictions, bagged_test_scores = get_score_cv_bags(
        score_type, predictions_test_list, ground_truths_test)

    df_scores = _score_matrix_from_scores(
        [score_type], ['valid', 'test'],
        [[bagged_valid_scores[-1]], [bagged_test_scores[-1]]])
    df_scores_rounded = _round_df_scores(df_scores, [score_type])
    _print_df_scores(df_scores_rounded, [score_type], indent='\t')

    if save_y_preds:
        # y_pred_bagged_train.csv contains _out of sample_ (validation)
        # predictions, but not for all points (contains nans)
        _save_submission(
            problem, bagged_valid_predictions.y_pred,
            data_path=ramp_data_dir, output_path=training_output_path,
            suffix='bagged_train')
        _save_submission(
            problem, bagged_test_predictions.y_pred, data_path=ramp_data_dir,
            output_path=training_output_path, suffix='bagged_test')
        # also save the partial combined scores (CV bagging learning curves)
        bagged_train_valid_scores_f_name = join(
            training_output_path, 'bagged_valid_scores.csv')
        np.savetxt(bagged_train_valid_scores_f_name, bagged_valid_scores)
        bagged_test_scores_f_name = join(
            training_output_path, 'bagged_test_scores.csv')
        np.savetxt(bagged_test_scores_f_name, bagged_test_scores)


def _pickle_model(fold_output_path, trained_workflow, model_name='model.pkl'):
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
        model_file = join(fold_output_path, model_name)
        with open(model_file, 'wb') as pickle_file:
            pickle.dump(trained_workflow, pickle_file)
        with open(model_file, 'r') as pickle_file:
            trained_workflow = pickle.load(pickle_file)
    except Exception as e:
        _print_warning("Warning: model can't be pickled.")
        _print_warning(e)
    return trained_workflow


def _train_test_submission(problem, module_path, X_train, y_train, X_test,
                           is_pickle, output_path,
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
    X_train : a list of testing instances
        returned by problem.get_test_data
    is_pickle : boolean
        True if the model should be pickled
    output_path : str
        the path into which the model will be pickled
    model_name : str (default='model.pkl')
        the file name of the pickled workflow
    train_is : a list of integers (default=None)
        training indices from the cross-validation fold, if None, train
        on full set
    Returns
    -------
    y_pred_train : a list of predictions
        on the training (train_train and train_valid) set
    y_pred_test : a list of predictions
        on the test set
    """
    trained_workflow = problem.workflow.train_submission(
        module_path, X_train, y_train, train_is=train_is)
    if is_pickle:
        trained_workflow = _pickle_model(
            output_path, trained_workflow, model_name)

    y_pred_train = problem.workflow.test_submission(
        trained_workflow, X_train)
    y_pred_test = problem.workflow.test_submission(
        trained_workflow, X_test)
    return y_pred_train, y_pred_test


def _run_submission_on_cv_fold(problem, module_path, X_train, y_train,
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
    y_test : a list of testing ground truth
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
    y_pred_train, y_pred_test = _train_test_submission(
        problem, module_path, X_train, y_train, X_test, is_pickle,
        fold_output_path, train_is=train_is)
    predictions_train_train = problem.Predictions(
        y_pred=y_pred_train[train_is])
    ground_truth_train_train = problem.Predictions(
        y_true=y_train[train_is])
    predictions_train_valid = problem.Predictions(
        y_pred=y_pred_train[valid_is])
    ground_truth_train_valid = problem.Predictions(
        y_true=y_train[valid_is])
    predictions_test = problem.Predictions(y_pred=y_pred_test)
    ground_truth_test = problem.Predictions(y_true=y_test)

    if save_y_preds:
        _save_y_pred(
            problem, y_pred_train, data_path=ramp_data_dir,
            output_path=fold_output_path, suffix='train')
        _save_y_pred(
            problem, y_pred_test, data_path=ramp_data_dir,
            output_path=fold_output_path, suffix='test')

    df_scores = _score_matrix(
        score_types,
        ground_truth=OrderedDict([('train', ground_truth_train_train),
                                  ('valid', ground_truth_train_valid),
                                  ('test', ground_truth_test)]),
        predictions=OrderedDict([('train', predictions_train_train),
                                 ('valid', predictions_train_valid),
                                 ('test', predictions_test)]),
    )
    return predictions_train_valid, predictions_test, df_scores


def _run_submission_on_full_train(problem, module_path, X_train, y_train,
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
    X_train : a list of testing instances
        returned by problem.get_test_data
    y_test : a list of testing ground truth
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
    y_pred_train, y_pred_test = _train_test_submission(
        problem, module_path, X_train, y_train, X_test, is_pickle,
        output_path, model_name='retrained_model.pkl')
    predictions_train = problem.Predictions(y_pred=y_pred_train)
    ground_truth_train = problem.Predictions(y_true=y_train)
    predictions_test = problem.Predictions(y_pred=y_pred_test)
    ground_truth_test = problem.Predictions(y_true=y_test)

    df_scores = _score_matrix(
        score_types,
        ground_truth=OrderedDict([('train', ground_truth_train),
                                  ('test', ground_truth_test)]),
        predictions=OrderedDict([('train', predictions_train),
                                 ('test', predictions_test)]),
    )
    df_scores_rounded = _round_df_scores(df_scores, score_types)
    _print_df_scores(df_scores_rounded, score_types, indent='\t')

    if save_y_preds:
        _save_y_pred(
            problem, y_pred_train, data_path=ramp_data_dir,
            output_path=output_path, suffix='retrain_train')
        _save_y_pred(
            problem, y_pred_test, data_path=ramp_data_dir,
            output_path=output_path, suffix='retrain_test')


def assert_submission(ramp_kit_dir='.', ramp_data_dir='.',
                      submission='starting_kit', is_pickle=False,
                      save_y_preds=False, retrain=False):
    """Helper to test a submission from a ramp-kit.

    Parameters
    ----------
    ramp_kit_dir : str, (default='.')
        the directory of the ramp-kit to be tested for submission

    ramp_data_dir : str, (default='.')
        the directory of the data

    submission_name : str, (default='starting_kit')
        the name of the submission to be tested
    """
    problem = assert_read_problem(ramp_kit_dir)
    assert_title(ramp_kit_dir)
    X_train, y_train, X_test, y_test = assert_data(ramp_kit_dir, ramp_data_dir)
    cv = assert_cv(ramp_kit_dir, ramp_data_dir)
    score_types = assert_score_types(ramp_kit_dir)

    module_path = join(ramp_kit_dir, 'submissions', submission)
    _print_title('Training {} ...'.format(module_path))

    training_output_path = ''
    if is_pickle or save_y_preds:
        # creating submissions/<submission>/training_output dir
        training_output_path = join(module_path, 'training_output')
        if not os.path.exists(training_output_path):
            os.mkdir(training_output_path)

    # saving predictions for CV bagging after the CV loop
    predictions_valid_list = []
    predictions_test_list = []
    df_scores_list = []

    for fold_i, fold in enumerate(cv):
        fold_output_path = ''
        if is_pickle or save_y_preds:
            # creating submissions/<submission>/training_output/fold_<i> dir
            fold_output_path = join(
                training_output_path, 'fold_{}'.format(fold_i))
            if not os.path.exists(fold_output_path):
                os.mkdir(fold_output_path)
        _print_title('CV fold {}'.format(fold_i))

        predictions_valid, predictions_test, df_scores =\
            _run_submission_on_cv_fold(
                problem, module_path, X_train, y_train, X_test, y_test,
                score_types, is_pickle, save_y_preds, fold_output_path,
                fold, ramp_data_dir)
        df_scores_rounded = _round_df_scores(df_scores, score_types)
        _print_df_scores(df_scores_rounded, score_types, indent='\t')

        # saving predictions for CV bagging after the CV loop
        df_scores_list.append(df_scores)
        predictions_valid_list.append(predictions_valid)
        predictions_test_list.append(predictions_test)

    _print_title('----------------------------')
    _print_title('Mean CV scores')
    _print_title('----------------------------')
    df_mean_scores = _mean_score_matrix(df_scores_list, score_types)
    _print_df_scores(df_mean_scores, score_types, indent='\t')

    if retrain:
        # We retrain on the full training set
        _print_title('----------------------------')
        _print_title('Retrain scores')
        _print_title('----------------------------')
        _run_submission_on_full_train(
            problem, module_path, X_train, y_train, X_test, y_test,
            score_types, is_pickle, save_y_preds, training_output_path,
            ramp_data_dir)
    _bag_submissions(
        problem, cv, y_train, y_test, predictions_valid_list,
        predictions_test_list, training_output_path,
        ramp_data_dir=ramp_data_dir, score_type_index=0,
        save_y_preds=save_y_preds)


def blend_submissions(submissions, ramp_kit_dir='.', ramp_data_dir='.',
                      save_y_preds=False, min_improvement=0.0):
    problem = assert_read_problem(ramp_kit_dir)
    _print_title('Blending {}'.format(problem.problem_title))
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
        _print_title('CV fold {}'.format(fold_i))
        ground_truths_valid = problem.Predictions(y_true=y_train[valid_is])
        predictions_valid_list = []
        predictions_test_list = []
        for submission in submissions:
            module_path = join(ramp_kit_dir, 'submissions', submission)
            training_output_path = join(module_path, 'training_output')
            fold_output_path = join(
                training_output_path, 'fold_{}'.format(fold_i))
            y_pred_train = _load_y_pred(
                problem, data_path=ramp_data_dir,
                input_path=fold_output_path, suffix='train')
            y_pred_test = _load_y_pred(
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

    training_output_path = join(ramp_kit_dir, 'training_output')
    if not os.path.exists(training_output_path):
        os.mkdir(training_output_path)
    # bagging the foldwise ensembles
    _bag_submissions(
        problem, cv, y_train, y_test, combined_predictions_valid_list,
        combined_predictions_test_list, training_output_path,
        ramp_data_dir=ramp_data_dir, score_type_index=0,
        save_y_preds=save_y_preds, score_table_title='Combined bagged scores')
    # bagging the foldwise best submissions
    _bag_submissions(
        problem, cv, y_train, y_test, foldwise_best_predictions_valid_list,
        foldwise_best_predictions_test_list, training_output_path,
        ramp_data_dir=ramp_data_dir, score_type_index=0,
        save_y_preds=save_y_preds,
        score_table_title='Foldwise best bagged scores')
