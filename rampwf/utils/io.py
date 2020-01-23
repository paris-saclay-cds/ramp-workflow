# coding: utf-8
"""
Utilities for saving and loading predictions
"""
import os
import sys
import traceback as tb
import numpy as np

from .pretty_print import print_warning


def save_y_pred(problem, y_pred, data_path='.', output_path='.',
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
            y_pred_f_name = os.path.join(output_path,
                                         'y_pred_{}'.format(suffix))
            np.savez_compressed(y_pred_f_name, y_pred=y_pred)
        except Exception as e:
            print_warning(
                "Warning: model can't be saved.\n{}\n".format(e) +
                'Consider implementing custom save_y_pred in problem.py\n')


def load_y_pred(problem, data_path='.', input_path='.', suffix='test'):
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
        y_pred_f_name = os.path.join(input_path,
                                     'y_pred_{}.npz'.format(suffix))
        return np.load(y_pred_f_name)['y_pred']


def set_state(state, save_output, output_path):
    """Save the submission state (per fold) in <output_path>/state.txt.

    In case save_output is True. Otherwise do nothing.

    Parameters
    ----------
    state : str
        The string representing the state. Possible values are 'new',
        'trained', 'validated', 'tested', 'scored', 'training_error',
        'validating_error', 'testing_error'.
    save_output : boolean
        True if state should be written in file
    output_path : str
        the path into which 'state.txt' will be saved
    """
    if save_output:
        with open(os.path.join(output_path, 'state.txt'), 'w') as fd:
            fd.write(state)


def print_submission_exception(save_output, output_path):
    """Print the exception trace corresponding the user submission.

    In case save_output is True, also save it into <output_path>/error.txt

    Parameters
    ----------
    save_output : boolean
        True if error should be written in file
    output_path : str
        the path into which 'error.txt' will be saved
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    # print the stack corresponding to user submission
    trace = tb.format_exception(exc_type, exc_value, exc_traceback)
    for s in trace:
        print(s)
    # some times traces is <5
    try:
        if trace[4].find('load_source') > -1:
            trace = trace[5:]
    except IndexError:
        trace = trace[4:]
    if save_output:
        with open(os.path.join(output_path, 'error.txt'), 'w') as fd:
            for s in trace:
                fd.write(s)
